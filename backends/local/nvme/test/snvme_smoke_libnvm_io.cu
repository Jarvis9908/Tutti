/*
 * snvme_smoke_libnvm_io.cu -- L1 Commit 3 smoke: GPU-resident NVMe IO
 * with bring-up and map registration driven entirely through libnvm
 * Commit 1+2 wrappers.
 *
 * Layer scope
 * -----------
 *
 * This is the first L1 smoke that actually issues Read/Write commands
 * end-to-end on a queue brought up via libnvm:
 *
 *   bring-up         nvm_controller_init_b3(cap=0)
 *                    nvm_create_group()
 *                    nvm_dma_map_ring_device(SQ, gid)   -- libnvm DMA API
 *                    nvm_dma_map_ring_device(CQ, gid)
 *                    nvm_dma_map_data_device(wbuf)      -- fd-scoped DATA
 *                    nvm_dma_map_data_device(rbuf)
 *                    nvm_add_user_queue()               -- one-shot batch
 *
 *   IO loop          GPU CUDA kernels build SQE + ring SQ doorbell, then
 *                    spin-poll CQE phase bit.  Identical machinery to
 *                    snvme_smoke_gpu.cu (L0); only the bring-up moved to
 *                    libnvm.
 *
 *   tear-down        nvm_destroy_group()                -- cascades RING_*
 *                    nvm_dma_unmap(wbuf) + (rbuf)       -- DATA explicit
 *                    _nvm_ctrl_put()                    -- skip unbind
 *                    SNVM_DEVICE_UNBIND + CHRDEV_REMOVE -- raw owner ioctl
 *
 * Differences vs the L0 GPU IO smoke (snvme_smoke_gpu.cu):
 *
 *   - bring-up + group + DMA registration go through libnvm wrappers,
 *     not raw ioctl.  Doorbell offsets come from
 *     nvm_ioctl_add_user_queue.out_pairs[i] (kernel-returned), not
 *     recomputed from QID via SQ_DBL/CQ_DBL macros.
 *
 *   - One queue, one tier (PRP1, 4 KiB).  The exhaustive PRP2 / PRP_List
 *     / SGL / SQ-tail-wrap matrix already lives in L0 -- this smoke is a
 *     correctness probe for libnvm's wrappers, not a re-implementation
 *     of that matrix.
 *
 *   - One round.  L0 already verifies destroy-group -> recreate
 *     repeatedly across 20 rounds; here we only need to prove the
 *     single happy path through libnvm works.
 *
 * DESTRUCTIVE: writes TEST_NR_IO LBAs at TEST_LBA_BASE.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ioctl.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>

/* Internal libnvm refcount drop (no unbind cascade).  Same workaround
 * as the other L1 smokes; obsolete once Commit 4 splits owner/client
 * roles. */
struct controller;
extern "C" void _nvm_ctrl_put(struct controller* ctrl);

/* ------------------------------------------------------------------ */
/* Test parameters                                                    */
/* ------------------------------------------------------------------ */

#define NVME_OPC_WRITE              0x01u
#define NVME_OPC_READ               0x02u
#define NVME_SQE_SIZE               64u
#define NVME_CQE_SIZE               16u
#define NVME_FLAG_PSDT_PRP          (0u << 6)

#define TEST_LBA_BASE               2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_IO                  8u
#define TEST_KERNEL_IOQ_CAP         36u

/* GPU page size used by the snvme kernel module (GPU_PAGE_SHIFT=16).
 * Buffers passed to NVM_MAP_DEVICE_MEMORY MUST be sized in multiples
 * of this. */
static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;     /* 64 KiB */

#define WRITE_PATTERN_BYTE(ioidx) \
    ((uint8_t)(0x5C ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* NVMe SQE / CQE                                                     */
/* ------------------------------------------------------------------ */

struct nvme_sqe {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    uint32_t nsid;
    uint64_t rsvd_2_3;
    uint64_t metadata;
    uint64_t prp1;
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE,
              "nvme_sqe must be exactly 64 bytes");

struct nvme_cqe {
    uint32_t result;
    uint32_t rsvd;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t cid;
    uint16_t status;
} __attribute__((packed));

static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE,
              "nvme_cqe must be exactly 16 bytes");

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

#define CUDA_OK(call_)                                                  \
    do {                                                                \
        cudaError_t _e = (call_);                                       \
        if (_e != cudaSuccess) {                                        \
            step_fail(0, "CUDA: %s -> %s", #call_, cudaGetErrorString(_e)); \
        }                                                               \
    } while (0)

static void format_status(uint16_t status, char* buf, size_t cap) {
    uint16_t s   = status >> 1;
    uint8_t  sc  = s & 0xff;
    uint8_t  sct = (s >> 8) & 0x7;
    snprintf(buf, cap, "0x%04x (SC=0x%02x SCT=0x%x)", status, sc, sct);
}

/* ------------------------------------------------------------------ */
/* Per-queue runtime state shared with GPU kernels                    */
/* ------------------------------------------------------------------ */

struct test_queue_dev {
    nvme_sqe*           sq;             /* device VA */
    nvme_cqe*           cq;             /* device VA */
    volatile uint32_t*  sq_db;          /* GPU VA into BAR0 */
    volatile uint32_t*  cq_db;          /* GPU VA into BAR0 */
    uint16_t            q_depth;
    uint16_t            qid;
};

/* ------------------------------------------------------------------ */
/* GPU kernels (verbatim from L0)                                     */
/* ------------------------------------------------------------------ */

__global__ void k_submit_rw(test_queue_dev qd,
                            uint16_t* sq_tail_io,
                            uint16_t cid,
                            uint8_t opcode, uint8_t flags, uint32_t nsid,
                            uint64_t dptr0, uint64_t dptr1,
                            uint64_t slba, uint16_t nlb_zero_based) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint16_t tail = *sq_tail_io;
    nvme_sqe* slot = &qd.sq[tail];
    uint8_t* p = (uint8_t*)slot;
    #pragma unroll
    for (int i = 0; i < (int)sizeof(nvme_sqe); i++) p[i] = 0;
    slot->opcode = opcode;
    slot->flags  = flags;
    slot->cid    = cid;
    slot->nsid   = nsid;
    slot->prp1   = dptr0;
    slot->prp2   = dptr1;
    slot->cdw10  = (uint32_t)(slba & 0xffffffffu);
    slot->cdw11  = (uint32_t)(slba >> 32);
    slot->cdw12  = (uint32_t)(nlb_zero_based & 0xffffu);
    __threadfence_system();
    uint16_t new_tail = (uint16_t)((tail + 1) % qd.q_depth);
    *qd.sq_db = new_tail;
    *sq_tail_io = new_tail;
}

__global__ void k_poll_one(test_queue_dev qd,
                           uint16_t* cq_head_io,
                           uint8_t* cq_phase_io,
                           nvme_cqe* out_cqe,
                           int* timed_out,
                           uint64_t max_iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint16_t head = *cq_head_io;
    uint8_t  expected = *cq_phase_io;
    uint64_t i = 0;
    for (;;) {
        volatile nvme_cqe* slot = &qd.cq[head];
        uint16_t status = slot->status;
        uint8_t phase = status & 0x1u;
        if (phase == expected) {
            nvme_cqe tmp;
            tmp.result  = slot->result;
            tmp.rsvd    = slot->rsvd;
            tmp.sq_head = slot->sq_head;
            tmp.sq_id   = slot->sq_id;
            tmp.cid     = slot->cid;
            tmp.status  = status;
            *out_cqe = tmp;
            uint16_t new_head = (uint16_t)((head + 1) % qd.q_depth);
            if (new_head == 0) expected ^= 1u;
            __threadfence_system();
            *qd.cq_db = new_head;
            *cq_head_io  = new_head;
            *cq_phase_io = expected;
            *timed_out = 0;
            return;
        }
        if (++i >= max_iters) { *timed_out = 1; return; }
    }
}

__global__ void k_fill_pattern(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    buf[idx] = pat ^ (uint8_t)(idx >> 12);
}

__global__ void k_verify_pattern(const uint8_t* buf, size_t bytes,
                                 uint8_t pat, int* mismatch_idx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) {
        atomicCAS(mismatch_idx, -1, (int)idx);
    }
}

/* ------------------------------------------------------------------ */
/* Host helpers                                                       */
/* ------------------------------------------------------------------ */

struct queue_state {
    test_queue_dev      dev;
    uint16_t*           sq_tail_um;
    uint16_t*           cq_head_um;
    uint8_t*            cq_phase_um;
    nvme_cqe*           out_cqe_um;
    int*                timed_out_um;
    uint16_t            next_cid;
};

static int submit_and_poll(queue_state& qs,
                           uint8_t opcode, uint8_t flags,
                           uint32_t nsid,
                           uint64_t dptr0, uint64_t dptr1,
                           uint64_t slba, uint16_t nlb_zero_based,
                           nvme_cqe* cqe_out, uint16_t* cid_out) {
    uint16_t cid = qs.next_cid++;
    if (cid_out) *cid_out = cid;

    k_submit_rw<<<1, 1>>>(qs.dev, qs.sq_tail_um, cid,
                          opcode, flags, nsid, dptr0, dptr1,
                          slba, nlb_zero_based);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_submit_rw launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }

    constexpr uint64_t MAX_ITERS = 50000000ULL;
    k_poll_one<<<1, 1>>>(qs.dev, qs.cq_head_um, qs.cq_phase_um,
                         qs.out_cqe_um, qs.timed_out_um, MAX_ITERS);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_poll_one launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }
    CUDA_OK(cudaDeviceSynchronize());

    if (*qs.timed_out_um) return -ETIMEDOUT;
    *cqe_out = *qs.out_cqe_um;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--gpu N] <PCI_BDF>\n"
        "  e.g.: %s --gpu 0 0000:08:00.0\n"
        "\n"
        "BINDS the target controller via libnvm (B3) and writes/reads\n"
        "%u LBAs starting at LBA %llu using GPU-resident rings + data.\n"
        "DESTRUCTIVE.\n",
        prog, prog, TEST_NR_IO, (unsigned long long)TEST_LBA_BASE);
}

int main(int argc, char** argv) {
    int cuda_dev = 0;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char* arg = argv[argi];
        if (strcmp(arg, "--gpu") == 0 && argi + 1 < argc) {
            cuda_dev = atoi(argv[++argi]);
            ++argi;
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (argi + 1 != argc) { usage(argv[0]); return 1; }
    const char* pci_addr = argv[argi];

    /* ===== [1] CUDA device ===== */
    {
        int dcount = 0;
        CUDA_OK(cudaGetDeviceCount(&dcount));
        if (cuda_dev < 0 || cuda_dev >= dcount)
            step_fail(EINVAL, "--gpu %d out of range (0..%d)", cuda_dev, dcount-1);
        CUDA_OK(cudaSetDevice(cuda_dev));
        cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, cuda_dev));
        step_ok("CUDA setDevice(%d) name='%s' cap=%d.%d",
                cuda_dev, prop.name, prop.major, prop.minor);
    }

    /* ===== [2] libnvm B3 bring-up (CHRDEV_CREATE -> CAP -> BIND ->
     *           wait probe -> GET_DEV_INFO -> ctrl_init ->
     *           cudaHostRegister BAR0) ===== */
    nvm_ctrl_t* ctrl = NULL;
    struct disk disk;
    memset(&disk, 0, sizeof(disk));
    int rc = nvm_controller_init_b3(&ctrl,
                                    "/dev/snvm_control",
                                    pci_addr,
                                    TEST_KERNEL_IOQ_CAP,
                                    &disk);
    if (rc != 0) step_fail(rc, "nvm_controller_init_b3");
    if (ctrl == NULL || ctrl->q_depth == 0)
        step_fail(EINVAL, "nvm_controller_init_b3: ctrl/q_depth invalid");
    step_ok("nvm_controller_init_b3 disk='%s' block_size=%zu q_depth=%u "
            "max_user_qid=%u sgls=0x%x",
            disk.disk_name, disk.block_size,
            (unsigned)ctrl->q_depth, ctrl->max_user_qid, ctrl->sgl_supported);

    if (disk.block_size != 4096)
        step_fail(0, "smoke assumes 4 KiB-LBA controller; got %zu",
                  disk.block_size);

    /* ===== [3] BAR0 -> GPU VA  (cudaHostRegister already done by
     *           nvm_controller_init_b3; we just translate to a GPU
     *           pointer for the current device).  Not needed if we
     *           used host doorbells, but matches L0 GPU smoke. ===== */
    void* bar0_gpu = NULL;
    {
        cudaError_t err = cudaHostGetDevicePointer(
            &bar0_gpu, (void*) ctrl->mm_ptr, 0);
        if (err != cudaSuccess)
            step_fail(0, "cudaHostGetDevicePointer(BAR0): %s",
                      cudaGetErrorString(err));
    }
    step_ok("BAR0 GPU VA=%p (translated from ctrl->mm_ptr=%p)",
            bar0_gpu, (void*) ctrl->mm_ptr);

    /* ===== [4] queue group ===== */
    uint32_t gid = 0, max_q = 0;
    rc = nvm_create_group(ctrl, &gid, &max_q);
    if (rc != 0) step_fail(rc, "nvm_create_group");
    step_ok("nvm_create_group gid=%u max_queues=%u", gid, max_q);

    /* ===== [5] cudaMalloc one SQ + one CQ ring (1 GPU page each). ===== */
    void* sq_dev = NULL;
    void* cq_dev = NULL;
    CUDA_OK(cudaMalloc(&sq_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&cq_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(sq_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(cq_dev, 0, GPU_PAGE_SIZE));
    step_ok("cudaMalloc rings: SQ=%p CQ=%p (%zu B each)",
            sq_dev, cq_dev, GPU_PAGE_SIZE);

    /* ===== [6] register rings via libnvm ring_device API.  Kernel
     *           tags them as RING_SQ / RING_CQ on group g.  Released
     *           by nvm_destroy_group cascade. ===== */
    nvm_dma_t* dma_sq = NULL;
    nvm_dma_t* dma_cq = NULL;
    rc = nvm_dma_map_ring_device(&dma_sq, ctrl, gid, sq_dev, GPU_PAGE_SIZE,
                                 /*is_cq=*/0);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_device SQ");
    rc = nvm_dma_map_ring_device(&dma_cq, ctrl, gid, cq_dev, GPU_PAGE_SIZE,
                                 /*is_cq=*/1);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_device CQ");
    step_ok("nvm_dma_map_ring_device gid=%u SQ.ioaddr=0x%" PRIx64
            " CQ.ioaddr=0x%" PRIx64,
            gid,
            (uint64_t)dma_sq->ioaddrs[0],
            (uint64_t)dma_cq->ioaddrs[0]);

    /* ===== [7] cudaMalloc + register data buffers (DATA, fd-scoped).
     *           One 64 KiB write buffer + one 64 KiB read buffer so the
     *           test can fit TEST_NR_IO * 4 KiB without overlap.  ===== */
    void* wbuf_dev = NULL;
    void* rbuf_dev = NULL;
    CUDA_OK(cudaMalloc(&wbuf_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&rbuf_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(wbuf_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(rbuf_dev, 0, GPU_PAGE_SIZE));

    nvm_dma_t* dma_wbuf = NULL;
    nvm_dma_t* dma_rbuf = NULL;
    rc = nvm_dma_map_data_device(&dma_wbuf, ctrl, wbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) step_fail(rc, "nvm_dma_map_data_device wbuf");
    rc = nvm_dma_map_data_device(&dma_rbuf, ctrl, rbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) step_fail(rc, "nvm_dma_map_data_device rbuf");
    step_ok("nvm_dma_map_data_device wbuf.ioaddr=0x%" PRIx64
            " rbuf.ioaddr=0x%" PRIx64 " (kind=DATA, fd-scoped)",
            (uint64_t)dma_wbuf->ioaddrs[0],
            (uint64_t)dma_rbuf->ioaddrs[0]);

    /* ===== [8] add user queue (one pair).  Kernel issues
     *           Create I/O CQ + Create I/O SQ admin commands and
     *           returns the BAR0 doorbell offsets. ===== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = gid;
    add_req.nr_pairs = 1;
    add_req.pairs[0].sq_vaddr = (uint64_t)(uintptr_t) sq_dev;
    add_req.pairs[0].cq_vaddr = (uint64_t)(uintptr_t) cq_dev;
    rc = nvm_add_user_queue(ctrl, &add_req);
    if (rc != 0) step_fail(rc, "nvm_add_user_queue");
    step_ok("nvm_add_user_queue gid=%u qid=%u sq_db=0x%x cq_db=0x%x",
            gid, add_req.out_pairs[0].qid,
            add_req.out_pairs[0].sq_doorbell_offset,
            add_req.out_pairs[0].cq_doorbell_offset);

    /* ===== [9] build queue_state for the GPU kernels ===== */
    queue_state qs;
    memset(&qs, 0, sizeof(qs));
    qs.dev.sq      = (nvme_sqe*) sq_dev;
    qs.dev.cq      = (nvme_cqe*) cq_dev;
    qs.dev.q_depth = ctrl->q_depth;
    qs.dev.qid     = (uint16_t) add_req.out_pairs[0].qid;
    qs.dev.sq_db   = (volatile uint32_t*)
        ((char*)bar0_gpu + add_req.out_pairs[0].sq_doorbell_offset);
    qs.dev.cq_db   = (volatile uint32_t*)
        ((char*)bar0_gpu + add_req.out_pairs[0].cq_doorbell_offset);

    CUDA_OK(cudaMallocManaged(&qs.sq_tail_um,   sizeof(uint16_t)));
    CUDA_OK(cudaMallocManaged(&qs.cq_head_um,   sizeof(uint16_t)));
    CUDA_OK(cudaMallocManaged(&qs.cq_phase_um,  sizeof(uint8_t)));
    CUDA_OK(cudaMallocManaged(&qs.out_cqe_um,   sizeof(nvme_cqe)));
    CUDA_OK(cudaMallocManaged(&qs.timed_out_um, sizeof(int)));
    *qs.sq_tail_um   = 0;
    *qs.cq_head_um   = 0;
    *qs.cq_phase_um  = 1;
    *qs.timed_out_um = 0;
    qs.next_cid = 0;
    step_ok("queue_state ready q_depth=%u qid=%u",
            qs.dev.q_depth, qs.dev.qid);

    /* ===== [10] write+read+verify TEST_NR_IO LBAs ===== */
    {
        uint32_t nsid = 1;
        const uint16_t LBA_PER_IO = 1;
        char status_buf[64];
        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < TEST_NR_IO; i++) {
            uint64_t lba = 2*TEST_LBA_BASE + i;
            uint8_t pat  = WRITE_PATTERN_BYTE(i);

            /* Fill 4 KiB slice of wbuf at offset i*4096 with the
             * pattern, then point PRP1 at it. */
            uint8_t* wslice =
                (uint8_t*)wbuf_dev + i * (size_t)disk.block_size;
            uint64_t wslice_io =
                (uint64_t)dma_wbuf->ioaddrs[0] + i * (uint64_t)disk.block_size;
            uint8_t* rslice =
                (uint8_t*)rbuf_dev + i * (size_t)disk.block_size;
            uint64_t rslice_io =
                (uint64_t)dma_rbuf->ioaddrs[0] + i * (uint64_t)disk.block_size;

            const size_t BLOCK = disk.block_size;
            const int    THR   = 256;
            const int    BLOCKS = (BLOCK + THR - 1) / THR;
            k_fill_pattern<<<BLOCKS, THR>>>(wslice, BLOCK, pat);
            CUDA_OK(cudaDeviceSynchronize());

            /* Write */
            nvme_cqe cqe;
            uint16_t cid;
            rc = submit_and_poll(qs, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                                 wslice_io, /*prp2=*/0,
                                 lba, /*nlb_zb=*/(uint16_t)(LBA_PER_IO - 1),
                                 &cqe, &cid);
            if (rc != 0)
                step_fail(-rc, "Write IO %u (lba=%" PRIu64 ") submit/poll",
                          i, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "Write IO %u (lba=%" PRIu64 ") cqe.status=%s",
                          i, lba, status_buf);
            }
            if (cqe.cid != cid)
                step_fail(0, "Write IO %u: cqe.cid=%u != expected %u",
                          i, cqe.cid, cid);

            /* Read into rslice */
            CUDA_OK(cudaMemset(rslice, 0xff, BLOCK));
            rc = submit_and_poll(qs, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                                 rslice_io, 0, lba, 0, &cqe, &cid);
            if (rc != 0)
                step_fail(-rc, "Read IO %u (lba=%" PRIu64 ") submit/poll",
                          i, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "Read IO %u (lba=%" PRIu64 ") cqe.status=%s",
                          i, lba, status_buf);
            }

            /* Verify rslice byte-for-byte against the same pattern. */
            *mismatch_um = -1;
            k_verify_pattern<<<BLOCKS, THR>>>(rslice, BLOCK, pat,
                                              mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um >= 0)
                step_fail(0, "IO %u (lba=%" PRIu64 ") mismatch at byte %d",
                          i, lba, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("Write+Read+verify x %u IOs at LBA [%llu..%llu] "
                "(GPU rings + GPU data, libnvm B3 path)",
                TEST_NR_IO,
                (unsigned long long)TEST_LBA_BASE,
                (unsigned long long)(TEST_LBA_BASE + TEST_NR_IO - 1));
    }

    /* ===== [11] tear-down ===== */

    cudaFree(qs.sq_tail_um);
    cudaFree(qs.cq_head_um);
    cudaFree(qs.cq_phase_um);
    cudaFree(qs.out_cqe_um);
    cudaFree(qs.timed_out_um);

    /* Drop ring DMA handles; their deleters call nvm_dma_unmap (no-op
     * kernel-side under B6 -- group cascade owns the maps) and
     * cudaFree the GPU pages. */
    nvm_dma_unmap(dma_sq);
    nvm_dma_unmap(dma_cq);
    cudaFree(sq_dev);
    cudaFree(cq_dev);
    step_ok("dropped ring DmaPtrs + cudaFree'd ring pages");

    /* Destroy queue group: cascade-deletes I/O CQ+SQ on the controller,
     * frees the user QID, drops every RING_* map attached to the group. */
    rc = nvm_destroy_group(ctrl, gid);
    if (rc != 0) step_fail(rc, "nvm_destroy_group");
    step_ok("nvm_destroy_group gid=%u (kernel: drained 2 ring map(s))", gid);

    /* DATA buffers survived destroy_group (fd-scoped); release explicitly. */
    nvm_dma_unmap(dma_wbuf);
    nvm_dma_unmap(dma_rbuf);
    cudaFree(wbuf_dev);
    cudaFree(rbuf_dev);
    step_ok("nvm_dma_unmap data buffers + cudaFree (DATA fd-scoped)");

    /* Drop libnvm ctrl ref WITHOUT nvm_ctrl_free's unbind cascade. */
    {
        struct controller* c =
            (struct controller*) ctrl_to_controller(ctrl);
        if (c != NULL) _nvm_ctrl_put(c);
    }
    step_ok("libnvm ctrl ref dropped (skipping nvm_ctrl_free's unbind)");

    /* Owner-only teardown: re-open snvm_control + raw UNBIND + CHRDEV_REMOVE.
     * Same workaround as snvme_smoke_libnvm_b3; obsolete once owner/client
     * roles are split (Commit 4). */
    int fd_ctrl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctrl < 0) step_fail(errno, "re-open /dev/snvm_control");

    struct pci_device_addr orig_bdf;
    if (sscanf(pci_addr, "%x:%x:%x.%x",
               &orig_bdf.domain, &orig_bdf.bus,
               &orig_bdf.slot, &orig_bdf.func) != 4)
        step_fail(EINVAL, "parse_bdf");

    {
        struct pci_device_addr a = orig_bdf;
        if (ioctl(fd_ctrl, SNVM_DEVICE_UNBIND, &a) < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND");
        step_ok("SNVM_DEVICE_UNBIND %s", pci_addr);
    }
    {
        struct pci_device_addr a = orig_bdf;
        if (ioctl(fd_ctrl, SNVM_CHRDEV_REMOVE, &a) < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE");
        step_ok("SNVM_CHRDEV_REMOVE %s", pci_addr);
    }
    close(fd_ctrl);

    fprintf(stderr,
            "\n=== snvme_smoke_libnvm_io: all %d steps passed ===\n", g_step);
    return 0;
}
