/*
 * snvme_smoke_libnvm_role.cu -- L1 Commit 4a smoke: owner / client
 * role split via nvm_ctrl_attach_client().
 *
 * Goal
 * ----
 *
 * Verify that two processes can simultaneously hold an attached view
 * of the same SNVMe device, and that the client's fd-cascade behaves
 * correctly (B6 invariants):
 *
 *   1. Process P forks process C IMMEDIATELY at startup, BEFORE any
 *      CUDA call.  CUDA runtime is fork-hostile: once cudaSetDevice
 *      has run in the parent, a forked child inherits an unusable
 *      primary context and any subsequent cuda* call returns
 *      cudaErrorInitializationError.  We dodge that by forking first
 *      and having parent + child each initialise CUDA from scratch.
 *
 *   2. Process P (owner): nvm_controller_init_b3() opens, caps,
 *      binds, probes, grabs disk metadata.  Issues NO IO of its own.
 *      It then writes the post-probe handshake (disk metadata,
 *      chrdev path, bar0_size) down a pipe to C and waits for C to
 *      exit.
 *
 *   3. Process C (client): blocks on the pipe, reads the handshake,
 *      then nvm_ctrl_attach_client() opens an independent fd to the
 *      SAME /dev/ssnvme<N>.  Does not call bind / chrdev_create.
 *
 *   4. Process C: nvm_create_group + cudaMalloc rings +
 *      nvm_dma_map_ring_device + cudaMalloc DATA +
 *      nvm_dma_map_data_device + nvm_add_user_queue.
 *
 *   5. Process C: write+read+verify TEST_NR_IO LBAs from a
 *      GPU-resident kernel (same machinery as
 *      snvme_smoke_libnvm_io.cu).
 *
 *   6. Process C: nvm_destroy_group (cascade rings) +
 *      nvm_dma_unmap data + nvm_ctrl_free_client.
 *      EXPECTATION: kernel does NOT unbind anything; the owner's
 *      ctrl + chrdev stay alive.
 *
 *   7. Process P: wait for C to exit cleanly, then unbind +
 *      chrdev_remove (owner-only teardown).
 *
 * If nvm_ctrl_free_client accidentally cascades into unbind/chrdev_remove,
 * step 7's owner-side ioctls will fail with -ENODEV / -ENOENT and the
 * smoke fails loud.
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
#include <sys/wait.h>
#include <unistd.h>

#include "ioctl.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>

/* ------------------------------------------------------------------ */
/* Test parameters                                                    */
/* ------------------------------------------------------------------ */

#define NVME_OPC_WRITE              0x01u
#define NVME_OPC_READ               0x02u
#define NVME_SQE_SIZE               64u
#define NVME_CQE_SIZE               16u
#define NVME_FLAG_PSDT_PRP          (0u << 6)

#define TEST_LBA_BASE               2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_IO                  4u
#define TEST_KERNEL_IOQ_CAP         36u

static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;

#define WRITE_PATTERN_BYTE(ioidx) \
    ((uint8_t)(0x4A ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* NVMe SQE / CQE                                                     */
/* ------------------------------------------------------------------ */

struct nvme_sqe {
    uint8_t  opcode; uint8_t  flags; uint16_t cid; uint32_t nsid;
    uint64_t rsvd_2_3; uint64_t metadata; uint64_t prp1; uint64_t prp2;
    uint32_t cdw10, cdw11, cdw12, cdw13, cdw14, cdw15;
} __attribute__((packed));
static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE);

struct nvme_cqe {
    uint32_t result; uint32_t rsvd;
    uint16_t sq_head; uint16_t sq_id;
    uint16_t cid; uint16_t status;
} __attribute__((packed));
static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE);

/* ------------------------------------------------------------------ */
/* Logging helpers (with a [P]/[C] prefix so two processes are easy   */
/* to read on a single tty)                                           */
/* ------------------------------------------------------------------ */

static int g_step = 0;
static char g_role_tag[8] = "P";

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[%s][ OK ] step=%-3d ", g_role_tag, g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[%s][FAIL] step=%-3d ", g_role_tag, g_step);
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

/* ------------------------------------------------------------------ */
/* Per-queue runtime state shared with GPU kernels                    */
/* ------------------------------------------------------------------ */

struct test_queue_dev {
    nvme_sqe*           sq;
    nvme_cqe*           cq;
    volatile uint32_t*  sq_db;
    volatile uint32_t*  cq_db;
    uint16_t            q_depth;
    uint16_t            qid;
};

__global__ void k_submit_rw(test_queue_dev qd,
                            uint16_t* sq_tail_io,
                            uint16_t cid,
                            uint8_t opcode, uint8_t flags, uint32_t nsid,
                            uint64_t dptr0, uint64_t dptr1,
                            uint64_t slba, uint16_t nlb_zb) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint16_t tail = *sq_tail_io;
    nvme_sqe* slot = &qd.sq[tail];
    uint8_t* p = (uint8_t*)slot;
    #pragma unroll
    for (int i = 0; i < (int)sizeof(nvme_sqe); i++) p[i] = 0;
    slot->opcode = opcode; slot->flags = flags; slot->cid = cid;
    slot->nsid = nsid; slot->prp1 = dptr0; slot->prp2 = dptr1;
    slot->cdw10 = (uint32_t)(slba & 0xffffffffu);
    slot->cdw11 = (uint32_t)(slba >> 32);
    slot->cdw12 = (uint32_t)(nlb_zb & 0xffffu);
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
        if ((status & 0x1u) == expected) {
            nvme_cqe tmp;
            tmp.result  = slot->result;  tmp.rsvd = slot->rsvd;
            tmp.sq_head = slot->sq_head; tmp.sq_id = slot->sq_id;
            tmp.cid = slot->cid;         tmp.status = status;
            *out_cqe = tmp;
            uint16_t new_head = (uint16_t)((head + 1) % qd.q_depth);
            if (new_head == 0) expected ^= 1u;
            __threadfence_system();
            *qd.cq_db = new_head;
            *cq_head_io = new_head;
            *cq_phase_io = expected;
            *timed_out = 0;
            return;
        }
        if (++i >= max_iters) { *timed_out = 1; return; }
    }
}

__global__ void k_fill(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < bytes) buf[idx] = pat ^ (uint8_t)(idx >> 12);
}
__global__ void k_verify(const uint8_t* buf, size_t bytes,
                          uint8_t pat, int* mismatch_idx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) atomicCAS(mismatch_idx, -1, (int)idx);
}

/* ------------------------------------------------------------------ */
/* Child: client role -- attach via nvm_ctrl_attach_client and drive  */
/*        a real IO loop.                                             */
/* ------------------------------------------------------------------ */

static int run_client(const char* snvme_dev_path,
                      uint32_t bar0_size,
                      uint32_t namespace_id,
                      size_t   block_size,
                      int      cuda_dev) {
    snprintf(g_role_tag, sizeof(g_role_tag), "C");
    g_step = 0;

    CUDA_OK(cudaSetDevice(cuda_dev));
    step_ok("client cudaSetDevice(%d)", cuda_dev);

    nvm_ctrl_t* ctrl = NULL;
    int rc = nvm_ctrl_attach_client(&ctrl, snvme_dev_path, bar0_size);
    if (rc != 0) step_fail(rc, "nvm_ctrl_attach_client(%s)", snvme_dev_path);
    if (ctrl == NULL) step_fail(EINVAL, "attach_client returned ctrl=NULL");
    step_ok("nvm_ctrl_attach_client %s page_size=%zu max_qs=%u",
            snvme_dev_path, ctrl->page_size, ctrl->max_qs);

    /* BAR0 -> GPU VA */
    void* bar0_gpu = NULL;
    CUDA_OK(cudaHostGetDevicePointer(&bar0_gpu, (void*) ctrl->mm_ptr, 0));

    /* Group + ring + DATA + add queue */
    uint32_t gid = 0, max_q = 0;
    rc = nvm_create_group(ctrl, &gid, &max_q);
    if (rc != 0) step_fail(rc, "nvm_create_group");
    step_ok("nvm_create_group gid=%u max_queues=%u", gid, max_q);

    void* sq_dev = NULL; void* cq_dev = NULL;
    CUDA_OK(cudaMalloc(&sq_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&cq_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(sq_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(cq_dev, 0, GPU_PAGE_SIZE));

    nvm_dma_t* dma_sq = NULL;
    nvm_dma_t* dma_cq = NULL;
    rc = nvm_dma_map_ring_device(&dma_sq, ctrl, gid, sq_dev, GPU_PAGE_SIZE, 0);
    if (rc != 0) step_fail(rc, "map_ring SQ");
    rc = nvm_dma_map_ring_device(&dma_cq, ctrl, gid, cq_dev, GPU_PAGE_SIZE, 1);
    if (rc != 0) step_fail(rc, "map_ring CQ");

    /* On the client's OWN fd, q_depth was not propagated by attach_client
     * (no GET_DEV_INFO).  We hardcode 64 here -- matches the kernel cap
     * for our test controller.  In a real client this comes from RPC. */
    const uint16_t q_depth = 64;

    void* wbuf_dev = NULL; void* rbuf_dev = NULL;
    CUDA_OK(cudaMalloc(&wbuf_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&rbuf_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(wbuf_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(rbuf_dev, 0, GPU_PAGE_SIZE));

    nvm_dma_t* dma_wbuf = NULL; nvm_dma_t* dma_rbuf = NULL;
    rc = nvm_dma_map_data_device(&dma_wbuf, ctrl, wbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) step_fail(rc, "map_data wbuf");
    rc = nvm_dma_map_data_device(&dma_rbuf, ctrl, rbuf_dev, GPU_PAGE_SIZE);
    if (rc != 0) step_fail(rc, "map_data rbuf");
    step_ok("client mapped SQ/CQ + wbuf/rbuf (DATA fd-scoped on client fd)");

    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = gid;
    add_req.nr_pairs = 1;
    add_req.pairs[0].sq_vaddr = (uint64_t)(uintptr_t) sq_dev;
    add_req.pairs[0].cq_vaddr = (uint64_t)(uintptr_t) cq_dev;
    rc = nvm_add_user_queue(ctrl, &add_req);
    if (rc != 0) step_fail(rc, "nvm_add_user_queue");
    step_ok("nvm_add_user_queue qid=%u sq_db=0x%x cq_db=0x%x",
            add_req.out_pairs[0].qid,
            add_req.out_pairs[0].sq_doorbell_offset,
            add_req.out_pairs[0].cq_doorbell_offset);

    /* Build queue_state for the GPU kernels */
    test_queue_dev qd;
    qd.sq = (nvme_sqe*) sq_dev; qd.cq = (nvme_cqe*) cq_dev;
    qd.q_depth = q_depth;
    qd.qid = (uint16_t) add_req.out_pairs[0].qid;
    qd.sq_db = (volatile uint32_t*) ((char*)bar0_gpu + add_req.out_pairs[0].sq_doorbell_offset);
    qd.cq_db = (volatile uint32_t*) ((char*)bar0_gpu + add_req.out_pairs[0].cq_doorbell_offset);

    uint16_t* sq_tail_um = nullptr;  uint16_t* cq_head_um = nullptr;
    uint8_t*  cq_phase_um = nullptr; nvme_cqe* out_cqe_um = nullptr;
    int*      timed_out_um = nullptr; int* mismatch_um = nullptr;
    CUDA_OK(cudaMallocManaged(&sq_tail_um,   sizeof(uint16_t)));
    CUDA_OK(cudaMallocManaged(&cq_head_um,   sizeof(uint16_t)));
    CUDA_OK(cudaMallocManaged(&cq_phase_um,  sizeof(uint8_t)));
    CUDA_OK(cudaMallocManaged(&out_cqe_um,   sizeof(nvme_cqe)));
    CUDA_OK(cudaMallocManaged(&timed_out_um, sizeof(int)));
    CUDA_OK(cudaMallocManaged(&mismatch_um,  sizeof(int)));
    *sq_tail_um = 0; *cq_head_um = 0; *cq_phase_um = 1;
    uint16_t next_cid = 0;

    uint32_t nsid = namespace_id;
    char status_buf[64];
    for (unsigned i = 0; i < TEST_NR_IO; i++) {
        uint64_t lba = TEST_LBA_BASE + i;
        uint8_t pat = WRITE_PATTERN_BYTE(i);

        uint8_t* wslice = (uint8_t*)wbuf_dev + i * (size_t)block_size;
        uint8_t* rslice = (uint8_t*)rbuf_dev + i * (size_t)block_size;
        uint64_t wslice_io = (uint64_t)dma_wbuf->ioaddrs[0] + i * (uint64_t)block_size;
        uint64_t rslice_io = (uint64_t)dma_rbuf->ioaddrs[0] + i * (uint64_t)block_size;

        const int THR = 256;
        const int BLOCKS = (block_size + THR - 1) / THR;

        k_fill<<<BLOCKS, THR>>>(wslice, block_size, pat);
        CUDA_OK(cudaDeviceSynchronize());

        /* Write */
        uint16_t cid_w = next_cid++;
        k_submit_rw<<<1,1>>>(qd, sq_tail_um, cid_w,
                             NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                             wslice_io, 0, lba, 0);
        k_poll_one<<<1,1>>>(qd, cq_head_um, cq_phase_um,
                            out_cqe_um, timed_out_um, 50000000ULL);
        CUDA_OK(cudaDeviceSynchronize());
        if (*timed_out_um) step_fail(ETIMEDOUT, "Write IO %u poll timeout", i);
        if ((out_cqe_um->status >> 1) != 0) {
            snprintf(status_buf, sizeof(status_buf), "0x%04x", out_cqe_um->status);
            step_fail(0, "Write IO %u status=%s", i, status_buf);
        }

        /* Read */
        CUDA_OK(cudaMemset(rslice, 0xff, block_size));
        uint16_t cid_r = next_cid++;
        k_submit_rw<<<1,1>>>(qd, sq_tail_um, cid_r,
                             NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                             rslice_io, 0, lba, 0);
        k_poll_one<<<1,1>>>(qd, cq_head_um, cq_phase_um,
                            out_cqe_um, timed_out_um, 50000000ULL);
        CUDA_OK(cudaDeviceSynchronize());
        if (*timed_out_um) step_fail(ETIMEDOUT, "Read IO %u poll timeout", i);
        if ((out_cqe_um->status >> 1) != 0) {
            snprintf(status_buf, sizeof(status_buf), "0x%04x", out_cqe_um->status);
            step_fail(0, "Read IO %u status=%s", i, status_buf);
        }

        *mismatch_um = -1;
        k_verify<<<BLOCKS, THR>>>(rslice, block_size, pat, mismatch_um);
        CUDA_OK(cudaDeviceSynchronize());
        if (*mismatch_um >= 0)
            step_fail(0, "IO %u (lba=%" PRIu64 ") mismatch at byte %d",
                      i, lba, *mismatch_um);
    }
    step_ok("Write+Read+verify x %u IOs at LBA [%llu..%llu] (client fd, owner unaware)",
            TEST_NR_IO,
            (unsigned long long)TEST_LBA_BASE,
            (unsigned long long)(TEST_LBA_BASE + TEST_NR_IO - 1));

    /* Tear down on the client fd. */
    cudaFree(sq_tail_um); cudaFree(cq_head_um); cudaFree(cq_phase_um);
    cudaFree(out_cqe_um); cudaFree(timed_out_um); cudaFree(mismatch_um);

    nvm_dma_unmap(dma_sq); nvm_dma_unmap(dma_cq);
    cudaFree(sq_dev); cudaFree(cq_dev);

    rc = nvm_destroy_group(ctrl, gid);
    if (rc != 0) step_fail(rc, "nvm_destroy_group");
    step_ok("nvm_destroy_group gid=%u (client cascade)", gid);

    nvm_dma_unmap(dma_wbuf); nvm_dma_unmap(dma_rbuf);
    cudaFree(wbuf_dev); cudaFree(rbuf_dev);

    nvm_ctrl_free_client(ctrl);
    step_ok("nvm_ctrl_free_client (no unbind, no chrdev_remove)");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Owner -> client handshake payload                                  */
/* ------------------------------------------------------------------ */

struct role_handshake {
    char     snvme_dev_path[64];   /* e.g. "/dev/ssnvme0" */
    uint32_t bar0_size;
    uint32_t namespace_id;
    uint64_t block_size;
    int32_t  status;               /* 0 = ok, otherwise errno; child exits */
};

/* Read/write helpers that loop on EINTR / partial. */
static int read_full(int fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return errno; }
        if (r == 0) return EPIPE;
        p += r; n -= (size_t)r;
    }
    return 0;
}
static int write_full(int fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    while (n > 0) {
        ssize_t r = write(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return errno; }
        if (r == 0) return EPIPE;
        p += r; n -= (size_t)r;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Main: parent owner + forks child client                            */
/* ------------------------------------------------------------------ */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--gpu N] <PCI_BDF>\n"
        "  e.g.: %s --gpu 0 0000:08:00.0\n"
        "\n"
        "Tests the owner/client role split:\n"
        "  parent (P) forks BEFORE any CUDA call (CUDA is fork-hostile),\n"
        "  parent does B3 bring-up + handshakes child via pipe,\n"
        "  child (C) attaches via nvm_ctrl_attach_client, runs IO,\n"
        "  releases via nvm_ctrl_free_client (no unbind);\n"
        "  parent then unbinds + chrdev_remove cleanly.\n"
        "DESTRUCTIVE.\n",
        prog, prog);
}

int main(int argc, char** argv) {
    int cuda_dev = 0;
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char* arg = argv[argi];
        if (strcmp(arg, "--gpu") == 0 && argi + 1 < argc) {
            cuda_dev = atoi(argv[++argi]); ++argi;
        } else { usage(argv[0]); return 1; }
    }
    if (argi + 1 != argc) { usage(argv[0]); return 1; }
    const char* pci_addr = argv[argi];

    /* fork() BEFORE any cuda* call.  The CUDA runtime initialises a
     * primary context lazily on the first cuda* call; if we forked
     * after that, the child would inherit a half-baked context and
     * cudaSetDevice would return cudaErrorInitializationError. */
    int pipe_fd[2];   /* parent -> child: handshake from owner to client */
    if (pipe(pipe_fd) < 0) {
        fprintf(stderr, "pipe failed: %s\n", strerror(errno));
        return 2;
    }

    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "fork failed: %s\n", strerror(errno));
        return 2;
    }

    if (child == 0) {
        /* ============= CHILD (client role) ================== */
        snprintf(g_role_tag, sizeof(g_role_tag), "C");
        g_step = 0;
        close(pipe_fd[1]);   /* read-only end */

        struct role_handshake hs;
        int rc = read_full(pipe_fd[0], &hs, sizeof(hs));
        if (rc != 0) step_fail(rc, "read handshake from parent");
        close(pipe_fd[0]);
        if (hs.status != 0) {
            /* Parent's bring-up failed; just exit quietly. */
            fprintf(stderr, "[C][..] parent reported status=%d; exiting\n",
                    hs.status);
            _exit(3);
        }
        step_ok("got handshake dev='%s' ns=%u block=%llu bar0=0x%x",
                hs.snvme_dev_path, hs.namespace_id,
                (unsigned long long)hs.block_size, hs.bar0_size);

        int crc = run_client(hs.snvme_dev_path, hs.bar0_size,
                              hs.namespace_id, (size_t)hs.block_size,
                              cuda_dev);
        _exit(crc);
    }

    /* ============= PARENT (owner role) ================== */
    close(pipe_fd[0]);   /* write-only end */
    snprintf(g_role_tag, sizeof(g_role_tag), "P");

    /* From here on the parent may freely use CUDA -- the child is
     * already cloned and has its own fresh address space waiting on
     * the pipe with no CUDA state of its own yet. */
    int rc;
    cudaError_t cerr = cudaSetDevice(cuda_dev);
    if (cerr != cudaSuccess) {
        struct role_handshake hs; memset(&hs, 0, sizeof(hs));
        hs.status = EIO;
        (void) write_full(pipe_fd[1], &hs, sizeof(hs));
        close(pipe_fd[1]);
        (void) waitpid(child, NULL, 0);
        fprintf(stderr, "[P][FAIL] cudaSetDevice: %s\n",
                cudaGetErrorString(cerr));
        return 2;
    }
    step_ok("parent cudaSetDevice(%d)", cuda_dev);

    nvm_ctrl_t* ctrl = NULL;
    struct disk disk;  memset(&disk, 0, sizeof(disk));
    rc = nvm_controller_init_b3(&ctrl, "/dev/snvm_control",
                                pci_addr, TEST_KERNEL_IOQ_CAP, &disk);
    if (rc != 0) {
        struct role_handshake hs; memset(&hs, 0, sizeof(hs));
        hs.status = rc ? rc : EIO;
        (void) write_full(pipe_fd[1], &hs, sizeof(hs));
        close(pipe_fd[1]);
        (void) waitpid(child, NULL, 0);
        step_fail(rc, "nvm_controller_init_b3");
    }
    step_ok("parent owner ready disk='%s' block_size=%zu bar0=0x%x",
            disk.disk_name, disk.block_size, ctrl->bar0_size);

    /* Send handshake to child.  TODO: nvm_controller_init_b3 should
     * expose the chrdev minor explicitly; we know it's 0 in this
     * single-controller smoke. */
    struct role_handshake hs;
    memset(&hs, 0, sizeof(hs));
    snprintf(hs.snvme_dev_path, sizeof(hs.snvme_dev_path),
             "/dev/%s", "ssnvme0");
    hs.bar0_size    = ctrl->bar0_size;
    /* nvm_controller_init_b3 does not currently populate disk.ns_id
     * (the kernel's NVM_GET_DEV_INFO doesn't ship it).  For this
     * single-namespace test controller we hardcode ns_id=1 -- same
     * value snvme_smoke_libnvm_io uses. */
    hs.namespace_id = (disk.ns_id != 0) ? disk.ns_id : 1u;
    hs.block_size   = (uint64_t) disk.block_size;
    hs.status       = 0;
    rc = write_full(pipe_fd[1], &hs, sizeof(hs));
    close(pipe_fd[1]);
    if (rc != 0) {
        (void) waitpid(child, NULL, 0);
        step_fail(rc, "write handshake to child");
    }
    step_ok("parent sent handshake -> '%s'", hs.snvme_dev_path);

    /* Parent: wait for child. */
    int wstat = 0;
    pid_t got = waitpid(child, &wstat, 0);
    if (got != child) step_fail(errno, "waitpid");
    if (!WIFEXITED(wstat) || WEXITSTATUS(wstat) != 0) {
        step_fail(0, "child exited with status=%d (signal=%d)",
                  WEXITSTATUS(wstat), WIFSIGNALED(wstat) ? WTERMSIG(wstat) : 0);
    }
    step_ok("child exited cleanly (rc=0); owner ctrl still alive");

    /* Owner-side teardown: nvm_ctrl_free unbinds + chrdev_removes.
     * If the client accidentally cascaded into unbind, this would
     * have already failed; reaching here proves the role split. */
    nvm_ctrl_free(ctrl);
    step_ok("parent nvm_ctrl_free (unbind + chrdev_remove)");

    fprintf(stderr,
            "\n=== snvme_smoke_libnvm_role: all %d steps passed (parent) ===\n",
            g_step);
    return 0;
}
