/*
 * snvme_smoke_libnvm_b3.c -- L1 Commit 2 smoke.
 *
 * Validates the high-level B3 bring-up entry point introduced in
 * libnvm Commit 2: a single nvm_controller_init_b3() call must do
 * the full chrdev-create -> SET_KERNEL_IOQ_CAP -> BIND -> probe wait
 * -> GET_DEV_INFO -> ctrl_init -> cudaHostRegister sequence in the
 * correct order, and the new wrapper API (nvm_create_group /
 * nvm_destroy_group / nvm_dma_map_*) must reach the kernel with the
 * right ABI.
 *
 * Difference vs snvme_smoke_libnvm.c:
 *
 *   smoke_libnvm.c   -- bring-up via raw ioctls; libnvm only used for
 *                       nvm_dma_map_* (Commit 1 surface).
 *   smoke_libnvm_b3  -- bring-up via nvm_controller_init_b3();
 *                       group/queue/admin via the new wrappers.
 *                       libnvm is the system-under-test all the way
 *                       up to (but not including) actual queue I/O.
 *
 * Test flow:
 *
 *   [ 1] nvm_controller_init_b3(cap=36)         -- one call, end-to-end
 *   [ 2] sanity-check ctrl + disk               -- B3 fields populated
 *   [ 3] nvm_create_group()                     -- per-fd queue group
 *   [ 4] nvm_dma_map_ring_host(SQ)              -- kind=RING_SQ, gid attached
 *   [ 5] nvm_dma_map_ring_host(CQ)              -- kind=RING_CQ
 *   [ 6] nvm_dma_map_data_host()                -- kind=DATA, fd-scoped
 *   [ 7] nvm_dma_unmap rings (no-op kernel-side; group cascade owns)
 *   [ 8] nvm_destroy_group()                    -- ring maps cascade out
 *   [ 9] nvm_dma_unmap data buffer
 *   [10] release ctrl ref WITHOUT nvm_ctrl_free's unbind cascade
 *        (Commit 4 will split owner vs client; today nvm_ctrl_free
 *        still wants to unbind, so we drop the ref via _nvm_ctrl_put
 *        and then issue UNBIND/CHRDEV_REMOVE ourselves with raw
 *        ioctls -- those are owner-only and don't go through
 *        libnvm anyway).
 *   [11] SNVM_DEVICE_UNBIND                    -- raw ioctl
 *   [12] SNVM_CHRDEV_REMOVE                    -- raw ioctl
 *
 * DESTRUCTIVE: binds the target controller; no LBA writes.
 */

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "ioctl.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>

/* Internal libnvm refcount drop (no unbind cascade).  See header
 * comment in snvme_smoke_libnvm.c for why this is exposed manually
 * rather than via lib_ctrl.h -- L1 is mid-refactor and nvm_ctrl_free
 * still hard-couples release to driver unbind. */
struct controller;
extern void _nvm_ctrl_put(struct controller* ctrl);

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

/* ------------------------------------------------------------------ */
/* DMA invariants                                                     */
/* ------------------------------------------------------------------ */

static void expect_dma_ok(const nvm_dma_t* d, const char* tag) {
    if (d == NULL)             step_fail(EINVAL, "%s: dma handle null", tag);
    if (d->n_ioaddrs == 0)     step_fail(EINVAL, "%s: n_ioaddrs == 0", tag);
    if (d->ioaddrs[0] == 0)    step_fail(EINVAL, "%s: ioaddr[0] == 0", tag);
    if (d->page_size == 0)     step_fail(EINVAL, "%s: page_size == 0", tag);
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr,
                "usage: %s DDDD:BB:DD.F\n"
                "  e.g.: %s 0000:08:00.0\n"
                "DESTRUCTIVE: binds the target controller via libnvm B3.\n",
                argv[0], argv[0]);
        return 1;
    }

    /* ===== [1] one-shot B3 bring-up via libnvm ===== */

    nvm_ctrl_t* ctrl = NULL;
    struct disk disk;
    memset(&disk, 0, sizeof(disk));

    int rc = nvm_controller_init_b3(&ctrl,
                                    "/dev/snvm_control",
                                    argv[1],
                                    /* kernel_ioq_cap */ 36,
                                    &disk);
    if (rc != 0) step_fail(rc, "nvm_controller_init_b3");
    if (ctrl == NULL) step_fail(EINVAL, "nvm_controller_init_b3: ctrl null");
    step_ok("nvm_controller_init_b3 -> ctrl=%p", (void*)ctrl);

    /* ===== [2] sanity-check ctrl + disk ===== */

    if (ctrl->page_size == 0)
        step_fail(EINVAL, "ctrl->page_size == 0 (BAR0 mmap?)");
    if (ctrl->q_depth == 0)
        step_fail(EINVAL, "ctrl->q_depth == 0 (GET_DEV_INFO not applied)");
    if (ctrl->max_user_qid <= ctrl->start_cq_idx && ctrl->start_cq_idx != 0)
        step_fail(EINVAL,
                  "ctrl: max_user_qid (%u) <= start_cq_idx (%u): no user QID room",
                  ctrl->max_user_qid, ctrl->start_cq_idx);
    if (ctrl->max_queues_per_group == 0)
        step_fail(EINVAL, "ctrl->max_queues_per_group == 0");
    if (disk.disk_name[0] == '\0')
        step_fail(EINVAL, "disk.disk_name empty (probe race?)");
    step_ok("ctrl page_size=%zu q_depth=%u start_cq_idx=%u max_user_qid=%u "
            "max_q_per_grp=%u sgls=0x%x",
            ctrl->page_size, (unsigned)ctrl->q_depth,
            ctrl->start_cq_idx, ctrl->max_user_qid,
            ctrl->max_queues_per_group, ctrl->sgl_supported);
    step_ok("disk name='%s' block_size=%zu max_data_size=%zu",
            disk.disk_name, disk.block_size, disk.max_data_size);

    /* ===== [3] queue group via libnvm wrapper ===== */

    uint32_t gid = 0, max_q = 0;
    rc = nvm_create_group(ctrl, &gid, &max_q);
    if (rc != 0) step_fail(rc, "nvm_create_group");
    if (gid == 0) step_fail(EINVAL, "nvm_create_group: gid == 0");
    if (max_q == 0) step_fail(EINVAL, "nvm_create_group: max_queues == 0");
    step_ok("nvm_create_group -> gid=%u max_queues=%u", gid, max_q);

    /* ===== [4..5] ring host buffers via the explicit-intent API ===== */

    /* SQ ring: q_depth * 64 B = 4096 B for q_depth=64; pad to a host page. */
    const size_t ring_bytes = ctrl->page_size;
    void* ring_sq = NULL;
    void* ring_cq = NULL;
    if (posix_memalign(&ring_sq, ctrl->page_size, ring_bytes) != 0)
        step_fail(errno, "posix_memalign ring_sq");
    if (posix_memalign(&ring_cq, ctrl->page_size, ring_bytes) != 0)
        step_fail(errno, "posix_memalign ring_cq");
    memset(ring_sq, 0, ring_bytes);
    memset(ring_cq, 0, ring_bytes);

    nvm_dma_t* dma_sq = NULL;
    rc = nvm_dma_map_ring_host(&dma_sq, ctrl, gid, ring_sq, ring_bytes,
                               /*is_cq=*/0);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_host SQ");
    expect_dma_ok(dma_sq, "ring_sq");
    step_ok("nvm_dma_map_ring_host SQ gid=%u ioaddr[0]=0x%" PRIx64,
            gid, (uint64_t)dma_sq->ioaddrs[0]);

    nvm_dma_t* dma_cq = NULL;
    rc = nvm_dma_map_ring_host(&dma_cq, ctrl, gid, ring_cq, ring_bytes,
                               /*is_cq=*/1);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_host CQ");
    expect_dma_ok(dma_cq, "ring_cq");
    step_ok("nvm_dma_map_ring_host CQ gid=%u ioaddr[0]=0x%" PRIx64,
            gid, (uint64_t)dma_cq->ioaddrs[0]);

    /* ===== [6] DATA buffer (fd-scoped) ===== */

    const size_t data_bytes = 64 * 1024;
    void* data_buf = NULL;
    if (posix_memalign(&data_buf, ctrl->page_size, data_bytes) != 0)
        step_fail(errno, "posix_memalign data_buf");
    memset(data_buf, 0, data_bytes);

    nvm_dma_t* dma_data = NULL;
    rc = nvm_dma_map_data_host(&dma_data, ctrl, data_buf, data_bytes);
    if (rc != 0) step_fail(rc, "nvm_dma_map_data_host");
    expect_dma_ok(dma_data, "data");
    step_ok("nvm_dma_map_data_host n_ioaddrs=%zu ioaddr[0]=0x%" PRIx64
            " (kind=DATA, fd-scoped)",
            dma_data->n_ioaddrs, (uint64_t)dma_data->ioaddrs[0]);

    /* ===== [7] unmap rings.  This invokes libnvm's user-side bookkeeping
     *      but is a no-op on the kernel side under B6: ring_* maps
     *      live on g->maps and only get released via the group's
     *      destroy cascade.  The point is to confirm libnvm itself
     *      is OK with the unmap (no double-free / no fd error). ===== */

    nvm_dma_unmap(dma_sq);
    nvm_dma_unmap(dma_cq);
    step_ok("nvm_dma_unmap rings (kernel cascade still owns the maps)");

    /* ===== [8] destroy the queue group via libnvm wrapper ===== */

    rc = nvm_destroy_group(ctrl, gid);
    if (rc != 0) step_fail(rc, "nvm_destroy_group");
    step_ok("nvm_destroy_group gid=%u (kernel: drained 2 ring map(s))", gid);

    /* ===== [9] DATA buffer survives destroy_group; unmap explicitly ===== */

    nvm_dma_unmap(dma_data);
    step_ok("nvm_dma_unmap data (DATA fd-scoped, fd-cascade would catch leaks)");

    free(ring_sq);
    free(ring_cq);
    free(data_buf);

    /* ===== [10] release libnvm ctrl ref WITHOUT unbind cascade ===== */
    {
        struct controller* c =
            (struct controller*) ctrl_to_controller(ctrl);
        if (c != NULL) _nvm_ctrl_put(c);
    }
    step_ok("libnvm ctrl ref dropped (skipping nvm_ctrl_free's unbind cascade)");

    /* ===== [11..12] owner-only teardown via raw ioctls.
     *
     * nvm_controller_init_b3 closed our copies of the fds (they got
     * dup'd into the libnvm-internal struct device), but UNBIND /
     * CHRDEV_REMOVE want fd_control which we don't have here anymore.
     *
     * Workaround for this smoke: re-open /dev/snvm_control and reuse
     * the BDF.  This mirrors the role split that Commit 4 will make
     * explicit: a separate "owner" object would have kept its own
     * fd_control.
     */
    int fd_ctrl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctrl < 0) step_fail(errno, "re-open /dev/snvm_control");

    struct pci_device_addr orig_bdf;
    if (sscanf(argv[1], "%x:%x:%x.%x",
               &orig_bdf.domain, &orig_bdf.bus,
               &orig_bdf.slot, &orig_bdf.func) != 4)
        step_fail(EINVAL, "parse_bdf");

    {
        struct pci_device_addr a = orig_bdf;
        if (ioctl(fd_ctrl, SNVM_DEVICE_UNBIND, &a) < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND");
        step_ok("SNVM_DEVICE_UNBIND %s", argv[1]);
    }
    {
        struct pci_device_addr a = orig_bdf;
        if (ioctl(fd_ctrl, SNVM_CHRDEV_REMOVE, &a) < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE");
        step_ok("SNVM_CHRDEV_REMOVE %s", argv[1]);
    }
    close(fd_ctrl);

    fprintf(stderr,
            "\n=== snvme_smoke_libnvm_b3: all %d steps passed ===\n", g_step);
    return 0;
}
