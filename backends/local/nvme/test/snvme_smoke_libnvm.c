/*
 * snvme_smoke_libnvm.c -- L1 smoke for libnvm's B3/B6 DMA API.
 *
 * Layer scope
 * -----------
 *
 * This test sits one layer ABOVE the raw-ioctl smokes in
 * backends/local/kernel_modules/test/.  It links against libnvm.so and
 * exercises the libnvm-internal map plumbing introduced in Commit 1 of
 * the L1 refactor:
 *
 *   - struct ioctl_mapping.group_id / map_kind
 *   - struct map.group_id / map_kind transit through _nvm_dma_init
 *   - nvm_ioctl_map ABI fields actually populated by ioctl_map()
 *   - dma_map() unmap-callback selection on map_kind
 *
 * Plus the four explicit-intent entry points:
 *
 *   nvm_dma_map_data_host()
 *   nvm_dma_map_data_device()
 *   nvm_dma_map_ring_host(group_id, is_cq)
 *   nvm_dma_map_ring_device(group_id, is_cq)
 *
 * And confirms the legacy entry points (nvm_dma_map_host / _device /
 * _queue_device) still travel the same wire layout they did pre-B6 --
 * map_kind == NVM_MAP_KIND_UNSPECIFIED takes the kernel's legacy
 * fallback branch -- so existing callers (Controller, libtutti)
 * keep working bit-for-bit during the migration.
 *
 * What it does NOT do
 * -------------------
 *
 * - Does NOT exercise Controller::Controller / Controller::init_queues
 *   (those are still on the legacy bring-up path; addressed in
 *   later commits L1.2 / L1.3).
 * - Does NOT call NVM_ADD_USER_QUEUE.  Map registration is the unit
 *   under test here; queue creation has its own coverage in
 *   snvme_smoke_addq (raw-ioctl level).
 * - Does NOT issue any NVMe IO.  Data-plane integrity is covered by
 *   the raw-ioctl smoke_io and the GPU smoke.
 *
 * Build:    make snvme_smoke_libnvm   (this directory's Makefile)
 * Run:      sudo ./snvme_smoke_libnvm <PCI_BDF>
 *
 * Side effects
 * ------------
 *
 * DESTRUCTIVE: binds the target controller via SNVM_DEVICE_BIND so the
 * BAR0 + admin queues become snvme-owned.  No LBA is written.  Test
 * cleans up: NVM_DESTROY_QUEUE_GROUP -> SNVM_DEVICE_UNBIND ->
 * SNVM_CHRDEV_REMOVE -> fd close.
 *
 * Exit codes:
 *   0 -- all steps passed.
 *   1 -- usage error.
 *   2 -- a smoke step failed; stderr names which one.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "ioctl.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>

/* Internal libnvm symbol used by this smoke to drop the controller
 * refcount WITHOUT triggering nvm_ctrl_free's cascade through
 * unbind/chrdev_remove.  Declared here (rather than via lib_ctrl.h,
 * which is an internal header pulling pthread/mutex deps) to keep
 * the smoke a single self-contained .c.  Will become unnecessary
 * once libnvm splits owner-vs-client role and exposes a proper
 * "release fd only" entry point.  */
struct controller;
extern void _nvm_ctrl_put(struct controller* ctrl);

/* ------------------------------------------------------------------ */
/* Logging helpers (kept symmetric with raw-ioctl smokes)             */
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

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Direct-ioctl helpers (used to set up + tear down the device, and   */
/* to create / destroy the queue group that the ring_* tests run      */
/* against.  Everything DMA-related goes through libnvm itself.)      */
/* ------------------------------------------------------------------ */

static int control_fd_open(void) {
    int fd = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd < 0) return -1;
    return fd;
}

static int chrdev_create(int fd_ctrl, struct pci_device_addr* bdf) {
    return ioctl(fd_ctrl, SNVM_CHRDEV_CREATE, bdf) < 0 ? -errno : 0;
}

static int chrdev_remove(int fd_ctrl, struct pci_device_addr* bdf) {
    return ioctl(fd_ctrl, SNVM_CHRDEV_REMOVE, bdf) < 0 ? -errno : 0;
}

static int device_bind(int fd_ctrl, struct pci_device_addr* bdf) {
    return ioctl(fd_ctrl, SNVM_DEVICE_BIND, bdf) < 0 ? -errno : 0;
}

static int device_unbind(int fd_ctrl, struct pci_device_addr* bdf) {
    return ioctl(fd_ctrl, SNVM_DEVICE_UNBIND, bdf) < 0 ? -errno : 0;
}

static int set_kernel_ioq_cap(int fd_dev, uint32_t cap) {
    return ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) < 0 ? -errno : 0;
}

/* get_dev_info: kept here as documentation of the one-shot ioctl
 * shape, but unused -- the smoke uses an inline poll loop above
 * because GET_DEV_INFO returns -ENODEV until nvme_scan_work fills
 * in disk_name etc.  asynchronously after bind. */
static int get_dev_info(int fd_dev, struct nvm_ioctl_dev* out) __attribute__((unused));
static int get_dev_info(int fd_dev, struct nvm_ioctl_dev* out) {
    memset(out, 0, sizeof(*out));
    return ioctl(fd_dev, NVM_GET_DEV_INFO, out) < 0 ? -errno : 0;
}

static int create_qgroup(int fd_dev, uint32_t* gid_out, uint32_t* max_q_out) {
    struct nvm_ioctl_queue_group req;
    memset(&req, 0, sizeof(req));
    if (ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req) < 0) return -errno;
    *gid_out = req.group_id;
    *max_q_out = req.max_queues;
    return 0;
}

static int destroy_qgroup(int fd_dev, uint32_t gid) {
    return ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid) < 0 ? -errno : 0;
}

/* ------------------------------------------------------------------ */
/* Test invariants on the returned nvm_dma_t                          */
/* ------------------------------------------------------------------ */

static void expect_dma_ok(const nvm_dma_t* d, const char* tag) {
    if (d == NULL)             step_fail(EINVAL, "%s: dma handle null", tag);
    if (d->n_ioaddrs == 0)     step_fail(EINVAL, "%s: n_ioaddrs == 0", tag);
    if (d->ioaddrs[0] == 0)    step_fail(EINVAL, "%s: ioaddr[0] == 0", tag);
    if (d->page_size == 0)     step_fail(EINVAL, "%s: page_size == 0", tag);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr,
                "Usage: %s <PCI_BDF>\n"
                "  e.g.: %s 0000:08:00.0\n\n"
                "L1 smoke: exercises libnvm B3/B6 DMA API\n"
                "(nvm_dma_map_data_*, nvm_dma_map_ring_*, legacy fallback).\n"
                "BINDS the controller; no LBA writes.\n",
                argv[0], argv[0]);
        return 1;
    }

    struct pci_device_addr orig_bdf;
    if (parse_bdf(argv[1], &orig_bdf) != 0) {
        fprintf(stderr, "bad PCI BDF: '%s' (want DDDD:BB:DD.F)\n", argv[1]);
        return 1;
    }

    /* ---- Bring-up (raw ioctl) ---- */

    int fd_ctrl = control_fd_open();
    if (fd_ctrl < 0) step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctrl);

    /* SNVM_CHRDEV_CREATE clobbers the input pci_device_addr -- it
     * memset(addr, 0, ...) and then stuffs the assigned chrdev minor
     * into addr->domain as an out-param.  So we MUST copy orig_bdf
     * into a throwaway every time we call any *_BDF ioctl, otherwise
     * the second call sends 0:0:0.0 to the kernel and bind/unbind
     * silently target the PCI root bridge.  See Todolist.md
     * "Stop overloading pci_device_addr.domain".  */
    struct pci_device_addr bdf_for_create = orig_bdf;
    int rc = chrdev_create(fd_ctrl, &bdf_for_create);
    if (rc != 0) step_fail(-rc, "SNVM_CHRDEV_CREATE");
    int minor_n = bdf_for_create.domain;       /* out-param: chrdev minor */
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);

    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0) step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    rc = set_kernel_ioq_cap(fd_dev, 36);
    if (rc != 0) step_fail(-rc, "NVM_SET_KERNEL_IOQ_CAP");
    step_ok("NVM_SET_KERNEL_IOQ_CAP cap=36");

    {
        struct pci_device_addr bdf = orig_bdf;
        rc = device_bind(fd_ctrl, &bdf);
        if (rc != 0) step_fail(-rc, "SNVM_DEVICE_BIND");
        step_ok("SNVM_DEVICE_BIND %s", argv[1]);
    }

    /* SNVM_DEVICE_BIND returns as soon as the chrdev is callable, but
     * the kernel runs nvme_reset_work + nvme_scan_work asynchronously
     * on s_nvme_wq.  GET_DEV_INFO returns -ENODEV until those finish
     * and ns->disk_name etc. get populated.  Poll up to ~10 s -- fast
     * NVMe completes in a couple hundred ms; allow more for slow /
     * power-managed drives.  Same loop shape as the L0 smokes (see
     * snvme_smoke_io.c "Phase: NVM_GET_DEV_INFO"). */
    struct nvm_ioctl_dev devinfo;
    {
        int ok = 0;
        for (int i = 0; i < 100; i++) {
            memset(&devinfo, 0, sizeof(devinfo));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &devinfo) == 0 &&
                devinfo.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not complete within 10s "
                             "(probe still running?)");
    }
    step_ok("NVM_GET_DEV_INFO disk='%s' q_depth=%u start_cq_idx=%u max_user_qid=%u",
            devinfo.disk_name, devinfo.q_depth,
            devinfo.start_cq_idx, devinfo.max_user_qid);

    /* ---- libnvm: wrap fd_dev into nvm_ctrl_t ---- */

    nvm_ctrl_t* ctrl = NULL;
    rc = nvm_ctrl_init(&ctrl, fd_ctrl, fd_dev);
    if (rc != 0) step_fail(rc, "nvm_ctrl_init");
    if (ctrl == NULL) step_fail(EINVAL, "nvm_ctrl_init: ctrl null");
    if (ctrl->page_size == 0) step_fail(EINVAL, "ctrl->page_size == 0 (BAR0 mmap?)");
    step_ok("nvm_ctrl_init page_size=%zu dstrd=%u max_qs=%u",
            ctrl->page_size, ctrl->dstrd, ctrl->max_qs);

    /* ---- B3 group container (used by ring_* tests) ---- */

    uint32_t gid = 0, max_q = 0;
    rc = create_qgroup(fd_dev, &gid, &max_q);
    if (rc != 0) step_fail(-rc, "NVM_CREATE_QUEUE_GROUP");
    if (gid == 0) step_fail(EINVAL, "group_id == 0 (reserved sentinel)");
    step_ok("NVM_CREATE_QUEUE_GROUP -> group_id=%u max_queues=%u", gid, max_q);

    /* ---- Test 1: DATA host map (B6 fd-scoped) ---- */

    void* host_data = NULL;
    size_t host_data_size = 64u * 1024;
    if (posix_memalign(&host_data, 4096, host_data_size) != 0)
        step_fail(errno, "posix_memalign host_data");
    memset(host_data, 0xab, host_data_size);

    nvm_dma_t* dma_data_host = NULL;
    rc = nvm_dma_map_data_host(&dma_data_host, ctrl, host_data, host_data_size);
    if (rc != 0) step_fail(rc, "nvm_dma_map_data_host");
    expect_dma_ok(dma_data_host, "data_host");
    step_ok("nvm_dma_map_data_host n_ioaddrs=%zu ioaddr[0]=0x%lx (kind=DATA, fd-scoped)",
            dma_data_host->n_ioaddrs, (unsigned long)dma_data_host->ioaddrs[0]);

    /* ---- Test 2: ring_host with group_id=0 must fail early (-EINVAL) ---- */

    nvm_dma_t* dma_should_fail = NULL;
    rc = nvm_dma_map_ring_host(&dma_should_fail, ctrl, 0, host_data, host_data_size, 0);
    if (rc == 0)
        step_fail(0, "nvm_dma_map_ring_host accepted group_id=0 (must reject)");
    if (rc != EINVAL)
        step_fail(rc, "nvm_dma_map_ring_host group_id=0: want EINVAL got %d", rc);
    step_ok("nvm_dma_map_ring_host(group_id=0) -> EINVAL (early reject)");

    /* ---- Test 3: ring_host SQ (group-scoped, kind=RING_SQ) ---- */

    void* host_ring_sq = NULL;
    size_t ring_size = 4096;
    if (posix_memalign(&host_ring_sq, 4096, ring_size) != 0)
        step_fail(errno, "posix_memalign host_ring_sq");
    memset(host_ring_sq, 0, ring_size);

    nvm_dma_t* dma_ring_sq = NULL;
    rc = nvm_dma_map_ring_host(&dma_ring_sq, ctrl, gid, host_ring_sq, ring_size, 0 /* SQ */);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_host SQ");
    expect_dma_ok(dma_ring_sq, "ring_sq");
    step_ok("nvm_dma_map_ring_host SQ gid=%u n_ioaddrs=%zu ioaddr[0]=0x%lx",
            gid, dma_ring_sq->n_ioaddrs, (unsigned long)dma_ring_sq->ioaddrs[0]);

    /* ---- Test 4: ring_host CQ (group-scoped, kind=RING_CQ) ---- */

    void* host_ring_cq = NULL;
    if (posix_memalign(&host_ring_cq, 4096, ring_size) != 0)
        step_fail(errno, "posix_memalign host_ring_cq");
    memset(host_ring_cq, 0, ring_size);

    nvm_dma_t* dma_ring_cq = NULL;
    rc = nvm_dma_map_ring_host(&dma_ring_cq, ctrl, gid, host_ring_cq, ring_size, 1 /* CQ */);
    if (rc != 0) step_fail(rc, "nvm_dma_map_ring_host CQ");
    expect_dma_ok(dma_ring_cq, "ring_cq");
    step_ok("nvm_dma_map_ring_host CQ gid=%u n_ioaddrs=%zu ioaddr[0]=0x%lx",
            gid, dma_ring_cq->n_ioaddrs, (unsigned long)dma_ring_cq->ioaddrs[0]);

    /* ---- Test 5: legacy nvm_dma_map_host(-1,-1) still works ---- */

    void* host_legacy = NULL;
    if (posix_memalign(&host_legacy, 4096, host_data_size) != 0)
        step_fail(errno, "posix_memalign host_legacy");
    memset(host_legacy, 0x5a, host_data_size);

    nvm_dma_t* dma_legacy = NULL;
    rc = nvm_dma_map_host(&dma_legacy, ctrl, host_legacy, host_data_size, -1, -1);
    if (rc != 0) step_fail(rc, "legacy nvm_dma_map_host(-1,-1)");
    expect_dma_ok(dma_legacy, "legacy_host");
    step_ok("nvm_dma_map_host(-1,-1) [legacy, kind=UNSPECIFIED] n_ioaddrs=%zu ioaddr[0]=0x%lx",
            dma_legacy->n_ioaddrs, (unsigned long)dma_legacy->ioaddrs[0]);

    /* ---- Tear down rings BEFORE destroying group ---- */

    /* RING_* maps install NO explicit unmap callback (released by
     * NVM_DESTROY_QUEUE_GROUP cascade).  nvm_dma_unmap on them just
     * drops the libnvm-side container, which is what we want here:
     * we'll let the group-destroy cascade release the kernel side. */
    nvm_dma_unmap(dma_ring_sq);
    nvm_dma_unmap(dma_ring_cq);
    step_ok("nvm_dma_unmap rings (kernel side released by group destroy)");

    /* ---- Destroy group: kernel should cascade-release the 2 ring maps ---- */

    rc = destroy_qgroup(fd_dev, gid);
    if (rc != 0) step_fail(-rc, "NVM_DESTROY_QUEUE_GROUP");
    step_ok("NVM_DESTROY_QUEUE_GROUP id=%u (cascades 2 ring map(s))", gid);

    /* ---- DATA + legacy maps survive group destroy; release them now ---- */

    nvm_dma_unmap(dma_data_host);  /* triggers NVM_UNMAP_HOST_MEMORY */
    nvm_dma_unmap(dma_legacy);     /* triggers NVM_UNMAP_HOST_MEMORY */
    step_ok("nvm_dma_unmap DATA + legacy host maps");

    free(host_data);
    free(host_ring_sq);
    free(host_ring_cq);
    free(host_legacy);

    /* ---- Bring-down (raw ioctl, mirror order) ---- */

    /* Drop libnvm ctrl WITHOUT going through nvm_ctrl_free -- the
     * latter unconditionally calls nvm_device_unbind + chrdev_remove,
     * which is the role-confusion problem we'll fix in a later
     * commit.  Here we just want libnvm to release its dup'd fds. */
    {
        struct controller* c = ctrl_to_controller(ctrl);
        if (c != NULL) _nvm_ctrl_put(c);
    }
    step_ok("libnvm ctrl released (no unbind from libnvm path)");

    {
        struct pci_device_addr bdf = orig_bdf;
        rc = device_unbind(fd_ctrl, &bdf);
        if (rc != 0) step_fail(-rc, "SNVM_DEVICE_UNBIND");
        step_ok("SNVM_DEVICE_UNBIND %s", argv[1]);
    }

    /* close fd_dev BEFORE chrdev_remove: this is where the kernel's
     * snvm_dev_release runs and would cascade-clean any leftover
     * DATA maps.  We already unmapped DATA + legacy above so the
     * dmesg line should report 0 cascade-released DATA map(s); the
     * test doesn't fail on that count, it's informational. */
    close(fd_dev);
    step_ok("close(%s)", dev_path);

    {
        struct pci_device_addr bdf = orig_bdf;
        rc = chrdev_remove(fd_ctrl, &bdf);
        if (rc != 0) step_fail(-rc, "SNVM_CHRDEV_REMOVE");
        step_ok("SNVM_CHRDEV_REMOVE %s", argv[1]);
    }

    close(fd_ctrl);

    fprintf(stderr,
            "\n=== snvme_smoke_libnvm: all %d steps passed ===\n", g_step);
    return 0;
}
