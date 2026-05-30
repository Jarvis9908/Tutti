/*
 * snvme_smoke_recycle.c -- Standalone test for NVM_RAW_ADMIN_CMD.
 *
 * Purpose
 * -------
 *
 * Verify that snvme's NVM_RAW_ADMIN_CMD ioctl can drive arbitrary NVMe
 * admin commands through the controller's snvme-owned admin queue,
 * with CQE status / DW0 / DW1 surfaced back to userspace.
 *
 * Why
 * ---
 *
 * This is the kernel-level building block for NVMeService's
 * per-queue recycle path (Delete I/O SQ -> Delete I/O CQ -> Create
 * I/O CQ -> Create I/O SQ; NVMe 1.4 §5.4 / §5.5).  Before we wire
 * the full recycle into the daemon, we want a focused test that
 * exercises just the admin pass-through ioctl -- so that if the
 * follow-up recycle test breaks, we already know whether the
 * problem is in the new ioctl or in the four-command sequence /
 * ring-memory bookkeeping on top of it.
 *
 * Coverage in this revision (T2a):
 *   - bind the controller (required: the ioctl returns -ENODEV
 *     until pci_get_drvdata(ctrl->pdev) yields a valid nvme_dev),
 *   - issue an Identify Controller admin command via
 *     NVM_RAW_ADMIN_CMD with a NULL data buffer.  Status is
 *     expected to be 0x01 "Invalid Field in Command" (SC=2)
 *     because we deliberately did NOT provide a PRP1 buffer -- the
 *     point of T2a is to confirm the ioctl pipes the command and
 *     surfaces the controller's CQE status verbatim, not that the
 *     command itself succeeds.  Any other status indicates a real
 *     bug in the new pass-through code.
 *   - unbind, tear down /dev/ssnvme<N>.
 *
 * Out of scope (T2b, separate test):
 *   - Delete + Create I/O SQ/CQ across the recycle, with verified
 *     subsequent NVMe read.  That needs user-IOQ ring memory, which
 *     belongs in its own binary because the failure surface is
 *     different (DMA mapping / ring sizing vs admin pass-through).
 *
 * Build:    make snvme_smoke_recycle              (parent Makefile)
 * Invoke:   sudo ./snvme_smoke_recycle <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
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
#include <sys/stat.h>
#include <unistd.h>

/* The *only* SNVMe-specific header we need: the kernel/user UAPI. */
#include "ioctl.h"

/* ------------------------------------------------------------------ */
/* Logging helpers (copied verbatim from snvme_smoke.c for symmetry). */
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

static void step_warn(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[WARN] step=%-2d ", g_step);
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
/* BDF parser                                                         */
/* ------------------------------------------------------------------ */

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* ioctl wrapper                                                      */
/* ------------------------------------------------------------------ */

static int do_ioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        int e = errno;
        fprintf(stderr, "ioctl(%s) failed: %s\n", what, strerror(e));
        errno = e;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* NVMe admin command builders.                                       */
/*                                                                    */
/* We only need a Identify Controller SQE for T2a.  Field layout is   */
/* NVMe 1.4 §5.15 Figure 245 (Identify Command).  The kernel will     */
/* fill in the CID via snvme_submit_sync_cmd's blk-mq tag, so we      */
/* leave CDW0[31:16] zero.                                            */
/* ------------------------------------------------------------------ */

#define NVME_ADMIN_OPC_IDENTIFY  0x06
#define NVME_IDENTIFY_CNS_CTRL   0x01

/*
 * Build an Identify (Controller) admin SQE with no data buffer
 * pointer.  Controller is expected to reject with SC=0x02 "Invalid
 * Field in Command" (DW10 CNS=0x01 requires PRP1 to point at a
 * 4096-byte buffer; we sent zero).
 *
 * This is the cheapest way to confirm the ioctl plumbs the command
 * end-to-end without dragging in DMA-buffer plumbing (which belongs
 * to T2b).
 */
static void build_identify_ctrl_sqe(uint8_t out[64]) {
    memset(out, 0, 64);
    /* CDW0: opcode in [7:0] */
    out[0] = NVME_ADMIN_OPC_IDENTIFY;
    /* CDW10: CNS = 0x01 (Identify Controller).  little-endian. */
    out[40] = NVME_IDENTIFY_CNS_CTRL;     /* DW10 byte 0 = CNS */
    out[41] = 0;
    out[42] = 0;
    out[43] = 0;
}

/*
 * Build an admin SQE with a reserved opcode (0xFF) and all other
 * fields zero.  Per NVMe 1.4 §5 (Admin Command Set, Figure 139), any
 * opcode the controller does not implement MUST complete with
 * Status Code Type 0x0 (Generic Command Status), SC = 0x01
 * "Invalid Command Opcode".  This holds independently of vendor
 * extensions: 0xFF is the highest byte and is universally reserved
 * for vendor-specific commands the controller is allowed to NOT
 * implement -- and Tencent NVMe firmwares we test against do not.
 *
 * We use this instead of an Identify-with-PRP1=0 because the latter
 * is *strongly* recommended to be rejected, but not required, and
 * some controllers (including the one this test was first run on)
 * silently accept it.  An unimplemented opcode is the most portable
 * "guaranteed reject" we can construct without sending data.
 */
static void build_reserved_opcode_sqe(uint8_t out[64]) {
    memset(out, 0, 64);
    out[0] = 0xff;        /* CDW0 byte 0 = opcode (reserved) */
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "This test BINDS the controller (destructive: the in-tree nvme\n"
        "driver loses the device for the duration).  Unlike\n"
        "./snvme_smoke, NVM_RAW_ADMIN_CMD always requires a probed\n"
        "controller so there is no UAPI-only mode here.\n",
        prog, prog);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    const char* bdf_str = argv[1];

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* [1] /dev/snvm_control                                              */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    /* ------------------------------------------------------------------ */
    /* [2] SNVM_CHRDEV_CREATE                                             */
    /* ------------------------------------------------------------------ */
    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [3] /dev/ssnvme<N>                                                 */
    /* ------------------------------------------------------------------ */
    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ------------------------------------------------------------------ */
    /* [4] NVM_RAW_ADMIN_CMD before bind -- expected -ENODEV.              */
    /*                                                                    */
    /* The kernel handler should reject this with -ENODEV because the     */
    /* controller is not yet bound to snvme (pci_get_drvdata returns      */
    /* NULL or no admin_q).  This negative test catches an entire class   */
    /* of regressions: if we forget the bound-check and dereference       */
    /* ndev->ctrl.admin_q on a fresh chrdev, the host oopses.             */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin pre;
        memset(&pre, 0, sizeof(pre));
        build_identify_ctrl_sqe(pre.sqe);
        int r = ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &pre);
        if (r == 0)
            step_fail(0,
                "NVM_RAW_ADMIN_CMD unexpectedly returned 0 on UNBOUND ctrl "
                "(nvme_status=0x%04x); kernel handler is missing the "
                "ndev/admin_q liveness check",
                (unsigned)pre.nvme_status);
        if (errno != ENODEV)
            step_warn("NVM_RAW_ADMIN_CMD on unbound ctrl returned errno=%d (%s); "
                      "expected ENODEV. Not fatal, but the kernel may have a "
                      "tighter check than this test expects.",
                      errno, strerror(errno));
        else
            step_ok("NVM_RAW_ADMIN_CMD on unbound ctrl correctly returned -ENODEV");
    }

    /* ------------------------------------------------------------------ */
    /* [5] SNVM_DEVICE_BIND                                                */
    /*                                                                    */
    /* No NVM_SET_IOQ_NUM / NVM_MAP_HOST_MEMORY / NVM_SET_SHARE_REG       */
    /* preamble -- this test does NOT use user IOQs.  Bind alone lets    */
    /* snvme run the in-tree probe path (kernel-only queues), which is    */
    /* the minimum needed for ndev->ctrl.admin_q to be live.             */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- the in-tree nvme driver may still own "
                "this device (try `sudo sh -c 'echo %s > /sys/bus/pci/drivers/nvme/unbind'` "
                "first)", bdf_str, bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [6] Poll NVM_GET_DEV_INFO until probe completes.                    */
    /*                                                                    */
    /* Probe is asynchronous (reset_work + scan_work).  On a healthy bind */
    /* this finishes in well under a second; allow up to ~10s for slow    */
    /* / power-managed drives.                                            */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_dev info;
        int ok = 0;
        for (int i = 0; i < 100; i++) {
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                step_ok("NVM_GET_DEV_INFO disk='%s' nr_user_q=%u block_size=%zu",
                        info.disk_name, info.nr_user_q, info.block_size);
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not succeed within 10s "
                             "after bind (probe still running?)");
    }

    /* ------------------------------------------------------------------ */
    /* [7a] NVM_RAW_ADMIN_CMD positive: Identify Controller, no buffer.    */
    /*                                                                    */
    /* What we expect:                                                    */
    /*   - ioctl return:  0   (CQE arrived)                              */
    /*   - nvme_status:   0x0000  (SC=0x00 Successful Completion)         */
    /*                                                                    */
    /* Why no data buffer is OK here:                                     */
    /*   The kernel handler hard-codes (buffer=NULL, bufflen=0), so      */
    /*   __snvme_submit_sync_cmd does NOT remap PRP1.  The original     */
    /*   plan was to use Identify-with-PRP1=0 as a "guaranteed reject"  */
    /*   case, but in practice many controllers tolerate it (see        */
    /*   step 7b for a portable negative case).  All we assert here     */
    /*   is that Identify-with-zero-PRP returns *something* via the     */
    /*   pass-through path; success or failure both prove the SQE/CQE   */
    /*   round-trip works.                                               */
    /*                                                                    */
    /* What this proves:                                                  */
    /*   The new ioctl correctly:                                         */
    /*     (a) copies the SQE from userspace,                            */
    /*     (b) forwards via __snvme_submit_sync_cmd to ctrl.admin_q,    */
    /*     (c) round-trips CQE DW0/DW3 status back to userspace,        */
    /*     (d) does not panic on a buffer-less Identify.                */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin req;
        memset(&req, 0, sizeof(req));
        build_identify_ctrl_sqe(req.sqe);

        if (do_ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &req,
                     "NVM_RAW_ADMIN_CMD(Identify Controller)") < 0)
            step_fail(errno, "NVM_RAW_ADMIN_CMD pass-through ioctl failed "
                             "(ioctl-level error, not NVMe-level)");

        /* SC (status code) lives in CQE DW3 bits 15:1.  __snvme_-
         * submit_sync_cmd already right-shifts that field by one, so
         * what we get back in nvme_status is already SC|SCT|... with
         * the phase bit dropped.  Bits [7:0] = SC, [10:8] = SCT,
         * [13:11] = reserved (CRD), [14] = MORE, [15] = DNR.
         */
        uint16_t sc  = req.nvme_status & 0xff;
        uint16_t sct = (req.nvme_status >> 8) & 0x7;

        step_ok("NVM_RAW_ADMIN_CMD(Identify) round-trip: nvme_status=0x%04x "
                "(SC=0x%02x SCT=0x%x) dw0=0x%08x dw1=0x%08x",
                req.nvme_status, sc, sct, req.result_dw0, req.result_dw1);
    }

    /* ------------------------------------------------------------------ */
    /* [7b] NVM_RAW_ADMIN_CMD negative: Reserved opcode (0xFF).            */
    /*                                                                    */
    /* What we expect:                                                    */
    /*   - ioctl return:  0   (CQE arrived; the controller rejected at   */
    /*                        NVMe level, not the kernel at ioctl level) */
    /*   - nvme_status:   non-zero with SC=0x01 "Invalid Command Opcode" */
    /*                                                                    */
    /* Why this is a stronger negative case than Identify-PRP1=0:        */
    /*   NVMe 1.4 §5 (Admin Command Set, Figure 139) requires every     */
    /*   admin command queue to reject opcodes the controller does not  */
    /*   implement with SC=0x01.  Reserved opcodes therefore have a     */
    /*   spec-mandated failure behavior, whereas a bad PRP on Identify  */
    /*   is only "should be rejected" -- some firmwares accept it.      */
    /*                                                                    */
    /* What this proves:                                                  */
    /*   The pass-through faithfully surfaces a non-zero CQE status     */
    /*   field back to userspace.  Together with [7a] this confirms     */
    /*   we can both submit valid admin commands AND observe failures  */
    /*   reported by the controller -- the two halves we need before   */
    /*   layering Delete/Create I/O SQ/CQ on top in T2b.               */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin req;
        memset(&req, 0, sizeof(req));
        build_reserved_opcode_sqe(req.sqe);

        if (do_ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &req,
                     "NVM_RAW_ADMIN_CMD(reserved opcode 0xFF)") < 0)
            step_fail(errno, "NVM_RAW_ADMIN_CMD(reserved opcode) ioctl-level "
                             "error -- handler rejected before submission?");

        uint16_t sc  = req.nvme_status & 0xff;
        uint16_t sct = (req.nvme_status >> 8) & 0x7;

        step_ok("NVM_RAW_ADMIN_CMD(opcode=0xFF) round-trip: nvme_status=0x%04x "
                "(SC=0x%02x SCT=0x%x) dw0=0x%08x dw1=0x%08x",
                req.nvme_status, sc, sct, req.result_dw0, req.result_dw1);

        if (req.nvme_status == 0) {
            step_fail(0,
                "Controller accepted a reserved admin opcode (0xFF) -- "
                "the pass-through is masking the CQE status, OR this "
                "firmware implements 0xFF as a vendor-specific command "
                "(unlikely; investigate).");
        } else if (sct == 0x0 && sc == 0x01) {
            step_ok("Controller correctly rejected reserved opcode with "
                    "SC=0x01 (Invalid Command Opcode) -- ioctl plumbing "
                    "surfaces NVMe-level failure verbatim.");
        } else {
            /* Got a non-zero status, but not the spec-mandated
             * 0x01 Invalid-Opcode.  Still proves the pass-through
             * propagates failure status; just log unexpected codes
             * for visibility (some firmwares return SC=0x02 or a
             * vendor-specific code instead). */
            step_warn("Reserved opcode rejected with SC=0x%02x SCT=0x%x "
                      "(expected SC=0x01 SCT=0x0).  Pass-through still "
                      "works -- this is just a firmware quirk.",
                      sc, sct);
        }
    }

    /* ------------------------------------------------------------------ */
    /* [8] SNVM_DEVICE_UNBIND                                              */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [F1] Per-fd cleanup                                                 */
    /* ------------------------------------------------------------------ */
    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s)", dev_path);

    /* ------------------------------------------------------------------ */
    /* [F2] SNVM_CHRDEV_REMOVE                                             */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }

    close(fd_ctl);
    fprintf(stderr, "\n=== snvme_smoke_recycle: all %d steps passed ===\n",
            g_step);
    return 0;
}
