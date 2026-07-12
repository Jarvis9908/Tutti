/**
 * snvm_recover_addr.cpp -- manually re-issue SNVM_DEVICE_UNBIND +
 * SNVM_CHRDEV_REMOVE for a specific BDF, without rmmod/insmod.
 *
 * SCOPE, CONFIRMED BY READING THE KERNEL SOURCE (do not overclaim
 * beyond this -- an earlier revision of this docstring did, and was
 * wrong; see PORTING.md's troubleshooting table for the full writeup):
 *
 *   This tool fixes ONLY the "struct ctrl left in ctrl_list" state
 *   that makes a subsequent SNVM_CHRDEV_CREATE for the same BDF fail
 *   with errno=14 after an owner (tutti_daemon / nvmeservice_daemon)
 *   was SIGKILL'd instead of exiting via SIGINT/SIGTERM.  (As of this
 *   revision that specific errno=14 is ALSO independently fixed at
 *   the source level -- snvm_chrdev_helper's tlinux-5.4 baseline was
 *   missing the 5.15 baseline's "create && ctrl" idempotent branch --
 *   but this tool remains useful against an un-rebuilt/un-reloaded
 *   module, or any other lingering-ctrl scenario.)
 *
 *   It does **NOT** fix a `rmmod snvme` "Module snvme is in use"
 *   refcount leak: neither SNVM_DEVICE_UNBIND (snvm_unbind_driver)
 *   nor SNVM_CHRDEV_REMOVE (snvm_chrdev_helper) calls
 *   try_module_get/module_put ANYWHERE in their implementations, so
 *   re-issuing them changes nothing about the module refcount no
 *   matter what state the BDF was in.  The Linux cdev framework
 *   already auto-pairs try_module_get/module_put on every open()/
 *   fd-close of /dev/snvm_control and /dev/ssnvme<N> regardless of
 *   the driver's own .release -- so a SIGKILL'd owner does not, by
 *   itself, leak that refcount either.  If rmmod still says "in use"
 *   after running this tool, check `mount | grep snvme` and
 *   `lsof /dev/snvme[0-9]* | grep -v ssnvme` (the block device and
 *   raw admin chardev, NOT /dev/snvm_control/ssnvme*) -- see
 *   `scripts/reset_snvme.sh` step 2b, which now checks both
 *   automatically, and PORTING.md's troubleshooting table.
 *
 * Usage:
 *   sudo ./snvm_recover_addr <bdf> [<bdf> ...]
 *   e.g. sudo ./snvm_recover_addr 0000:08:00.0 0000:4b:00.0
 *
 * Safe to run speculatively (e.g. right before starting the daemon):
 * if the BDF's state is already clean, both ioctls just fail with
 * ENODEV/EINVAL-ish errors that are reported but not fatal to the
 * overall run -- this tool always tries every BDF given.
 */

#include "ioctl.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

static bool parse_bdf(const char* s, struct pci_device_addr* out) {
    return std::sscanf(s, "%x:%x:%x.%x", &out->domain, &out->bus,
                        &out->slot, &out->func) == 4;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <bdf> [<bdf> ...]\n"
                              "  e.g.  %s 0000:08:00.0 0000:4b:00.0\n",
                      argv[0], argv[0]);
        return 1;
    }

    int fd = ::open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        std::fprintf(stderr, "open(/dev/snvm_control): %s\n", std::strerror(errno));
        return 1;
    }

    int n_ok = 0, n_total = 0;
    for (int i = 1; i < argc; ++i) {
        struct pci_device_addr addr{};
        if (!parse_bdf(argv[i], &addr)) {
            std::fprintf(stderr, "%s: bad BDF (want DDDD:BB:DD.F)\n", argv[i]);
            continue;
        }
        ++n_total;

        struct pci_device_addr a_unbind = addr;
        int rc_unbind = ::ioctl(fd, SNVM_DEVICE_UNBIND, &a_unbind);
        int err_unbind = errno;

        struct pci_device_addr a_remove = addr;
        int rc_remove = ::ioctl(fd, SNVM_CHRDEV_REMOVE, &a_remove);
        int err_remove = errno;

        std::printf("%s: UNBIND %s (%s) | CHRDEV_REMOVE %s (%s)\n", argv[i],
                    rc_unbind == 0 ? "ok" : "failed",
                    rc_unbind == 0 ? "-" : std::strerror(err_unbind),
                    rc_remove == 0 ? "ok" : "failed",
                    rc_remove == 0 ? "-" : std::strerror(err_remove));
        if (rc_remove == 0) ++n_ok;
    }

    ::close(fd);
    std::printf("Recovered %d/%d BDF(s) (CHRDEV_REMOVE succeeded).\n", n_ok, n_total);
    return 0;
}
