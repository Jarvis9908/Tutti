#ifndef __NVM_INTERNAL_LINUX_IOCTL_H__
#define __NVM_INTERNAL_LINUX_IOCTL_H__
#include <asm-generic/ioctl.h>
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#ifndef __KERNEL__ //the header shared between kernel and user space
#include <stddef.h>
#include <stdint.h>
#endif

#define NVM_IOCTL_TYPE          0x80
#define NVM_CTRL_IOCTL_TYOE     0x90

#define DISK_NAME_LEN       32

#define DISK_NAME_COPY(dest, src) \
    memcpy(dest, src, DISK_NAME_LEN)


/*
 * Memory map request (NVM_MAP_HOST_MEMORY / NVM_MAP_DEVICE_MEMORY /
 * NVM_MAP_DEVICE_QUEUE_MEMORY).
 *
 * ABI rev note (queue-group plan, step B2):
 *
 *   Two new fields (group_id + reserved) are appended.  This
 *   breaks the old _IOC_SIZE so any pre-B2 userspace binary
 *   trying the legacy 32-byte layout will be rejected with
 *   -ENOTTY at ioctl entry rather than silently mis-decoding.
 *   The project is open-source / pre-stable; we deliberately do
 *   not preserve the legacy layout.
 *
 * ABI rev note (queue-group plan, step B6):
 *
 *   `reserved` (uint32_t) split into `map_kind` (1 byte) +
 *   `reserved0` (3 bytes).  This is a *layout-compatible* change:
 *   pre-B6 binaries set `reserved = 0`, which maps to
 *   `map_kind = NVM_MAP_KIND_UNSPECIFIED` and triggers the
 *   legacy code path (kind-tag is ignored, group_id discriminates
 *   between legacy global lists and per-fd queue groups exactly
 *   as in B2..B5).  _IOC_SIZE is unchanged.
 *
 * Modes (kernel branches on group_id and map_kind):
 *
 *   group_id == 0   "legacy" mode.  The map is registered against
 *                   the controller-global host_list / device_list /
 *                   device_queue_list as before, and is NOT attached
 *                   to any per-fd queue group.  This is the path
 *                   used by the legacy NVM_SET_IOQ_NUM bring-up
 *                   (which still expects ioq_idx / is_cq tags) and
 *                   by ad-hoc data-buffer registrations from
 *                   libnvm/dma.cpp.
 *
 *   group_id != 0   "new" mode.  The kernel looks up the group_id
 *                   in the calling fd's own->groups list (the
 *                   group must have been created via
 *                   NVM_CREATE_QUEUE_GROUP on the same fd; cross-fd
 *                   ids are -ENOENT).  The map is registered in
 *                   the global list AND linked into group->maps so
 *                   that NVM_DESTROY_QUEUE_GROUP / fd-close
 *                   cascade-cleanup can release it without the
 *                   user having to call NVM_UNMAP_*.
 *
 *                   ioq_idx / is_cq are IGNORED in this mode --
 *                   per-queue identity is established later by
 *                   NVM_ADD_USER_QUEUE (B3), which takes
 *                   (group_id, sq_vaddr, cq_vaddr).  Userspace
 *                   should still set them to -1 / -1 for
 *                   forward-compat with possible future use.
 *
 * map_kind (B6):
 *
 *   When map_kind is non-zero (any of RING_SQ / RING_CQ / DATA),
 *   the kernel knows what role the buffer plays without having to
 *   infer it from vaddr alignment or NVM_ADD_USER_QUEUE timing:
 *
 *     RING_SQ / RING_CQ:  must carry a non-zero group_id.  The
 *                         map is linked onto g->maps as in B2.
 *                         NVM_ADD_USER_QUEUE then enforces that
 *                         pairs[i].sq_vaddr resolves to a RING_SQ
 *                         map and pairs[i].cq_vaddr to a RING_CQ
 *                         map; mismatched kinds are rejected with
 *                         -EINVAL up front (instead of issuing
 *                         Create I/O SQ on a data buffer and
 *                         silently corrupting controller state).
 *
 *     DATA:               linked onto a per-fd `data_maps` list,
 *                         NOT g->maps.  group_id, if supplied, is
 *                         IGNORED for lifecycle purposes -- DATA
 *                         maps survive NVM_DESTROY_QUEUE_GROUP and
 *                         only get released on fd close (or by an
 *                         explicit NVM_UNMAP_*).  This decouples
 *                         the data-buffer DMA pool's lifetime from
 *                         the queue group's, which matches the
 *                         common usage pattern (one long-lived
 *                         data pool, many short-lived queue
 *                         groups).
 *
 *     UNSPECIFIED (= 0):  legacy / pre-B6 binary.  The kernel
 *                         falls back to the B2..B5 behaviour:
 *                         group_id alone discriminates between
 *                         per-fd group attachment and the
 *                         controller-global lists, and
 *                         NVM_ADD_USER_QUEUE's vaddr lookup does
 *                         not consult the kind tag.  No new
 *                         compile-time UAPI users should set this.
 *
 * `reserved0` is MBZ; future revisions may add e.g. NUMA hints
 * without another ABI break.
 */

/* Map-kind enum used by struct nvm_ioctl_map::map_kind (B6).        */
enum nvm_map_kind {
    NVM_MAP_KIND_UNSPECIFIED = 0,   /* legacy / pre-B6 binary       */
    NVM_MAP_KIND_RING_SQ     = 1,   /* user IO Submission Queue ring */
    NVM_MAP_KIND_RING_CQ     = 2,   /* user IO Completion Queue ring */
    NVM_MAP_KIND_DATA        = 3,   /* PRP / SGL data buffer         */
};

struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
    int         ioq_idx;        /* legacy mode only; -1 in new mode  */
    int         is_cq;          /* legacy mode only; -1 in new mode  */
    uint32_t    group_id;       /* 0 = legacy; nonzero = new mode    */
    uint8_t     map_kind;       /* enum nvm_map_kind (B6)            */
    uint8_t     reserved0[3];   /* MBZ; future extension             */
};

/*
 * Per-controller info returned by NVM_GET_DEV_INFO.
 *
 * ABI rev (queue-group plan, step B3): four new fields appended.
 * The ioctl number's _IOC_SIZE is therefore different from the
 * pre-B3 layout; old userspace binaries get -ENOTTY at ioctl
 * entry rather than silently mis-decoding a shorter struct.
 *
 * All NEW fields are sourced either from NVMe spec / controller
 * CAP register (q_depth, bar0_size, max_user_qid) or from snvme
 * compile-time limits (max_queues_per_group); userspace can rely
 * on them as the single source of truth and MUST NOT recompute
 * them from CAP independently.
 *
 * Notes on existing fields:
 *
 *   nr_user_q      Legacy field set by NVM_SET_IOQ_NUM.  In the
 *                  new (queue-group) flow userspace does NOT call
 *                  NVM_SET_IOQ_NUM -- this field then reads back
 *                  as 0 and should be ignored.  Kept for backward
 *                  compatibility with the bind-time bring-up path.
 *
 *   start_cq_idx   First QID available to user IOQs.  In the new
 *                  flow this equals dev->online_queues (admin +
 *                  kernel IOQ count) and bounds the bottom of the
 *                  user QID pool; the top is max_user_qid below.
 */
struct nvm_ioctl_dev
{
    /* === existing (semantics unchanged) === */
    uint32_t    nr_user_q;          /* legacy NVM_SET_IOQ_NUM result; 0 in new flow */
    uint32_t    start_cq_idx;       /* first QID available to user IOQs */
    uint8_t     dstrd;              /* CAP.DSTRD: doorbell stride exponent */
    size_t      max_data_size;      /* CTRL.MDTS in bytes */
    size_t      block_size;         /* 1 << ns->lba_shift */
    char        disk_name[DISK_NAME_LEN];

    /* === NEW for B3 === */
    uint16_t    q_depth;            /* NVMe CAP.MQES + 1, clamped: applies to    */
                                    /* ALL user queues (snvme does not support   */
                                    /* per-queue depth)                          */
    uint16_t    reserved0;          /* MBZ */
    uint32_t    bar0_size;          /* pci_resource_len(pdev, BAR0); userspace   */
                                    /* mmaps up to this size for doorbells       */
    uint32_t    max_user_qid;       /* highest QID kernel will hand out via      */
                                    /* NVM_ADD_USER_QUEUE; user pool is          */
                                    /* [start_cq_idx, max_user_qid]              */
    uint32_t    max_queues_per_group;  /* echoes NVM_MAX_QUEUES_PER_GROUP */
    /*
     * Identify Controller SGLS dword (NVMe spec figure 247 byte 536-539).
     *   bit 0: SGL data block descriptor supported
     *   bit 1: keyed SGL data block descriptor supported
     *   bit 2: SGL bit bucket descriptor supported
     *   bit 16: SGL byte-aligned virtual contiguous data buffer
     *   bit 17: SGL transport DMA byte-aligned data buffer
     *   bit 19: SGL address field shall specify offset
     *   bit 20: transport SGL data block descriptor supported
     *   bit 21: keyed SGL data block, key>=0
     * Use SGL only when (sgls & 0x3) != 0 (bit 0 OR 1 set).  0 means
     * controller is PRP-only.
     */
    uint32_t    sgl_supported;      /* echo of dev->ctrl.sgls */
    uint32_t    reserved1[5];       /* MBZ; future extension */
};

/*
 * Queue budget descriptor passed to NVM_SET_IOQ_NUM.
 *
 * Replaces the historical (and undersized) "pack ioq count into
 * nvm_ioctl_map.ioq_idx" pattern.  Lets userspace declare, in one
 * shot:
 *
 *   - how many IOQs it intends to register for itself (ioq_num /
 *     on_host -- same semantics as the legacy two fields),
 *   - what cap to put on the kernel side so the controller has room
 *     to grant both sets (cap_kernel_ioq, nr_write, nr_poll),
 *   - how to slice the user share across logical owners (typically
 *     GPUs) via the groups array.
 *
 * Layout is fixed-size (groups[NVM_MAX_QUEUE_GROUPS]) so the ioctl
 * is single-copy and _IOC_SIZE is stable.  Eight groups matches the
 * 8-GPU H20 / H100 SXM topology and is the upper bound exposed via
 * sys_config.yaml's queue_groups list.
 */
#define NVM_MAX_QUEUE_GROUPS  8

struct nvm_queue_group {
    uint32_t    owner_id;    /* opaque tag, typically a GPU id.        */
    uint32_t    count;       /* SQ + CQ entry count for this group     */
                             /* (kernel unit: 2 * QueuePair count).    */
                             /* Kernel enforces                        */
                             /*   sum(groups[].count) == ioq_num       */
                             /* when nr_groups > 0; see pci.c          */
                             /* NVM_SET_IOQ_NUM handler.               */
    int32_t     numa_node;   /* doc-only hint; kernel does NOT enforce */
                             /* this.  -1 = "don't care".              */
    uint32_t    reserved;    /* MBZ.                                   */
};

#define NVM_QUEUE_SETUP_F_ON_HOST   (1U << 0)
                                     /* Set => user IOQ ring memory   */
                                     /* lives in host RAM (pinned via */
                                     /* NVM_MAP_HOST_MEMORY).  Clear  */
                                     /* => GPU device memory (pinned  */
                                     /* via NVM_MAP_DEVICE_QUEUE_     */
                                     /* MEMORY).                      */

struct nvm_ioctl_setup
{
    uint32_t    ioq_num;        /* total user-side IOQ count (sum of  */
                                /* SQ+CQ pair counts; legacy field).  */
    uint32_t    flags;          /* NVM_QUEUE_SETUP_F_*.               */
    uint32_t    cap_kernel_ioq; /* upper bound on kernel-side IOQ     */
                                /* count requested from the           */
                                /* controller.  0 = use the upstream  */
                                /* default (num_possible_cpus()).     */
                                /* Set this when the controller's     */
                                /* MSI-X grant is smaller than the    */
                                /* host CPU count and you want the    */
                                /* user share path to actually        */
                                /* activate instead of falling back   */
                                /* to dma_alloc_coherent.             */
    uint32_t    nr_write;       /* per-BDF override of the            */
                                /* write_queues module parameter;     */
                                /* 0 = leave at module default.       */
    uint32_t    nr_poll;        /* ditto for poll_queues.             */
    uint32_t    nr_groups;      /* <= NVM_MAX_QUEUE_GROUPS.  0 means  */
                                /* "no per-owner partitioning";       */
                                /* groups[] is then ignored.          */
    uint32_t    reserved[2];    /* MBZ; future extension.             */
    struct nvm_queue_group  groups[NVM_MAX_QUEUE_GROUPS];
};

struct pci_device_addr{ // Removed redundant definition
    int domain;
    int bus;
    int slot;
    int func;
};

/*
 * Raw admin command pass-through (NVM_RAW_ADMIN_CMD).
 *
 * Lets userspace send any NVMe admin SQE through the controller's
 * snvme-owned admin queue and read back the completion's DW0/DW1
 * (NVMe spec "Command Specific" + reserved) plus the 16-bit status
 * field (SC | SCT | DNR | MORE) verbatim.
 *
 * Originally added to support per-queue recycle (Delete I/O SQ ->
 * Delete I/O CQ -> Create I/O CQ -> Create I/O SQ; NVMe 1.4 §5.4 /
 * §5.5) without re-running snvme bring-up.  Generic enough to also
 * carry Abort (§5.1), Get Log Page (§5.10), vendor admin commands
 * etc.; the kernel handler does NOT inspect the opcode and only
 * forwards via snvme_submit_sync_cmd().
 *
 * Restrictions enforced by the kernel handler:
 *   - controller must be bound (NVME_CTRL_LIVE).
 *   - the caller's fd is a /dev/ssnvme<N> chrdev (snvm_dev_map_ioctl
 *     dispatcher checks ctrl_find_by_inode).
 *   - data-buffer admin commands are NOT supported in this revision
 *     (`buffer` field is reserved/ignored; bufflen=0 is hard-coded).
 *     Add a follow-up path if Get Log Page / Set Features with
 *     payloads is ever needed.
 *
 * Field semantics:
 *   sqe        -- 64 raw bytes, byte-for-byte equivalent to one
 *                 entry in the NVMe admin SQ ring.  Userspace
 *                 fills in opcode, CID, NSID, CDW10..15, etc.
 *                 The kernel re-uses the CID by handing this to
 *                 snvme_submit_sync_cmd which manages its own tag.
 *   result_dw0 -- CQE DW0 (command-specific result; e.g.
 *                 Create I/O SQ returns 0 on success).  Populated
 *                 only on successful return.
 *   result_dw1 -- CQE DW1.  Mostly reserved; populated for the
 *                 same reason as DW0.
 *   nvme_status-- NVMe status field from CQE DW3 (15:1 = SC|SCT|
 *                 More|DNR, bit 0 = phase).  Populated even when
 *                 the ioctl returns 0 -- userspace must check
 *                 (status & 0xFFFE) == 0 for spec success.  When
 *                 the ioctl returns negative, this field is
 *                 undefined (the request never reached the
 *                 controller, e.g. -EFAULT on copy_from_user).
 *   reserved   -- MBZ; future extension for nsid / data buffer
 *                 forwarding etc.
 */
struct nvm_ioctl_raw_admin {
    uint8_t     sqe[64];        /* in:  one NVMe admin SQE        */
    uint32_t    result_dw0;     /* out: CQE DW0                   */
    uint32_t    result_dw1;     /* out: CQE DW1                   */
    uint16_t    nvme_status;    /* out: CQE DW3[31:17] (SC|SCT|...)*/
    uint16_t    reserved0;      /* MBZ                            */
    uint32_t    reserved1[4];   /* MBZ; future expansion          */
};

/*
 * Per-fd queue group container (NVM_CREATE_QUEUE_GROUP /
 * NVM_DESTROY_QUEUE_GROUP).
 *
 * A "queue group" is the runtime resource container that future
 * NVM_ADD_USER_QUEUE / NVM_RECYCLE_USER_QUEUE calls hang off of.
 * It is the kernel-side dual of "one logical client" (typically:
 * one process binding to one GPU on one NVMe controller).
 *
 * Lifecycle / ownership:
 *
 *   * A group is created with NVM_CREATE_QUEUE_GROUP on an open
 *     /dev/ssnvme<N> fd.  The kernel returns an opaque, globally
 *     unique group_id.  The group is bound to (file, ctrl); fd
 *     close cascades destroy automatically.
 *
 *   * A group is destroyed with NVM_DESTROY_QUEUE_GROUP(group_id)
 *     OR implicitly when the owning fd is closed.  In B1 (this
 *     header) the group still has nothing in it; later steps add
 *     map registration (B2) and user IO queues (B3) which are
 *     freed on group destroy in LIFO order.
 *
 *   * A single fd may hold up to NVM_MAX_GROUPS_PER_FD groups.
 *     The default (1) matches the "one client = one group" model
 *     used by NVMeService daemons.
 *
 * group_id namespace:
 *
 *   group_id is opaque to userspace.  Internally allocated via
 *   ida_simple_get(); userspace MUST treat it as a 32-bit cookie
 *   and pass it back verbatim.  Group id 0 is invalid (reserved
 *   to mean "no group" / sentinel).
 *
 * Why a separate ioctl from NVM_SET_IOQ_NUM:
 *
 *   NVM_SET_IOQ_NUM is the legacy, bind-time, per-controller setup
 *   path.  It pre-declares the total user IOQ budget for the entire
 *   controller, then probe consumes it in one shot.  Queue groups
 *   are the new, per-fd, runtime path -- groups can be created and
 *   destroyed at any time after bind, independently of each other.
 *   The two paths coexist; legacy callers see no behavioural change.
 */
#define NVM_MAX_GROUPS_PER_FD       1
#define NVM_MAX_QUEUES_PER_GROUP   16    /* hardcoded cap; see B6 */

struct nvm_ioctl_queue_group {
    uint32_t    group_id;       /* out: kernel-assigned, opaque   */
    uint32_t    flags;          /* MBZ; reserved for future       */
    uint32_t    max_queues;     /* out: per-group queue cap       */
                                /*      (echoes the kernel cap)   */
    uint32_t    reserved[5];    /* MBZ                            */
};

/*
 * Per-queue-pair input/output for NVM_ADD_USER_QUEUE.
 *
 * Userspace pre-registers two ring buffers per queue (one for SQ,
 * one for CQ) via NVM_MAP_HOST_MEMORY / NVM_MAP_DEVICE_MEMORY
 * against the group_id.  ADD_USER_QUEUE then takes the *vaddrs*
 * of those rings -- not map ids -- and the kernel does an O(group
 * maps) lookup to recover the DMA addresses.
 *
 * Ring buffer sizing constraints (userspace MUST satisfy these
 * before NVM_MAP, otherwise Create I/O CQ/SQ will fail at the
 * controller):
 *
 *   SQ buffer >= q_depth * 64 bytes  (NVMe spec: SQE = 64B fixed)
 *   CQ buffer >= q_depth * 16 bytes  (NVMe spec: CQE = 16B fixed)
 *   Both buffers physically contiguous (NVMe spec
 *     "PHYS_CONTIG" flag -- only single-page-coverage is
 *     verified by the current snvme implementation, so q_depth *
 *     entry_size MUST fit in one host page; this is true for
 *     q_depth <= 64 / 256 respectively on a 4 KiB page).
 *   Both buffers page-size aligned (PRP1 has the lower 12 bits
 *     reserved as zero on this code path).
 *
 * q_depth and the page size are obtained from NVM_GET_DEV_INFO.
 *
 * Output fields populated only on overall ioctl success.
 */
struct nvm_user_queue_pair_in {
    uint64_t    sq_vaddr;       /* userspace VA of an SQ ring registered      */
                                /* against this group via NVM_MAP_HOST_MEMORY */
                                /* / NVM_MAP_DEVICE_MEMORY.                   */
    uint64_t    cq_vaddr;       /* same, for the CQ ring                      */
};

struct nvm_user_queue_pair_out {
    uint32_t    sq_doorbell_offset;     /* BAR0 byte offset for SQ tail dbl   */
    uint32_t    cq_doorbell_offset;     /* BAR0 byte offset for CQ head dbl   */
    uint32_t    qid;                    /* informational; userspace does NOT  */
                                        /* need it to write SQEs (NVMe SQE   */
                                        /* has no SQID field).  Useful for   */
                                        /* dmesg correlation only.           */
    uint32_t    reserved;               /* MBZ                                */
};

/*
 * NVM_ADD_USER_QUEUE payload.
 *
 * Submits up to NVM_MAX_QUEUES_PER_GROUP (SQ, CQ) pairs in one
 * ioctl.  The kernel handles them as an all-or-nothing batch:
 * either every pair successfully gets a Create I/O CQ + Create
 * I/O SQ admin command through the controller, or none do (any
 * pairs already created in the same call are unwound via Delete
 * I/O SQ + Delete I/O CQ before the ioctl returns the error).
 *
 * Add operations are incremental: a group may receive multiple
 * NVM_ADD_USER_QUEUE calls as long as cur_queues + nr_pairs <=
 * max_queues.  The matching teardown is NVM_DESTROY_QUEUE_GROUP
 * (drains all queues) or NVM_RECYCLE_USER_QUEUE (re-creates them
 * with the same rings; planned for B5).
 *
 * Pre-conditions enforced by the kernel:
 *
 *   1. Controller must be bound (NVME_CTRL_LIVE).  The user QID
 *      pool is only known once nvme_probe has finished allocating
 *      kernel IOQs -- ADD_USER_QUEUE before bind returns -ENODEV.
 *   2. group_id must belong to the calling fd (cross-fd usage is
 *      -ENOENT, same as B1/B2 isolation rules).
 *   3. nr_pairs in [1, NVM_MAX_QUEUES_PER_GROUP].
 *   4. flags / reserved fields MBZ.
 *   5. each (sq_vaddr, cq_vaddr) must resolve to exactly one map
 *      already registered against this group.  The kernel rejects
 *      with -ENOENT otherwise; rings registered against a
 *      different group (even on the same fd) are not visible.
 *   6. cur_queues + nr_pairs <= max_queues_per_group.  Returns
 *      -EBUSY if the group is full.
 *   7. No two pairs in one call may share the same sq_vaddr or
 *      the same cq_vaddr.  -EINVAL.
 *
 * Output: out_pairs[i] is populated for i < nr_pairs only;
 * trailing entries are left zero.
 */
struct nvm_ioctl_add_user_queue {
    /* in */
    uint32_t    group_id;
    uint32_t    nr_pairs;       /* 1..NVM_MAX_QUEUES_PER_GROUP */
    uint32_t    flags;          /* MBZ */
    uint32_t    reserved[5];    /* MBZ */
    struct nvm_user_queue_pair_in   pairs[NVM_MAX_QUEUES_PER_GROUP];

    /* out */
    struct nvm_user_queue_pair_out  out_pairs[NVM_MAX_QUEUES_PER_GROUP];
};

/* Supported operations */
enum nvm_ioctl_type{
    NVM_MAP_HOST_MEMORY             = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY           = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_QUEUE_MEMORY     = _IOW(NVM_IOCTL_TYPE, 3, struct nvm_ioctl_map),
    NVM_UNMAP_HOST_MEMORY           = _IOW(NVM_IOCTL_TYPE, 4, uint64_t),
    NVM_UNMAP_DEVICE_MEMORY         = _IOW(NVM_IOCTL_TYPE, 5, uint64_t),
    NVM_UNMAP_DEVICE_QUEUE_MEMORY   = _IOW(NVM_IOCTL_TYPE, 6, uint64_t),
    /*
     * NVM_SET_IOQ_NUM now carries a struct nvm_ioctl_setup, NOT the
     * historical nvm_ioctl_map.  This is an ABI break vs.
     * pre-Tutti snvme-5.15-public binaries -- the project is
     * open-source and we deliberately do not keep the legacy
     * payload.  Any new userspace MUST populate nvm_ioctl_setup;
     * the ioctl number has the new _IOC_SIZE baked in, so a stale
     * binary trying the old layout will be rejected with -ENOTTY
     * at ioctl entry rather than silently mis-decoding the request.
     */
    NVM_SET_IOQ_NUM                 = _IOW(NVM_IOCTL_TYPE, 7, struct nvm_ioctl_setup),
    NVM_SET_SHARE_REG               = _IOW(NVM_IOCTL_TYPE, 8, struct nvm_ioctl_dev),
    NVM_GET_DEV_INFO                = _IOR(NVM_IOCTL_TYPE, 9, struct nvm_ioctl_dev),   
    NVM_CLEAR_IOQ_NUM               = _IOW(NVM_IOCTL_TYPE, 10, struct nvm_ioctl_dev),
    /*
     * NVM_RAW_ADMIN_CMD: generic admin SQE forwarder, used to drive
     * per-queue recycle (Delete + Create I/O SQ/CQ) and any other
     * admin-only command that snvme does not need a dedicated
     * ioctl for.  See struct nvm_ioctl_raw_admin above for the
     * payload contract.
     */
    NVM_RAW_ADMIN_CMD               = _IOWR(NVM_IOCTL_TYPE, 11, struct nvm_ioctl_raw_admin),
    /*
     * NVM_CREATE_QUEUE_GROUP / NVM_DESTROY_QUEUE_GROUP:
     *   Per-fd runtime container for user IO queues.  Step B1 of
     *   the queue-group plan -- container only, no NVMe resources
     *   yet.  See struct nvm_ioctl_queue_group above.
     *
     *   CREATE returns the new group_id in the payload.
     *   DESTROY takes the group_id directly (uint32_t payload).
     *   FD close auto-destroys all groups owned by that fd.
     */
    NVM_CREATE_QUEUE_GROUP          = _IOWR(NVM_IOCTL_TYPE, 12, struct nvm_ioctl_queue_group),
    NVM_DESTROY_QUEUE_GROUP         = _IOW (NVM_IOCTL_TYPE, 13, uint32_t),
    /*
     * NVM_ADD_USER_QUEUE: create (SQ, CQ) pairs against a queue
     * group, all-or-nothing batch.  Userspace pre-registers ring
     * buffers via NVM_MAP_HOST_MEMORY/NVM_MAP_DEVICE_MEMORY (with
     * group_id), then passes the ring vaddrs in the payload's
     * pairs[] array.  See struct nvm_ioctl_add_user_queue above
     * for the full contract; this is the B3 step in the
     * queue-group plan.
     */
    NVM_ADD_USER_QUEUE              = _IOWR(NVM_IOCTL_TYPE, 14, struct nvm_ioctl_add_user_queue),
    /*
     * NVM_SET_KERNEL_IOQ_CAP: B3 cap-only path.
     *
     * Pre-bind, declare an upper bound on how many IO queues the
     * kernel side may consume from the controller's granted IOQ
     * count.  Whatever the controller actually grants (typically
     * limited by its MSI-X vector count) above this cap becomes
     * available to the NVM_ADD_USER_QUEUE user pool.
     *
     * Distinct from NVM_SET_IOQ_NUM:
     *   - NVM_SET_IOQ_NUM is the legacy bring-up flow that also
     *     declares ioq_num (pre-registered user CQ count via
     *     NVM_MAP_*), use_sreg, and per-owner groups[].  It is
     *     intended for the dma_alloc_coherent-vs-user-share probe
     *     path and forces ctrl->use_sreg / ctrl->ioq_num state.
     *   - NVM_SET_KERNEL_IOQ_CAP does ONLY the cap.  ctrl->ioq_num
     *     stays at 0, use_sreg stays at 0, the probe path runs as
     *     plain in-tree-style nvme.  This is the right knob when
     *     userspace plans to create IOQs dynamically post-bind via
     *     NVM_ADD_USER_QUEUE rather than declaring them upfront.
     *
     * Payload is a uint32_t = the cap value.  Zero means "no cap"
     * and clears any previously-set cap on this controller.
     *
     * Must be called before SNVM_DEVICE_BIND for the cap to take
     * effect; calls after bind set ctrl->setup.cap_kernel_ioq but
     * have no probe to apply it to.
     */
    NVM_SET_KERNEL_IOQ_CAP          = _IOW (NVM_IOCTL_TYPE, 15, uint32_t),
};

// snvm_ctrl_ioctl_type
//
// Note: opcode 5 (formerly SNVM_CACULATE_PCIDISTANCE) was removed. Do NOT
// reuse it for a new command for at least one release cycle, otherwise old
// userspace binaries will silently get a different result.
enum snvm_ctrl_ioctl_type{
    SNVM_DEVICE_BIND                = _IOW(NVM_CTRL_IOCTL_TYOE, 1, struct pci_device_addr),
    SNVM_DEVICE_UNBIND              = _IOW(NVM_CTRL_IOCTL_TYOE, 2, struct pci_device_addr),
    SNVM_CHRDEV_CREATE              = _IOWR(NVM_CTRL_IOCTL_TYOE, 3, struct pci_device_addr),
    SNVM_CHRDEV_REMOVE              = _IOW(NVM_CTRL_IOCTL_TYOE, 4, struct pci_device_addr),
};


/* SNVME initiazation process*/
/*
1. Use NVM_SET_IOQ_NUM to declare both the kernel-side IO-queue
   budget AND the user-side IO-queue partition.  The argument is a
   struct nvm_ioctl_setup; key fields:
     ioq_num         total user IOQ count (kernel reserves room for
                     ioq_num CQs on top of cap_kernel_ioq)
     cap_kernel_ioq  upper bound on kernel-side IOQ count requested
                     from the controller.  Set this to less than
                     num_possible_cpus() when the controller's
                     MSI-X grant is smaller than the host CPU count
                     (e.g. Intel DC SSD: MSI-X=136 vs 192 vCPUs),
                     otherwise the user share path falls back to
                     dma_alloc_coherent.
     nr_groups       optional per-owner (typically per-GPU) split.
2. Use NVM_MAP_HOST_MEMORY/NVM_MAP_DEVICE_MEMORY/
   NVM_MAP_DEVICE_QUEUE_MEMORY to register DMA addresses for each
   declared user IOQ; the total must match ioq_num.
3. Use NVM_SET_SHARE_REG to flip the use_sreg gate to 1; the next
   SNVM_DEVICE_BIND will then commit the user share at probe time
   instead of letting upstream nvme allocate the queue pages.
4. Use SNVM_DEVICE_BIND to detach the in-tree nvme driver from the
   target BDF and bind snvme; probe consumes the use_sreg gate and
   commits the kernel/user IOQ split that NVM_SET_IOQ_NUM declared.
*/
#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
