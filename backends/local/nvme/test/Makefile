# Build the L1 (libnvm-level) sanity test.
#
# This directory hosts smokes that live ONE layer above the raw-ioctl
# smokes in backends/local/kernel_modules/test/ -- i.e. they link
# against libnvm.so and exercise the C API surface (nvm_ctrl_init,
# nvm_dma_map_*, etc.) rather than poking the snvme ioctl interface
# directly.
#
# Layered test inventory (each layer isolated in its own directory):
#
#   backends/local/kernel_modules/test/   -- L0: raw ioctl smokes
#   backends/local/nvme/test/             -- L1: libnvm C API smokes  (this dir)
#   backends/local/NVMeService/test/      -- L2: NVMeService client/daemon
#                                                IPC smokes  (planned)
#   filesystems/.../test/                 -- L3: Controller / libgeminifs
#                                                smokes  (planned)
#
# Initial L1 binaries:
#
#   snvme_smoke_libnvm  -- Commit 1 of L1: validates the B3/B6 DMA API
#                          (nvm_dma_map_data_*, nvm_dma_map_ring_*) and
#                          confirms the legacy nvm_dma_map_host(-1,-1)
#                          path still travels the UNSPECIFIED fallback
#                          branch through the kernel.
#                          DESTRUCTIVE: binds the controller.
#
# Build artefacts depend on libnvm.so + its public headers.  We resolve
# both out of the CMake build tree at $(REPO_ROOT)/build/ by default;
# override BUILD_DIR= if you build out-of-tree elsewhere.

THIS_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# THIS_DIR  = .../backends/local/nvme/test/
# REPO_ROOT = .../  (four levels up)
REPO_ROOT := $(abspath $(THIS_DIR)/../../../..)

BUILD_DIR ?= $(REPO_ROOT)/build

LIBNVM_INC := $(REPO_ROOT)/backends/local/nvme/libnvm/include
LIBNVM_LIB := $(BUILD_DIR)/lib

# Pre-flight: libnvm.so must exist.  If not, point the user at the
# right build command instead of producing a cryptic ld error.
LIBNVM_SO := $(LIBNVM_LIB)/libnvm.so

TARGET_LIBNVM := snvme_smoke_libnvm
TARGET_LIBNVM_B3 := snvme_smoke_libnvm_b3
TARGET_LIBNVM_IO := snvme_smoke_libnvm_io
TARGET_LIBNVM_ROLE := snvme_smoke_libnvm_role

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 \
           -I$(LIBNVM_INC)
LDFLAGS ?= -L$(LIBNVM_LIB) -Wl,-rpath,$(LIBNVM_LIB) -lnvm

# nvcc path -- mirrors the L0 GPU smoke Makefile so users can override
# CUDA_ARCH or NVCC the same way.
NVCC       ?= nvcc

DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap \
                       --format=csv,noheader 2>/dev/null \
                   | head -n1 \
                   | sed -e 's/[[:space:]]//g' -e 's/\.//' )
ifneq ($(DETECTED_CC),)
CUDA_ARCH  ?= sm_$(DETECTED_CC)
else
CUDA_ARCH  ?= sm_80
endif

NVCCFLAGS  ?= -O2 -std=c++17 -arch=$(CUDA_ARCH) -I$(LIBNVM_INC) \
              -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter
NVCC_LDFLAGS ?= -L$(LIBNVM_LIB) -Xlinker -rpath=$(LIBNVM_LIB) -lnvm \
                -lcudart

HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

.PHONY: all clean check-libnvm run

ifeq ($(HAVE_NVCC),yes)
all: check-libnvm $(TARGET_LIBNVM) $(TARGET_LIBNVM_B3) $(TARGET_LIBNVM_IO) $(TARGET_LIBNVM_ROLE)
else
all: check-libnvm $(TARGET_LIBNVM) $(TARGET_LIBNVM_B3)
	@echo "[i] nvcc not found; skipping $(TARGET_LIBNVM_IO) + $(TARGET_LIBNVM_ROLE)."
	@echo "    Install CUDA Toolkit (or set NVCC=...) and re-run make to enable it."
endif

check-libnvm:
	@if [ ! -f "$(LIBNVM_SO)" ]; then \
	    echo "ERROR: $(LIBNVM_SO) not found."; \
	    echo "       Build libnvm first:"; \
	    echo "         cd $(BUILD_DIR) && cmake --build . --target libnvm -j"; \
	    echo "       Or override BUILD_DIR= if your build tree lives elsewhere."; \
	    exit 1; \
	fi

$(TARGET_LIBNVM): snvme_smoke_libnvm.c $(LIBNVM_SO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_LIBNVM_B3): snvme_smoke_libnvm_b3.c $(LIBNVM_SO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_LIBNVM_IO): snvme_smoke_libnvm_io.cu $(LIBNVM_SO)
	@echo "[+] nvcc: CUDA_ARCH=$(CUDA_ARCH) (override with 'make CUDA_ARCH=sm_XX')"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCC_LDFLAGS)

$(TARGET_LIBNVM_ROLE): snvme_smoke_libnvm_role.cu $(LIBNVM_SO)
	@echo "[+] nvcc: CUDA_ARCH=$(CUDA_ARCH) (override with 'make CUDA_ARCH=sm_XX')"
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCC_LDFLAGS)

clean:
	rm -f $(TARGET_LIBNVM) $(TARGET_LIBNVM_B3) $(TARGET_LIBNVM_IO) $(TARGET_LIBNVM_ROLE)

# Convenience: PCI=0000:08:00.0 make run
run: $(TARGET_LIBNVM)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run"; exit 1; \
	fi
	sudo ./$(TARGET_LIBNVM) $(PCI)
