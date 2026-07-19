/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *
 */

#include <linux/pci.h>
#include <linux/pci_regs.h>
#include <linux/pci_ids.h>
#include <linux/seq_file.h>

#include "nvfs-pci.h"
#include "nvfs-core.h"
#include <linux/seq_file.h>
#include <linux/topology.h>

// PCI_EXPRESS_LINK_STATUS_REGISTER : LinkSpeed  :4 bits, LinkWidth  :6 bits

// from drivers/pci/pci.h.
const unsigned char nvfs_pcie_link_speed_table[MAX_LNKSPEED_ENTRIES] = {
	PCI_SPEED_UNKNOWN,		/* 0 */
	PCIE_SPEED_2_5GT,		/* 1 */
	PCIE_SPEED_5_0GT,		/* 2 */
	PCIE_SPEED_8_0GT,		/* 3 */
	PCIE_SPEED_16_0GT,		/* 4 */

	PCI_SPEED_UNKNOWN,		/* 6 */
	PCI_SPEED_UNKNOWN,		/* 7 */
	PCI_SPEED_UNKNOWN,		/* 8 */
	PCI_SPEED_UNKNOWN,		/* 9 */
	PCI_SPEED_UNKNOWN,		/* A */
	PCI_SPEED_UNKNOWN,		/* B */
	PCI_SPEED_UNKNOWN,		/* C */
	PCI_SPEED_UNKNOWN,		/* D */
	PCI_SPEED_UNKNOWN,		/* E */
	PCI_SPEED_UNKNOWN		/* F */
};

const unsigned char nvfs_pcie_link_width_table[MAX_LNKWIDTH_ENTRIES] = {
	PCIE_LNK_WIDTH_RESRV,   /* 0 */
	PCIE_LNK_X1,            /* 1 */
	PCIE_LNK_X2,            /* 2 */
	PCIE_LNK_X4,            /* 3 */
	PCIE_LNK_X8,            /* 4 */
	PCIE_LNK_X12,           /* 5 */
	PCIE_LNK_X16,           /* 6 */
	PCIE_LNK_X32,           /* 7 */
	PCIE_LNK_WIDTH_UNKNOWN, /* 8 */
	PCIE_LNK_WIDTH_UNKNOWN, /* 9 */
	PCIE_LNK_WIDTH_UNKNOWN, /* A */
	PCIE_LNK_WIDTH_UNKNOWN, /* B */
	PCIE_LNK_WIDTH_UNKNOWN, /* C */
	PCIE_LNK_WIDTH_UNKNOWN, /* D */
	PCIE_LNK_WIDTH_UNKNOWN, /* E */
	PCIE_LNK_WIDTH_UNKNOWN, /* F */
};
