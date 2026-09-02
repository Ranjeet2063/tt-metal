// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Support spine of the DRISC drain kernel (drisc_profiler_filler.cpp): the D2H write/credit primitives
// and the poll-free GDDR DMA issue variants.
#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/socket_api.h"
#include "hostdevcommon/profiler_common.h"
#include "internal/tt-1xx/risc_common.h"

#include "experimental/drisc_mode.h"
#include "experimental/gddr_dma.h"

// DRISC firmware doesn't define cb_interface (no CB infra on DRAM cores).
CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

// D2H: write L1 to PCIe host RAM in NOC_MAX_BURST_SIZE chunks. The caller runs
// noc_write_init_state<write_cmd_buf>(NOC_INDEX, vc) once per push -- a push makes several calls (one
// per frame plus the notify), so a per-call init would repeat command-buffer setup that nothing between
// the calls invalidates. Alternating two command buffers to pipeline NIU acceptance was measured
// stall-neutral at the delay-16 saturation wall and deleted.
inline void write_to_host_chunked(uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    while (size) {
        const uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}

// Stop-interruptible replacement for socket_reserve_pages (socket_api.h), which spins on
// `bytes_free < num_bytes` with no escape. Waits on host credit, so it answers the host's stop word:
// lifecycle is the close path's job, not a deadline's. Returning false means "ship nothing"; the caller
// drops the frame -- the heads were already written back, so producers keep running and only capture is
// lost.
inline bool reserve_pages(const SocketSenderInterface& socket, uint32_t num_pages, volatile tt_l1_ptr uint32_t* stop) {
    const uint32_t num_bytes = num_pages * socket.page_size;
    volatile tt_l1_ptr uint32_t* acked = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(socket.bytes_acked_base_addr);
    const uint32_t acked_end = socket.bytes_acked_base_addr + socket.num_downstreams * bytes_acked_size_bytes;
    while (reinterpret_cast<uint32_t>(acked) < acked_end) {
        for (;;) {
            invalidate_l1_cache();
            // bytes_acked is never ahead of bytes_sent, so this cannot underflow
            const uint32_t bytes_free = socket.downstream_fifo_total_size - (socket.bytes_sent - *acked);
            if (bytes_free >= num_bytes) {
                break;
            }
            if (*stop != 0) {
                return false;
            }
        }
        acked =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uint32_t>(acked) + bytes_acked_size_bytes);
    }
    return true;
}

// dma_async_write/read (gddr_dma.h) poll the engine's ready status before every issue -- an MMIO round
// trip plus a per-iteration stack bounce (the volatile union: lw/sw/lw/andi/beqz in the compiled ELF).
// The poll guards the queue-full case, which the filler's pipeline makes unreachable by construction:
// the generation gate keeps stream-0 writes at most ~10 of the 15 the queue holds, and the pump keeps
// stream-1 reads at most 2 of 255. These variants are the same register programming with the poll gone.
inline void dma_write_unchecked(uint8_t stream, uint32_t src_l1, uint64_t dst_gddr, uint32_t size_bytes) {
    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.start_of_packet = 0;
    attrs.f.end_of_packet = 0;
    attrs.f.transfer_start_raw = 1;
    experimental::program_dma_write_addresses_(stream, src_l1, dst_gddr);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}

inline void dma_read_unchecked(uint8_t stream, uint64_t src_gddr, uint32_t dst_l1, uint32_t size_bytes) {
    DmaTxqTransferAttrs_u attrs = {.val = DmaTxqTransferAttrs_DEFAULT};
    attrs.f.transfer_size_words = size_bytes >> 4;
    attrs.f.transfer_start_read = 1;
    experimental::program_dma_read_addresses_(stream, src_gddr, dst_l1);
    WRITE_TX_STREAM_REG(stream, TX_REG_STREAM_TRANSFER_ATTRIBUTES_REG_OFFSET, attrs.val);
}
