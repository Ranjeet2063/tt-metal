// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The streaming profiler's FILLER. Each filler is resident on one DRAM bank's free DRISC and owns a
// slice of the worker grid. It polls each worker's per-RISC SPSC ring tails, gather-reads the live
// runs into packed wire frames in L1 staging, DMAs the frames into a spool ring in its own GDDR
// bank, and a non-blocking pump forwards spool bytes to the host FIFO through its own D2H socket.
// Producers are lossless: a full ring blocks the worker, so the whole pipeline is flow-controlled
// end to end and the producer stall counter is the perturbation ground truth.
//
// Wire format and placement history: tools/drisc_drain/FINDINGS.md. The diagnostic tiers this file
// used to carry (phase counters, service histograms, the drainer's own device zones, the NIU
// footprint sampler, the sync fiducial) and the results buffer they reported through were removed
// on 2026-09-01; tools/drisc_drain/INSTRUMENTATION_NOTES.md records their layouts and findings.
//
// Compile-time arguments are passed by NAME, so adding or retiring one is a local edit -- the
// positional form made every index a JIT cache key and forced dead slots to stay occupied.

#include "drisc_drain_common.hpp"

// ---- kernel arguments and derived constants --------------------------------------------------------

constexpr uint32_t kStageBase = get_named_compile_time_arg_val("stage_base");
constexpr uint32_t kNStage = get_named_compile_time_arg_val("n_stage");
constexpr uint32_t kHeadScratch = get_named_compile_time_arg_val("head_scratch");
constexpr uint32_t kDoneAddr = get_named_compile_time_arg_val("done_addr");
// Host writes 1 = quiesce, 2 = free the NIU.
constexpr uint32_t kStopAddr = get_named_compile_time_arg_val("stop_addr");
constexpr uint32_t kSocketConfigAddr = get_named_compile_time_arg_val("socket_config_addr");
constexpr uint32_t kMaxCores = get_named_compile_time_arg_val("max_cores");
// Static VC for PCIe pushes, spread across fillers by the host.
constexpr uint32_t kWriteVc = get_named_compile_time_arg_val("write_vc");
// Ship threshold, percent of one ring. Binds on the core's fullest LANE, not its span: the
// producer that blocks is always a single lane, and a span-percent under-reads the binding ring.
constexpr uint32_t kShipMinPct = get_named_compile_time_arg_val("ship_min_pct");
// GDDR spool ring in this DRISC's own bank; 0 bytes selects the direct-push path (frames go
// straight from staging to the host FIFO).
constexpr uint32_t kSpoolBase = get_named_compile_time_arg_val("spool_base");
constexpr uint32_t kSpoolBytes = get_named_compile_time_arg_val("spool_bytes");

constexpr uint32_t kNumRisc = 5;
static_assert(kNumRisc == 5, "the control scans are unrolled for exactly five RISCs");
constexpr uint32_t kRingWords = kernel_profiler::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kCtrlWords = kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kSpanWords = kCtrlWords + kNumRisc * kRingWords;
constexpr uint32_t kPrefix = kernel_profiler::SPSC_SPAN_PREFIX_WORDS;
// Slots hold a full span on purpose: every sub-span cap tried deferred whole lanes at speed and
// starved TRISC2's producer.
constexpr uint32_t kSlotWords = kernel_profiler::spsc_span_slot_words(kNumRisc);
constexpr uint32_t kSlotBytes = kSlotWords * 4u;
constexpr uint32_t kWireCtrl = kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPayloadCapWords = kSlotWords - kPrefix - kWireCtrl;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
// Reads take the NoC the writes do not: NOC_INDEX carries egress, the other NoC carries gathers.
constexpr uint8_t kReadNoc = NOC_INDEX == 0 ? 1 : 0;
constexpr bool kSpool = kSpoolBytes != 0;
constexpr uint8_t kDmaShip = 0;   // TX stream 0: staging -> spool
constexpr uint8_t kDmaDrain = 1;  // TX stream 1: spool -> bounce
// Staging layout: two-core batches in kNGens generations, one slot of CV staging, and (spool mode)
// two drain bounce buffers.
constexpr uint32_t kGenSlots = 2;
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kNGens = (kNStage - 1u - kNBounce) / kGenSlots;
static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
constexpr uint32_t kCvBase = kStageBase + kNGens * kGenSlots * kSlotBytes;
constexpr uint32_t kCvReadBytes = 32;
constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
static_assert(kCvReadBytes * kMaxCores <= kSlotBytes, "CV staging must fit its slot");
// The bounces take the rest of the CV slot's space plus their own slots, split in two and
// page-rounded: wide bounces are what pull the sustained drain equilibrium below production.
constexpr uint32_t kBounceBase0 = kCvBase + kCvReadBytes * kMaxCores;
constexpr uint32_t kBounceBytes = (((kNBounce + 1u) * kSlotBytes - kCvReadBytes * kMaxCores) / 2u) & ~(kPageBytes - 1u);
static_assert(kBounceBase0 % kPageBytes == 0, "bounces start on a page");
static_assert(
    !kSpool || kBounceBase0 + kNBounce * kBounceBytes <= kStageBase + kNStage * kSlotBytes,
    "bounces must fit inside the mapped staging arena");
static_assert(!kSpool || kSpoolBytes % kPageBytes == 0, "spool wraps on pages");
constexpr uint32_t kLaneShipWords = (kRingWords * kShipMinPct) / 100u;
// Per-lane ship trigger; kCvBusyPeak is also where idle backoff must stop growing, because a head
// only reaches a producer on a ship -- backing off while lanes fill toward the trigger would blind
// the filler exactly when it is needed.
constexpr uint32_t kLaneTrigger = kRingWords / 2u;
constexpr uint32_t kCvBusyPeak = kLaneTrigger / 2u;
constexpr uint64_t kCyclesPerUs = 1350;  // DRISC wall clock at the 1.35 GHz AICLK
// Idle backoff ceiling. 20 us exceeded a lane's fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 5 * kCyclesPerUs;
// Worst-case host staleness for a workload too light to reach the occupancy bands.
constexpr uint64_t kSpoolFreshCycles = 50'000 * kCyclesPerUs;
constexpr uint64_t kStopDrainCycles = 1'000'000 * kCyclesPerUs;
// How long the exit waits for the host's NIU-restore word before restoring anyway.
constexpr uint64_t kNiuRestoreWaitCycles = 10'000'000 * kCyclesPerUs;

static_assert(kSpanWords * 4u <= NOC_MAX_BURST_SIZE, "a span read must fit one NoC burst");
static_assert(kRingWords * 4u <= NOC_MAX_BURST_SIZE, "a whole-ring gather must fit one NoC burst");
static_assert(kNumRisc <= kernel_profiler::PROFILER_SPSC_MAX_RISC, "control layout too small");
static_assert(kSlotWords % kPageWords == 0, "a slot must be a whole number of socket pages");
// Packed-gather congruence: pads bring each run to its ring phase, and everything else -- slot
// base, payload base, wrap continuations -- must land congruent with no pad. One pad rule serves
// both hops (the gather read into staging and the frame's PCIe write).
static_assert(
    kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_PCIE_WRITE_ALIGNMENT_BYTES &&
        kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u == NOC_L1_READ_ALIGNMENT_BYTES,
    "the shared pad rule no longer matches this part's NoC congruence");
static_assert(
    kRingWords % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        (kPrefix + kWireCtrl) % kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS == 0 &&
        kStageBase % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0 &&
        kSlotBytes % (kernel_profiler::SPSC_SPAN_PACK_ALIGN_WORDS * 4u) == 0,
    "packed-gather congruence broken");

// Control-vector wave: read cores [lo, hi)'s tails into CV staging, then wait until `expect`
// responses have landed since `rd0`. Counted, not barriered: gather responses in flight also bump
// the counter, which can only hand a scan stale-but-valid tails (they are monotonic).
__attribute__((always_inline)) inline void cv_wave(
    const uint64_t* core_noc, uint32_t lo, uint32_t hi, uint32_t rd0, uint32_t expect) {
    for (uint32_t i = lo; i < hi; i++) {
        noc_async_read<kCvReadBytes>(core_noc[i] + kCvReadSrcOff, kCvBase + i * kCvReadBytes, kCvReadBytes, kReadNoc);
    }
    while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED) - rd0 < expect) {
    }
    invalidate_l1_cache();
}

void kernel_main() {
    const uint32_t num_cores = get_arg_val<uint32_t>(0);
    const uint32_t cv_src = get_arg_val<uint32_t>(1);  // profiler_msg_t base on every worker
    volatile tt_l1_ptr uint32_t* coords = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(2));
    // Per-core NoC address of the profiler control block, computed once: get_noc_addr's coordinate
    // arithmetic would otherwise run at every issue site of a sweep that is instruction-stream
    // bound.
    static uint64_t core_noc[kMaxCores];
    for (uint32_t i = 0; i < num_cores; i++) {
        const uint32_t xy = coords[i];
        core_noc[i] = get_noc_addr(xy & 0xFFFFu, xy >> 16, cv_src);
    }
    // Resync the software NoC counter mirrors from hardware. They persist across launches on this
    // never-reset core and firmware only initialises them at boot, so a previous run that ended
    // with unacked writes would wedge this run's first barrier.
    noc_local_state_init(NOC_INDEX);
    noc_local_state_init(kReadNoc);

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = sender.d2h.pcie_xy_enc;
    const uint64_t pcie_base = (static_cast<uint64_t>(sender.d2h.data_addr_hi) << 32) | sender.downstream_fifo_addr;
    set_sender_socket_page_size(sender, kPageBytes);
    // Egress write command state, programmed once: nothing else on this core touches write_cmd_buf
    // on the egress NoC, and re-programming per push cost ~0.5 us a sweep.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(NOC_INDEX, kWriteVc);

    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr);
    *stop = 0;
    volatile tt_l1_ptr uint32_t* acked0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender.bytes_acked_base_addr);
    // Liveness the host can read while the loop runs: without the phase word it cannot tell
    // "exited" from "blocked" from "idle". Every blocking call gets its own phase value.
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 4);
    volatile tt_l1_ptr uint32_t* phase = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr + 8);
    constexpr uint32_t kPhaseInit = 1, kPhasePoll = 2, kPhaseReserve = 3, kPhaseWrite = 4, kPhaseExit = 5;
    constexpr uint32_t kPhWrChunk = 6, kPhWrPush = 7, kPhWrNotify = 8, kPhWrDone = 9;
    constexpr uint32_t kPhDropped = 10, kPhBar1 = 11, kPhSockBar = 14, kPhTailBar = 15;
    *hb = 0;
    *phase = kPhaseInit;

    // Every frame's prefix is identical, and of the control words only heads, tails and the core
    // identity are staged per frame -- the rest must read zero on the wire. Written once here.
    for (uint32_t sl = 0; sl < kNStage; sl++) {
        volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + sl * kSlotBytes);
        pfx[0] = kernel_profiler::spsc_span_w0();
        for (uint32_t k = 1; k < kPrefix + kCtrlWords; k++) {
            pfx[k] = 0;
        }
    }

    // Statics persist across launches on this core, so everything the loop trusts is re-initialised
    // explicitly.
    static uint32_t head_mirror[kMaxCores * kNumRisc];
    // Sum of a core's five tails at its last scan. Tails are monotonic, so the delta is exactly the
    // words produced in one service interval -- the growth term the ship deferral needs.
    static uint32_t tails_seen[kMaxCores];
    static uint8_t hot[kMaxCores];        // shipped real words last scan; hot + empty scan = publish lag
    static uint8_t ship_list[kMaxCores];  // this sweep's ship set, dense core indices
    // Per-slot frame geometry, written at gather issue and consumed a whole batch later by the
    // ship. Stored rather than recomputed so the two phases cannot diverge.
    static uint8_t slot_core[kNStage];
    static uint32_t slot_payload[kNStage];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        hot[i] = 0;
    }
    // Seed the head mirrors from the tails as they stand now: everything published before this
    // launch predates the capture.
    cv_wave(core_noc, 0, num_cores, NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED), num_cores);
    for (uint32_t c = 0; c < num_cores; c++) {
        const tt_l1_ptr uint32_t* tails = reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
        uint32_t tsum = 0;
        for (uint32_t r = 0; r < kNumRisc; r++) {
            head_mirror[c * kNumRisc + r] = tails[r];
            tsum += tails[r];
        }
        tails_seen[c] = tsum;
    }

    uint32_t frames = 0;
    uint32_t sweeps = 0;
    uint32_t gap = 0;
    // Ship-threshold arming. Batching must never hold pre-burst trickle across a burst onset (a
    // pre-loaded ring tips over during the detection latency), and occupancy alone cannot tell
    // one-shot trickle from a light workload's steady sub-threshold lanes. Growth persistence can:
    // defer only after kBatchArmSweeps consecutive growing sweeps, flush after kFlushQuietSweeps
    // dead ones.
    bool grid_busy = false;
    uint32_t grow_streak = 0;
    uint32_t quiet_streak = 0;
    constexpr uint32_t kBatchArmSweeps = 3;
    constexpr uint32_t kFlushQuietSweeps = 8;
    // Which staging generations may still have a ship in flight. Persists across sweeps so a
    // sweep's final ship drains under the pace gap or the next CV pass, not on its own critical
    // path.
    bool gen_shipped[kNGens] = {};

    // GDDR spool state (dead code when kSpool == 0). Byte counters are monotonic 64-bit (long
    // captures exceed 4 GiB); ring offsets are kept incrementally so the hot path never takes a
    // runtime modulo.
    uint64_t spool_wr = 0;          // bytes appended by the ship DMA
    uint64_t spool_done = 0;        // bytes whose DMA writes completed (safe for stream-1 to read)
    uint64_t spool_rd_iss = 0;      // bytes a bounce refill has been issued for
    uint64_t spool_rd = 0;          // bytes whose refill reads completed (safe to overwrite)
    uint32_t spool_wr_off = 0;      // spool_wr % kSpoolBytes
    uint32_t spool_rd_iss_off = 0;  // spool_rd_iss % kSpoolBytes
    uint32_t dma_issued = 0;        // cumulative stream-0 writes, for the per-generation completion gate
    uint32_t gen_dma_mark[kNGens] = {};
    // Bounce buffers: at most one READING and one SHIPPING at a time, so every pump pass is a poll
    // and the drain can never stall the sweep.
    constexpr uint32_t kBounceEmpty = 0, kBounceReading = 1, kBounceReady = 2, kBounceShipping = 3;
    uint32_t b_state[2] = {kBounceEmpty, kBounceEmpty};
    uint32_t b_bytes[2] = {};       // spool bytes held
    uint32_t b_off[2] = {};         // bytes already pushed to the host (partial ships under credit)
    uint32_t b_ack_target[2] = {};  // write-ack mirror at ship: this bounce's flush line
    uint32_t b_seq[2] = {};         // refill order, so a both-ready pass ships the older bytes first
    uint32_t b_rd_mark[2] = {};     // stream-1 issue count at refill: this bounce's completion line
    uint64_t b_rd_end[2] = {};      // what spool_rd advances to when this bounce turns READY
    uint32_t dma_rd_issued = 0;     // cumulative stream-1 reads
    bool notify_pending = false;    // pump ships owe the host a bytes_sent notify (batched per sweep)
    // Pump effort, a pure function of spool occupancy. 0: idle sweeps and the pace gap only.
    // 1: one post-sweep pass every other sweep. 2: post-sweep every sweep. 3: also inline per-batch
    // and in the read-wait spin. Graduated because the bang-bang predecessor (engage 5/8, release
    // 1/16) applied a +2.4 us step to sweeps running a couple of microseconds from ring-full; each
    // graduated step costs well under a microsecond, and at sustained equilibrium the stream flows
    // continuously instead of in ~50 ms sawtooth dumps.
    uint32_t pump_level = 0;
    // The freshness deadline wants at least a per-sweep trickle. Latched: a light workload that
    // never reaches the occupancy bands degrades into a permanent trickle after the first deadline,
    // which is the "reasonably real-time" host stream; cleared when the spool empties.
    bool fresh_boost = false;
    uint64_t spool_oldest = 0;  // when the spool last went non-empty; 0 = empty
    uint32_t fresh_tick = 0;
    bool drain_dead = false;  // teardown escalated (stop=2) with bytes stranded in the spool
    uint32_t credit_timeouts = 0;
    uint32_t drain_chunks = 0;  // also the refill sequence number: b_seq ordering reads it

    // Write `len` bytes at FIFO offset `dst`, splitting a piece that crosses the FIFO wrap --
    // socket_push_pages only wraps the pointer. fifo_size is a whole number of pages, so the split
    // preserves the pack pads' NoC congruence.
    auto push_fifo = [&](uint32_t src, uint32_t dst, uint32_t len) {
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        if (dst >= fifo_size) {
            dst -= fifo_size;
        }
        const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
        write_to_host_chunked(pcie_xy_enc, src, pcie_base + dst, first);
        if (first < len) {
            write_to_host_chunked(pcie_xy_enc, src + first, pcie_base, len - first);
        }
    };

    // The bytes_sent notify. Not socket_notify_receiver: that re-inits write_cmd_buf onto a
    // different VC, and the mesh can then deliver the bytes_sent word ahead of the data it
    // announces. Same command state, VC and route as the data makes delivery order the issue
    // order again.
    auto notify_host = [&]() {
        volatile tt_l1_ptr sender_socket_md* cfg =
            reinterpret_cast<volatile tt_l1_ptr sender_socket_md*>(sender.config_addr);
        cfg->bytes_sent = sender.bytes_sent;
        asm volatile("fence" ::: "memory");
        write_to_host_chunked(
            pcie_xy_enc,
            sender.config_addr,
            (static_cast<uint64_t>(sender.d2h.bytes_sent_addr_hi) << 32) | sender.downstream_bytes_sent_addr,
            4u);
    };

    // Occupancy changes only where spool_wr or spool_rd advance (emit and the pump), so the pump
    // level is maintained at those two sites instead of being recomputed from two u64s per batch.
    auto pump_band = [](uint32_t occ) -> uint32_t {
        return occ >= kSpoolBytes / 2u + kSpoolBytes / 8u   ? 3u
               : occ >= kSpoolBytes / 2u                    ? 2u
               : occ >= kSpoolBytes / 4u + kSpoolBytes / 8u ? 1u
                                                            : 0u;
    };

    // One pump pass: never a spin -- every wait this stage could have is a state a later pass
    // observes, so the pump can delay host delivery but never the sweep.
    auto drain_pump = [&]() -> bool {
        if constexpr (!kSpool) {
            return false;
        } else {
            // L1-only early-out so idle passes cost no DMA or NIU register reads.
            if (spool_wr == spool_rd && b_state[0] == kBounceEmpty && b_state[1] == kBounceEmpty) {
                return false;
            }
            bool did = false;
            // SHIPPING -> EMPTY once the egress writes are acked. Full flush, not "sent": the
            // bounce's next writer is the DMA engine, which a sent-only gate does not fence. Most
            // passes make no progress, so every NIU register poll below is gated on the L1 state
            // that could consume it -- an idle pass costs loads the core already has in hand.
            if (b_state[0] == kBounceShipping || b_state[1] == kBounceShipping) {
                const uint32_t acked = NOC_STATUS_READ_REG(NOC_INDEX, NIU_MST_WR_ACK_RECEIVED);
                for (uint32_t i = 0; i < 2; i++) {
                    if (b_state[i] == kBounceShipping && static_cast<int32_t>(acked - b_ack_target[i]) >= 0) {
                        b_state[i] = kBounceEmpty;
                        did = true;
                    }
                }
            }
            // READING -> READY when a bounce's stream-1 reads retire. Stream completion is FIFO, so
            // one outstanding count gives each bounce its own line and both can fill at once.
            if (b_state[0] == kBounceReading || b_state[1] == kBounceReading) {
                const uint32_t rd_out = experimental::dma_get_reads_outstanding(kDmaDrain);
                for (uint32_t i = 0; i < 2; i++) {
                    if (b_state[i] == kBounceReading && rd_out <= dma_rd_issued - b_rd_mark[i]) {
                        b_state[i] = kBounceReady;
                        if (b_rd_end[i] > spool_rd) {
                            spool_rd = b_rd_end[i];
                        }
                        did = true;
                    }
                }
            }
            // Refill an empty bounce before shipping the ready one, so the read runs under the
            // ship's NoC issue. The second concurrent refill only at full pressure: at a burst the
            // extra in-flight GDDR read deepens the bank queue exactly when the ship DMA needs it.
            const uint32_t emp = b_state[0] == kBounceEmpty ? 0u : (b_state[1] == kBounceEmpty ? 1u : 2u);
            const bool want_refill = emp != 2u && (pump_level >= 3u || b_state[emp ^ 1u] != kBounceReading);
            // Only ship-completed bytes are readable: nothing short of a stream-0 write's
            // completion orders a stream-1 read of the same address behind it. Advanced lazily:
            // polled only when a refill could consume more than the window it already sees.
            if (want_refill && spool_done != spool_wr &&
                static_cast<uint32_t>(spool_done - spool_rd_iss) < kBounceBytes &&
                experimental::dma_get_writes_outstanding(kDmaShip) == 0) {
                spool_done = spool_wr;
            }
            if (want_refill && spool_done != spool_rd_iss) {
                uint32_t len = static_cast<uint32_t>(spool_done - spool_rd_iss);
                if (len > kBounceBytes) {
                    len = kBounceBytes;
                }
                if (len > kSpoolBytes - spool_rd_iss_off) {
                    len = kSpoolBytes - spool_rd_iss_off;
                }
                dma_read_unchecked(kDmaDrain, kSpoolBase + spool_rd_iss_off, kBounceBase0 + emp * kBounceBytes, len);
                spool_rd_iss += len;
                spool_rd_iss_off += len;
                if (spool_rd_iss_off == kSpoolBytes) {
                    spool_rd_iss_off = 0;
                }
                b_rd_mark[emp] = ++dma_rd_issued;
                b_rd_end[emp] = spool_rd_iss;
                b_bytes[emp] = len;
                b_off[emp] = 0;
                b_seq[emp] = drain_chunks;
                b_state[emp] = kBounceReading;
                drain_chunks++;
                did = true;
            }
            // Ship a READY bounce, as much as the host FIFO has credit for right now. Partial ships
            // keep the FIFO fed under credit pressure. Oldest first when both are ready: the socket
            // is a byte stream and the younger bounce would reorder the wire. No per-ship command
            // init and no per-ship notify -- together most of a shipping pass's cost.
            uint32_t rdy = 2u;
            if (b_state[0] == kBounceReady && b_state[1] == kBounceReady) {
                rdy = static_cast<int32_t>(b_seq[0] - b_seq[1]) < 0 ? 0u : 1u;
            } else if (b_state[0] == kBounceReady) {
                rdy = 0u;
            } else if (b_state[1] == kBounceReady) {
                rdy = 1u;
            }
            if (rdy != 2u) {
                invalidate_l1_cache();
                const uint32_t bytes_free = sender.downstream_fifo_total_size - (sender.bytes_sent - *acked0);
                uint32_t nb = b_bytes[rdy] - b_off[rdy];
                if (bytes_free < nb) {
                    nb = bytes_free & ~(kPageBytes - 1u);
                }
                if (nb != 0) {
                    push_fifo(kBounceBase0 + rdy * kBounceBytes + b_off[rdy], sender.write_ptr, nb);
                    socket_push_pages(sender, nb / kPageBytes);
                    notify_pending = true;
                    b_off[rdy] += nb;
                    if (b_off[rdy] == b_bytes[rdy]) {
                        b_state[rdy] = kBounceShipping;
                        b_off[rdy] = 0;
                        // The ack mirror is cumulative, so this also covers earlier partial ships.
                        b_ack_target[rdy] = noc_nonposted_writes_acked[NOC_INDEX];
                    }
                    did = true;
                }
            }
            if (did) {
                pump_level = pump_band(static_cast<uint32_t>(spool_wr - spool_rd));
            }
            return did;
        }
    };

    // Ship `count` adjacent staged slots. A staged slot is already its frame's wire image, so a
    // frame is one write (or one DMA), and the trailing page fill is never written -- the host
    // derives every offset from the control vector and reads past it.
    auto emit_slots = [&](uint32_t start, uint32_t count) {
        if (count == 0) {
            return;
        }
        if constexpr (kSpool) {
            uint32_t bytes = 0;
            for (uint32_t f = 0; f < count; f++) {
                bytes += kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) * 4u;
            }
            // Full spool: pump until there is room. This wait, not a drop, is the spool's
            // back-pressure -- frames stay safe in staging, the sweep slows, producers stall.
            // Interruptible only by the host's stop: lifecycle lives in the close path.
            if (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes) {
                *phase = kPhaseReserve;
                while (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes && *stop == 0) {
                    invalidate_l1_cache();
                    drain_pump();
                }
            }
            if (kSpoolBytes - static_cast<uint32_t>(spool_wr - spool_rd) < bytes) {
                *phase = kPhDropped;
                return;
            }
            // The DMA engine reads the control and length words the scalar core staged; Blackhole
            // stores can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            *phase = kPhaseWrite;
            for (uint32_t f = 0; f < count;) {
                uint32_t fsrc = kStageBase + (start + f) * kSlotBytes;
                // Whole page-rounded frames, dead tail bytes included: the spool offset then
                // advances in lockstep with the FIFO write pointer, so the spool is a byte-exact
                // image of the wire and the drain needs no frame geometry at all.
                uint32_t len = kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) * 4u;
                // A full-span frame fills its slot exactly, so adjacent full frames are
                // wire-contiguous in staging and ship as one DMA write.
                uint32_t nfused = 1;
                while (f + nfused < count && len == nfused * kSlotBytes) {
                    len += kernel_profiler::spsc_span_frame_words(slot_payload[start + f + nfused]) * 4u;
                    nfused++;
                }
                f += nfused;
                while (len != 0) {
                    const uint32_t piece = len > kSpoolBytes - spool_wr_off ? kSpoolBytes - spool_wr_off : len;
                    dma_write_unchecked(kDmaShip, fsrc, kSpoolBase + spool_wr_off, piece);
                    dma_issued++;
                    spool_wr += piece;
                    spool_wr_off += piece;
                    if (spool_wr_off == kSpoolBytes) {
                        spool_wr_off = 0;
                    }
                    fsrc += piece;
                    len -= piece;
                }
            }
            const uint32_t occ = static_cast<uint32_t>(spool_wr - spool_rd);
            pump_level = pump_band(occ);
            *phase = kPhWrDone;
            return;
        }
        // Direct push: reserve host FIFO credit, then write the frames straight to the host.
        uint32_t npages = 0;
        for (uint32_t f = 0; f < count; f++) {
            npages += kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) / kPageWords;
        }
        asm volatile("fence" ::: "memory");
        *phase = kPhaseReserve;
        const bool credited = reserve_pages(sender, npages, stop);
        *phase = kPhaseWrite;
        if (!credited) {
            // Drop rather than block: the heads for these slots were already written back, so the
            // producers stay unblocked and the workload completes. Capture is best-effort; the
            // workload is not.
            *phase = kPhDropped;
            credit_timeouts++;
            return;
        }
        *phase = kPhWrChunk;
        const uint32_t fifo_size = sender.downstream_fifo_curr_size;
        uint32_t wr = sender.write_ptr;
        for (uint32_t f = 0; f < count; f++) {
            const uint32_t payload = slot_payload[start + f];
            push_fifo(kStageBase + (start + f) * kSlotBytes, wr, (kPrefix + payload) * 4u);
            wr += kernel_profiler::spsc_span_frame_words(payload) * 4u;
            if (wr >= fifo_size) {
                wr -= fifo_size;
            }
        }
        *phase = kPhWrPush;
        socket_push_pages(sender, npages);
        *phase = kPhWrNotify;
        notify_host();
        *phase = kPhWrDone;
    };

    // The batched bytes_sent notify for pump ships: once per sweep instead of once per chunk.
    auto drain_notify = [&]() {
        if constexpr (kSpool) {
            if (notify_pending) {
                notify_host();
                notify_pending = false;
            }
        }
    };

    // Main loop. On stop=1, keep sweeping until one whole sweep moves nothing, so markers still in
    // worker rings ship instead of being stranded; exiting on the stop word directly is what
    // silently truncated captures.
    uint64_t stop_seen_at = 0;
    uint32_t frames_at_stop_check = 0;
    for (;;) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_seen_at == 0) {
                stop_seen_at = get_timestamp();
            } else if (frames == frames_at_stop_check || get_timestamp() - stop_seen_at > kStopDrainCycles) {
                break;
            }
            frames_at_stop_check = frames;
        }
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint32_t frames_at_sweep_start = frames;

        uint32_t sweep_peak = 0;
        bool sweep_grew = false;
        // Software pipeline: gather generation G on the read NoC while G^1 ships on the egress
        // side. The CV pass is pipelined into the batch flights: all tail reads issue up front,
        // the wait covers only the first chunk's responses, and the rest of the grid is scanned
        // just in time when the ship list runs low.
        //
        // No lambdas anywhere in the scan region: wrapping it in a by-reference lambda cost
        // 1-2% of sweep time in capture codegen alone, and the saturation boundary amplifies
        // that ~200x. The scan must stay in this exact compilation context.
        uint32_t gen = 0;
        uint32_t pend_n = 0;
        bool have_pend = false;
        uint32_t n_ship = 0;

        const uint32_t cv_chunk = num_cores < kGenSlots * 2u ? num_cores : kGenSlots * 2u;
        const uint32_t rd0 = NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED);
        // Responses can arrive out of order, so a counted response may belong to a
        // later core -- a chunk core then scans last sweep's tails, which are stale but
        // valid (tails are monotonic): it under-ships and catches up next visit.
        // Only the first chunk's CVs are read here; the rest of the grid's are issued
        // at the refill pause, mid-sweep, so the late scan sees tails fresh enough to
        // catch a core that started producing in this very sweep.
        cv_wave(core_noc, 0, cv_chunk, rd0, cv_chunk);
        uint32_t scan_lo = 0;
        uint32_t scan_hi = cv_chunk;
        uint32_t cur = 0;
        for (;;) {
            for (uint32_t c = scan_lo; c < scan_hi; c++) {
                const tt_l1_ptr uint32_t* __restrict tails =
                    reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                uint32_t* __restrict mine = &head_mirror[c * kNumRisc];
                // The scan is unrolled into registers on purpose: a loop over indexed arrays
                // spills on this core, and each spilled word is an L1 round trip per core per
                // sweep.
                const uint32_t d0 = tails[0] - mine[0];
                const uint32_t d1 = tails[1] - mine[1];
                const uint32_t d2 = tails[2] - mine[2];
                const uint32_t d3 = tails[3] - mine[3];
                const uint32_t d4 = tails[4] - mine[4];
                const uint32_t c0 = d0 > kRingWords ? kRingWords : d0;  // overflow is counted at issue
                const uint32_t c1 = d1 > kRingWords ? kRingWords : d1;
                const uint32_t c2 = d2 > kRingWords ? kRingWords : d2;
                const uint32_t c3 = d3 > kRingWords ? kRingWords : d3;
                const uint32_t c4 = d4 > kRingWords ? kRingWords : d4;
                const uint32_t live = c0 + c1 + c2 + c3 + c4;
                uint32_t grew = 0;
                if constexpr (kLaneShipWords != 0) {
                    const uint32_t tsum = tails[0] + tails[1] + tails[2] + tails[3] + tails[4];
                    grew = tsum - tails_seen[c];
                    tails_seen[c] = tsum;
                    sweep_grew |= grew != 0;
                }
                uint32_t peak = c0;
                if (c1 > peak) {
                    peak = c1;
                }
                if (c2 > peak) {
                    peak = c2;
                }
                if (c3 > peak) {
                    peak = c3;
                }
                if (c4 > peak) {
                    peak = c4;
                }
                if (peak > sweep_peak) {
                    sweep_peak = peak;
                }
                if (live == 0) {
                    // A hot core scanning empty is almost always the producer's 64-word batched
                    // tail publish, not idleness. Skipping it would hand the core a two-sweep
                    // service interval, so ship it -- by issue time the in-flight tail refresh
                    // has usually crossed a publish boundary. One-shot: a genuinely idle core
                    // wastes at most one empty frame before going cold.
                    if (hot[c] == 0) {
                        continue;
                    }
                    hot[c] = 0;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                    continue;
                }
                // Deferral must be safe against one more service interval of production, and
                // the level alone cannot promise that: a core scanned just under the threshold
                // at a high rate blows the ring-fill margin two sweeps later. `grew` is the
                // words produced in the last interval, so requiring it under the threshold too
                // bounds a deferred core at ~2x threshold next visit, while trickle cores batch
                // as before.
                if (grid_busy && stop_seen_at == 0 && peak < kLaneShipWords && grew < kLaneShipWords &&
                    peak < kLaneTrigger) {
                    continue;
                }
                hot[c] = 1;
                ship_list[n_ship++] = static_cast<uint8_t>(c);
            }
            scan_lo = scan_hi;

            // Stage one core's frame: write the prefix and control words locally, then
            // gather-read each live run straight to its packed wire offset. The pads bring each
            // destination to its ring phase, so read src == dst (mod 16 B) holds for every
            // piece, including a wrap split, whose continuation is congruent because the ring
            // capacity is a multiple of the alignment.
            auto issue_core = [&](uint32_t c, uint32_t sl) {
                const uint32_t slot = kStageBase + sl * kSlotBytes;
                const uint32_t xy = coords[c];
                const tt_l1_ptr uint32_t* __restrict tails =
                    reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                uint32_t* __restrict mine = &head_mirror[c * kNumRisc];
                volatile tt_l1_ptr uint32_t* __restrict cv =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot + kPrefix * 4u);
                // The head payload and mirror advance are staged here, hidden behind the NIU's
                // acceptance of the same lane's gather read; after the batch barrier only the
                // posted head write remains on the release path. Safe because nothing reads the
                // mirror between issue and that barrier.
                volatile tt_l1_ptr uint32_t* __restrict scp =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kHeadScratch + c * 32u);
                uint32_t live = 0;
                uint32_t off = kPrefix + kWireCtrl;
                ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(kReadNoc, read_cmd_buf, core_noc[c]);
                // The per-lane walk stays a loop, unlike the scan: lane r's bookkeeping hides
                // behind lane r-1's NIU acceptance, and unrolling front-loaded it against every
                // issue and measurably regressed.
                for (uint32_t r = 0; r < kNumRisc; r++) {
                    const uint32_t tail = tails[r];
                    uint32_t run = tail - mine[r];
                    if (run > kRingWords) {
                        run = kRingWords;
                    }
                    const uint32_t start = tail - run;
                    // Frames cap at the slot's payload capacity in whole lanes only: a
                    // published tail is a packet boundary but an arbitrary word count is not,
                    // and clamping mid-run split packets across frames and corrupted the lane
                    // stream.
                    uint32_t take = run;
                    uint32_t pad = 0;
                    const bool img = kernel_profiler::spsc_span_wrap_image(start, take, kRingWords);
                    if (take != 0) {
                        pad = kernel_profiler::spsc_span_pack_pad(img ? 0u : start, off);
                        const uint32_t used = off - (kPrefix + kWireCtrl);
                        const uint32_t room = kPayloadCapWords > used + pad ? kPayloadCapWords - used - pad : 0;
                        // A ring-image ship occupies the whole ring in the slot, not its extent.
                        const uint32_t need = img ? kRingWords : take;
                        if (need > room) {
                            take = 0;
                            pad = 0;
                        }
                    }
                    const uint32_t nh = mine[r] + take;
                    mine[r] = nh;
                    scp[r] = nh;
                    live += take;
                    cv[kernel_profiler::SPSC_WIRE_HEAD_0 + r] = start;
                    cv[kernel_profiler::SPSC_WIRE_TAIL_0 + r] = start + take;
                    if (take == 0) {
                        continue;
                    }
                    off += pad;
                    const uint32_t ring_src = cv_src + (kCtrlWords + r * kRingWords) * 4u;
                    const uint32_t hm = start & (kRingWords - 1u);
                    if (img) {
                        // A near-full wrapping run ships as its whole ring image in one read;
                        // the decoder linearises by head using the same shared predicate. This
                        // is nearly every lane at the saturation boundary, where the wrap
                        // split's second issue is pure loss. One ring per read is also the
                        // measured optimum: coalescing adjacent whole-ring lanes into bigger
                        // reads starves the producer's own L1 port (a clean dose-response, up
                        // to ~70x the stall floor at five rings per read).
                        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                            kReadNoc, read_cmd_buf, ring_src, slot + off * 4u, kRingWords * 4u);
                        off += kRingWords;
                    } else if (hm + take > kRingWords) {
                        // A small wrapping run ships as the two-piece split, byte-exact: at
                        // sustained rates the image's dead remainder is most of the ring, and
                        // there the drain, not the sweep, is the binding resource.
                        const uint32_t first = kRingWords - hm;
                        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                            kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, first * 4u);
                        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                            kReadNoc, read_cmd_buf, ring_src, slot + (off + first) * 4u, (take - first) * 4u);
                        off += take;
                    } else {
                        ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                            kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                        off += take;
                    }
                }
                cv[kernel_profiler::SPSC_WIRE_XY] = xy;
                // pfx[0] is constant and staged once at init; only the payload word varies.
                volatile tt_l1_ptr uint32_t* pfx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot);
                pfx[1] = off - kPrefix;
                slot_payload[sl] = off - kPrefix;
                slot_core[sl] = static_cast<uint8_t>(c);
            };

            // Heads go out the moment the batch's read barrier passes, not with the frame
            // emit: the payload is resident in staging once the reads land, so the producer's
            // ring slots are free regardless of when the frame reaches the host.
            auto advance_heads = [&](uint32_t n, uint32_t g) {
                for (uint32_t i = 0; i < n; i++) {
                    const uint32_t sl = g * kGenSlots + i;
                    const uint32_t c = slot_core[sl];
                    const uint32_t sc = kHeadScratch + c * 32u;
                    // Posted (the barriers protect staging reuse, which a head write never
                    // touches; scratch reuse is safe on the slot rotation) and on the read NoC:
                    // on the egress NoC this small packet queues behind frame data, so head
                    // visibility inherited the PCIe tile's acceptance jitter.
                    noc_async_write_one_packet<true, true>(
                        sc, core_noc[c] + kernel_profiler::SPSC_RING_HEAD_0 * 4u, kNumRisc * 4u, kReadNoc);
                    frames++;
                }
            };

            auto ship_frames = [&](uint32_t n, uint32_t g) {
                emit_slots(g * kGenSlots, n);
                if constexpr (kSpool) {
                    gen_dma_mark[g] = dma_issued;
                }
                gen_shipped[g] = true;
            };

            while (cur < n_ship) {
                // Refill the ship list before it runs dry: the per-batch tail refresh below
                // only covers cores already on the list, so a batch issued right after a
                // dry-out scan would consume tails a whole sweep stale.
                if (scan_hi < num_cores && n_ship - cur <= kGenSlots) {
                    break;
                }

                // This generation's previous ship must be out of staging before its slots
                // refill. gen_shipped persists across sweeps, so a sweep's last ship is never
                // waited on inside its own sweep -- this is the wait that catches it if the
                // pace gap has not already drained it.
                if (gen_shipped[gen]) {
                    *phase = kPhBar1;
                    // Bare waits: both predicates complete on this device alone (the DMA
                    // engine's writes to GDDR, the NIU's sent counter), so no consumer
                    // state can hang them. Host-gated waits keep their bounds.
                    if constexpr (kSpool) {
                        // This generation's ship writes only: stream completion is FIFO,
                        // so outstanding <= later-issues means this generation retired.
                        const uint32_t since = dma_issued - gen_dma_mark[gen];
                        const uint32_t cap = since > 15u ? 15u : since;
                        while (experimental::dma_get_writes_outstanding(kDmaShip) > cap) {
                        }
                    } else {
                        // Sent-only is legal here because the staging slots' next writer is
                        // this core's own NIU read responses.
                        while (!ncrisc_noc_nonposted_writes_sent(NOC_INDEX)) {
                        }
                    }
                    gen_shipped[gen] = false;
                }

                uint32_t n = 0;
                uint32_t slots = 0;
                while (slots < kGenSlots && cur < n_ship) {
                    issue_core(ship_list[cur], gen * kGenSlots + slots);
                    cur++;
                    n++;
                    slots++;
                }
                // Refresh the next batch's tails in the same flight: on the sweep-start
                // snapshot alone the last cores would be served a sweep stale, and the
                // scan-order-last core took all the stalls. This generation's read barrier
                // covers these reads too.
                const uint32_t nn = (n_ship - cur) < kGenSlots ? (n_ship - cur) : kGenSlots;
                for (uint32_t i = 0; i < nn; i++) {
                    const uint32_t c = ship_list[cur + i];
                    ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(kReadNoc, read_cmd_buf, core_noc[c]);
                    ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                        kReadNoc, read_cmd_buf, cv_src + kCvReadSrcOff, kCvBase + c * kCvReadBytes, kCvReadBytes);
                }

                // The overlap: the previous batch ships on the egress side while this batch's
                // gather reads fly on the read NoC.
                if (have_pend) {
                    ship_frames(pend_n, gen == 0u ? kNGens - 1u : gen - 1u);
                }
                if constexpr (kSpool) {
                    if (pump_level >= 3u) {
                        drain_pump();
                    }
                }

                // Read barrier before the heads go out. The spin doubles as the pump's slot
                // -- cycles the core burns anyway -- but only at full pressure: below it
                // the pump's GDDR reads contend with the ship DMA and the landing gathers.
                while (!ncrisc_noc_reads_flushed(kReadNoc)) {
                    if constexpr (kSpool) {
                        // Level 3 means occupancy is over the 5/8 line, so nonempty holds.
                        if (pump_level >= 3u) {
                            drain_pump();
                        }
                    }
                }
                invalidate_l1_cache();
                advance_heads(n, gen);

                pend_n = n;
                have_pend = true;
                gen = gen + 1u == kNGens ? 0u : gen + 1u;
            }
            if (cur >= n_ship && scan_hi >= num_cores) {
                if (have_pend) {
                    ship_frames(pend_n, gen == 0u ? kNGens - 1u : gen - 1u);
                    have_pend = false;
                }
                break;
            }
            // The rest of the grid's CV reads issue HERE, mid-sweep, not at sweep start:
            // the late scan runs mid-sweep either way, and sweep-start data maximizes the
            // staleness of exactly the cores scanned last. Read now, their tails can show a
            // core that started producing during this sweep, cutting the join blind window by
            // most of a sweep. Same reads, later timing.
            cv_wave(core_noc, scan_hi, num_cores, rd0, num_cores);
            scan_hi = num_cores;
        }

        // Busy sweeps below the first band skip the post-sweep pump entirely: the spool is the
        // burst absorber, and a capture that fits in it deserves pure gather.
        if constexpr (kSpool) {
            if (pump_level >= 2u || (pump_level == 1u && (sweeps & 1u) != 0) || fresh_boost ||
                frames == frames_at_sweep_start) {
                drain_pump();
                drain_notify();
            }
        }

        // Ship-threshold arming (see the declarations above for why growth persistence, not level).
        if constexpr (kLaneShipWords != 0) {
            if (sweep_grew) {
                grow_streak++;
                quiet_streak = 0;
            } else if (++quiet_streak >= kFlushQuietSweeps) {
                grow_streak = 0;
            }
            grid_busy = grow_streak >= kBatchArmSweeps;
        }
        // Freshness deadline, checked every 64 sweeps because even one wall-clock read per sweep
        // measurably stalls producers at the saturation boundary (~1 ms of stride is noise against
        // the 50 ms bound). A workload too light to reach the occupancy bands would otherwise sit
        // in the spool for seconds; the deadline escalates it to the latched per-sweep trickle,
        // never to inline pumping -- freshness is a latency bound, not a pressure emergency.
        if constexpr (kSpool) {
            if (++fresh_tick >= 64u) {
                fresh_tick = 0;
                if (spool_wr == spool_rd) {
                    spool_oldest = 0;
                    fresh_boost = false;
                } else {
                    const uint64_t now = get_timestamp();
                    if (spool_oldest == 0 || pump_level >= 1u || fresh_boost) {
                        spool_oldest = now;
                        fresh_boost = pump_level == 0u && fresh_boost;
                    } else if (now - spool_oldest > kSpoolFreshCycles) {
                        fresh_boost = true;
                    }
                }
            }
        }
        // Idle pacing: collapse on work, creep toward the ceiling when idle. Live-but-untriggered
        // lanes count as work here -- a head only reaches a producer on a ship, so sleeping while
        // lanes fill toward the trigger is exactly wrong.
        if (frames != frames_at_sweep_start || sweep_peak >= kCvBusyPeak) {
            gap = 0;
        } else {
            uint32_t inc = gap >> 1;
            if (inc < 256u) {
                inc = 256u;
            }
            gap = (gap + inc > kCvIdleGapMax) ? kCvIdleGapMax : gap + inc;
        }
        if (gap != 0) {
            const uint64_t until = get_timestamp() + gap;
            while (get_timestamp() < until) {
                if constexpr (kSpool) {
                    drain_pump();  // idle time is drain time
                }
            }
        }
    }

    // Exit. Everything the run spooled must reach the host FIFO before the socket barrier can
    // pass; bounded, so a consumer that stopped acking strands bytes (counted) instead of wedging
    // teardown.
    if constexpr (kSpool) {
        while (experimental::dma_get_writes_outstanding(kDmaShip) != 0) {
        }
        while (spool_rd_iss != spool_wr || b_state[0] != kBounceEmpty || b_state[1] != kBounceEmpty) {
            drain_pump();
            // Notify per pass, not per sweep: with a host FIFO smaller than the backlog, the
            // acks that free credit only come after the host has seen the bytes.
            drain_notify();
            // The host's teardown escalates stop to 2 after its own timeout -- the close path's
            // kill switch for a drain whose consumer will never finish it.
            invalidate_l1_cache();
            if (*stop == 2u) {
                drain_dead = true;
                break;
            }
        }
        drain_notify();
    }

    // socket_barrier waits for the host to ack everything, so it would hang on a dead consumer.
    const bool consumer_gone = credit_timeouts != 0 || drain_dead;
    *phase = kPhSockBar;
    if (!consumer_gone) {
        socket_barrier(sender);
    }
    *phase = kPhTailBar;
    while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {
    }
    // The posted head write-backs are outside that barrier's predicate; drain their sent counter
    // (small packets stream out in nanoseconds) so no unstreamed head is left behind.
    const uint64_t t_ps = get_timestamp() + 1000 * kCyclesPerUs;
    while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
           get_timestamp() < t_ps) {
    }
    *phase = kPhaseExit;
    // Written back only for a live consumer: after dropped frames the socket's view of bytes_sent
    // is already out of sync with the host's, and the socket is being torn down either way.
    if (!consumer_gone) {
        update_socket_config(sender);
    }

    // Published last, after the socket barrier, so the host only sees `done` once every page is
    // out.
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDoneAddr);
    *done = 0xD09E0000u | (frames & 0xFFFFu);

    // NIU restore, on the host's word. NIU_CFG_0 persists until chip reset, so whoever set stream
    // mode owns putting it back -- and last, because the flip to NOC2AXI takes this L1 (`done`,
    // the results, bytes_acked) out of the host's view.
    for (const uint64_t t_end = get_timestamp() + kNiuRestoreWaitCycles; *stop != 2u && get_timestamp() < t_end;) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
