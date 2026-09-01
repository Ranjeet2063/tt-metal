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
// File layout: compile-time arguments and derived constants first, then kernel_main, which is the
// pipeline itself. All diagnostic-tier machinery (state, helpers) lives in `==== instrumentation`
// banner blocks, whose lines end in a bare `//`; everything in them compiles out in record builds.
// In the pipeline proper, instrumentation appears only as bare I_*() macro lines -- skip those;
// the macro table right below the constants says what each expands to.
//
// Wire format and placement history: tools/drisc_drain/FINDINGS.md.

#include "drisc_drain_common.hpp"

// ---- kernel arguments and derived constants --------------------------------------------------------

constexpr uint32_t kStageBase = get_compile_time_arg_val(0);
constexpr uint32_t kNStage = get_compile_time_arg_val(1);
constexpr uint32_t kHeadScratch = get_compile_time_arg_val(2);
constexpr uint32_t kResultsAddr = get_compile_time_arg_val(3);
constexpr uint32_t kDoneAddr = get_compile_time_arg_val(4);
constexpr uint32_t kStopAddr = get_compile_time_arg_val(5);  // host writes 1 = quiesce, 2 = free the NIU
constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(6);
constexpr uint32_t kMaxSweeps = get_compile_time_arg_val(7);
constexpr uint32_t kMaxCores = get_compile_time_arg_val(8);
constexpr uint32_t kGapCycles = get_compile_time_arg_val(9);  // idle pacing seed; 0 = continuous
constexpr uint32_t kNocInit = get_compile_time_arg_val(11);   // 0 only to reproduce the stale-mirror wedge
constexpr uint32_t kPcieEncOverride = get_compile_time_arg_val(16);
constexpr uint32_t kWriteVc = get_compile_time_arg_val(20);  // static VC for PCIe pushes, spread by the host
// Ship threshold, percent of one ring. Binds on the core's fullest LANE, not its span: the
// producer that blocks is always a single lane, and a span-percent under-reads the binding ring.
constexpr uint32_t kShipMinPct = get_compile_time_arg_val(39);
// GDDR spool ring in this DRISC's own bank; 0 bytes selects the direct-push path (frames go
// straight from staging to the host FIFO).
constexpr uint32_t kSpoolBase = get_compile_time_arg_val(42);
constexpr uint32_t kSpoolBytes = get_compile_time_arg_val(43);
// Diagnostic tiers, all off in shipped builds. kInstr: phase cycle counters (~55 clock reads a
// sweep). kSvcInstr: per-core service-interval histogram and per-phase worst cases. kSelfZones:
// the drainer's own device zones, framed like a worker span and shipped down the path it already
// owns. kNocFootprint: this core's NIU counters per sweep. kSyncEvent: a host-triggered common
// fiducial that rides the self-zone ring.
constexpr uint32_t kSelfZones = get_compile_time_arg_val(32);
constexpr uint32_t kSelfHoldCycles = get_compile_time_arg_val(33);
constexpr uint32_t kSelfXY = get_compile_time_arg_val(34);  // this DRISC's own virtual (y<<16)|x
constexpr uint32_t kSelfMaxFrames = get_compile_time_arg_val(35);
constexpr uint32_t kSelfDetail = get_compile_time_arg_val(36);  // 0 = sweep+pace zones, 1 = per-batch phases too
constexpr uint32_t kNocFootprint = get_compile_time_arg_val(37);
constexpr uint32_t kSyncEvent = get_compile_time_arg_val(38);
constexpr uint32_t kSvcInstr = get_compile_time_arg_val(40);
constexpr uint32_t kInstr = get_compile_time_arg_val(41);
constexpr bool kSelfPhases = kSelfZones != 0 && kSelfDetail != 0;
// Args 10, 12..15, 17..18 and 21..31 are retired but their positions stay occupied: argument
// indices appear in JIT cache keys and in FINDINGS notes.

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
// Staging layout: two-core batches in kNGens generations, CV staging, and (spool mode) two drain
// bounce buffers. Self-zone builds park the CV staging inside the self frame's slot, in the ring
// 1..4 area -- safe because only the self frame's ring 0 is ever live and the host walks a lane
// only between its head and tail, so those bytes ship but are never decoded.
constexpr uint32_t kGenSlots = 2;
constexpr uint32_t kNBounce = kSpool ? 2u : 0u;
constexpr uint32_t kCvOwnSlot = kSelfZones != 0 ? 0u : 1u;
constexpr uint32_t kNGens = (kNStage - kCvOwnSlot - kNBounce) / kGenSlots;
static_assert(kNGens >= 2, "the ship pipeline needs at least two staging generations");
constexpr uint32_t kBounceSlot0 = kNGens * kGenSlots + kCvOwnSlot;
static_assert(kBounceSlot0 + kNBounce <= kNStage, "bounce slots must fit inside the staging arena");
// The self frame lives in slot kNStage, one past every slot the drain pipeline can touch (the
// host reserves it by passing nstage - 1 when self-zones are on).
constexpr uint32_t kSelfSlot = kNStage;
constexpr uint32_t kCvSlot = kSelfZones != 0 ? kNStage : kNGens * kGenSlots;
constexpr uint32_t kCvBase =
    kStageBase + kCvSlot * kSlotBytes + (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u);
constexpr uint32_t kCvReadBytes = 32;
constexpr uint32_t kCvReadSrcOff = kernel_profiler::SPSC_RING_TAIL_0 * 4u;
static_assert(
    (kSelfZones != 0 ? (kPrefix + kCtrlWords + kRingWords) * 4u : 0u) + kCvReadBytes * kMaxCores <= kSlotBytes,
    "CV staging must fit its slot");
// The bounces take the rest of the CV slot's space plus their own slots, split in two and
// page-rounded: wide bounces are what pull the sustained drain equilibrium below production.
constexpr uint32_t kBounceBase0 =
    kSelfZones != 0 ? kStageBase + kBounceSlot0 * kSlotBytes : kCvBase + kCvReadBytes * kMaxCores;
constexpr uint32_t kBounceBytes =
    kSelfZones != 0 ? kSlotBytes
                    : (((kNBounce + 1u) * kSlotBytes - kCvReadBytes * kMaxCores) / 2u) & ~(kPageBytes - 1u);
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
// Idle backoff ceiling, ~5 us. 20 us exceeded a lane's fill time at high rates.
constexpr uint32_t kCvIdleGapMax = 6750;
// ~50 ms: the worst-case host staleness for a workload too light to reach the occupancy bands.
constexpr uint64_t kSpoolFreshCycles = 67500000ull;
constexpr uint64_t kStopDrainCycles = 1350000000;

static_assert(kSelfZones == 0 || kSelfHoldCycles >= 1, "a 0-cycle window hold would trace nothing");
static_assert(kSelfDetail <= 1, "detail is 0 (sweep+pace) or 1 (full per-batch phases)");
static_assert(kSelfZones == 0 || kSelfMaxFrames >= 1, "self-profiling with a 0 frame budget captures nothing");
static_assert(kSyncEvent == 0 || kSelfZones != 0, "the sync event rides the self-zone ring; enable zones");
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

// ==== instrumentation: pipeline hooks ===============================================================
// Inside kernel_main, every diagnostic statement is one bare I_*() macro from this table, so the
// pipeline reads as functional code plus skippable tokens. The macros expand to the helper
// lambdas defined in the banner blocks inside kernel_main and bind pipeline locals textually;
// every one of them compiles to nothing in record builds.
#define I_NOC_IDX_BEFORE() const uint32_t noc_index_before = noc_index                                          //
#define I_NOC_IDX_AFTER() const uint32_t noc_index_after = noc_index                                            //
#define I_PUMP_BEGIN() const uint64_t t_dp0 = kSvcInstr != 0 ? instr_now() : 0                                  //
#define I_PUMP_END() svc_note_max(did, max_pump, max_pump_at, t_dp0)                                            //
#define I_STARVED() drain_starved++                                                                             //
#define I_SHIPPED() drain_ships++                                                                               //
#define I_SPOOL_BEGIN() const uint64_t t0 = kInstr != 0 ? instr_now() : 0; ZoneWrite z_write(self_mark_phase)   //
#define I_SPOOL_END() instr_add64(c_write, t0)                                                                 //
#define I_WR_BEGIN() const uint64_t t0 = kInstr != 0 ? instr_now() : 0                                          //
#define I_ZONE_CREDIT() ZoneCreditWait z_credit(self_mark_phase)                                                //
#define I_WR_DROP() instr_add64(c_reserve, t0)                                                                  //
#define I_WR_CREDITED() const uint64_t t1 = instr_reserve_end(t0); ZoneWrite z_write(self_mark_phase)           //
#define I_WR_CHUNKED() const uint64_t t2 = instr_lap(c_wr_chunk, t1)                                            //
#define I_WR_PUSHED() const uint64_t t3 = instr_lap(c_wr_push, t2)                                              //
#define I_WR_DONE() instr_wr_end(t1, t3)                                                                        //
#define I_STOP_WORDS() words_at_stop = total_words                                                              //
#define I_STOP_SWEEP() stop_sweeps++                                                                            //
#define I_SWEEP_BEGIN() const uint64_t t_sweep0 = instr_sweep_begin()                                           //
#define I_ZONE_SWEEP() ZoneSweep z_sweep(self_mark_now)                                                         //
#define I_CV_BEGIN() const uint64_t t_cv0 = kInstr != 0 ? instr_now() : 0                                       //
#define I_CV_ZONE() ZoneRead z_cv(self_mark_phase); const uint64_t t_cvw0 = kSvcInstr != 0 ? instr_now() : 0    //
#define I_CV_ISSUED() instr_add32(n_cv_rd, cv_chunk)                                                            //
#define I_LATE_CVS() instr_add32(n_cv_rd, num_cores - scan_hi)                                                  //
#define I_CV_WAITED() svc_note_max(true, max_cvw, max_cvw_at, t_cvw0)                                           //
#define I_CV_END() instr_cv_end(t_cv0)                                                                          //
#define I_SCAN_BEGIN() const uint64_t t_scan0 = kInstr != 0 ? instr_now() : 0                                   //
#define I_SCAN_PEAK() instr_max32(max_occ, peak)                                                                //
#define I_SCAN_DEFER() instr_add32(ship_deferred, 1u)                                                           //
#define I_SCAN_END() instr_add64(c_scan, t_scan0)                                                               //
#define I_GATHER_READ() instr_add32(n_gather_rd, 1u)                                                            //
#define I_PROC_BEGIN() const uint64_t t_p0 = kInstr != 0 ? instr_now() : 0; ZoneProc z_proc(self_mark_phase)    //
#define I_HEAD_BEGIN() const uint64_t t_h0 = svc_head_begin(c)                                                  //
#define I_HEAD_END() svc_head_end(t_h0)                                                                         //
#define I_PROC_END() instr_add64(c_proc, t_p0)                                                                  //
#define I_SELF_ARM() self_arm_if(n != 0)                                                                        //
#define I_BAR_BEGIN() const uint64_t t_b0 = (kInstr != 0 || kSvcInstr != 0) ? instr_now() : 0                   //
#define I_ZONE_BAR() ZoneWrBarrier z_bar(self_mark_phase)                                                       //
#define I_BAR_END() instr_add64(c_barrier, t_b0); svc_note_max(true, max_bar, max_bar_at, t_b0)                 //
#define I_BATCH_BEGIN() const uint64_t t_batch0 = kInstr != 0 ? instr_now() : 0                                 //
#define I_ZONE_ISSUE() ZoneRead z_issue(self_mark_phase)                                                        //
#define I_BATCH_CVS() instr_add32(n_cv_rd, nn)                                                                  //
#define I_CVAGE_WAVE() cvage_stamp(t_cv_cur)                                                                    //
#define I_CVAGE_PREF() cvage_stamp(t_cv_next)                                                                   //
#define I_CVAGE_RELIEF() cvage_relief()                                                                         //
#define I_ISSUE_MARK() const uint64_t t_issue = kInstr != 0 ? instr_now() : 0                                   //
#define I_PROC_MARK() const uint64_t t_after_proc = kInstr != 0 ? instr_now() : 0                               //
#define I_WAIT_BEGIN() const uint64_t t_rw0 = kSvcInstr != 0 ? instr_now() : 0                                  //
#define I_ZONE_WAIT() ZoneReadWait z_wait(self_mark_phase)                                                      //
#define I_WAIT_END() svc_note_max(true, max_rdw, max_rdw_at, t_rw0)                                             //
#define I_BATCH_END() instr_batch_end(t_batch0, t_issue, t_after_proc)                                          //
#define I_SWEEP_LAP() sweep_cyc = instr_sweep_lap(t_sweep0)                                                     //
#define I_SWEEP_END() const bool win2_work = instr_sweep_end(t_sweep0, sweep_cyc, frames_at_sweep_start, sweep_peak)  //
#define I_DRAIN_BEGIN() const uint64_t t_d0 = kInstr != 0 ? instr_now() : 0; ZoneDrain z_drain(self_mark_now)   //
#define I_DRAIN_END() instr_add64(c_drain, t_d0)                                                                //
#define I_ZONE_PACE() ZonePace z_pace(self_mark_now)                                                            //
#define I_PACE_END() c_pace += get_timestamp() - t_g0                                                           //
#define I_WIN_LAST() instr_win_last(win2_work)                                                                  //
// Gated: taking &emit_slots escapes its closure, which pessimizes alias analysis for every
// captured local in record builds too. The hook must not exist unless the self-zone tier does.
#define I_WIRE_SELF_EMIT()                                                                       \
    if constexpr (kSelfZones != 0) {                                                             \
        self_emit.ctx = static_cast<void*>(&emit_slots);                                         \
        self_emit.fn = [](void* p, uint32_t sl, uint32_t n) {                                    \
            (*static_cast<decltype(emit_slots)*>(p))(sl, n);                                     \
        };                                                                                       \
    }                                                                                            \
    static_assert(true)
#define I_SYNC_EVENT() self_sync_event()                                                                        //
#define I_SELF_SWEEP_ARM() self_sweep_arm(t_sweep0)                                                             //
#define I_SELF_ACCOUNT() self_sweep_account(t_sweep0, sweep_cyc, frames != frames_at_sweep_start)               //
#define I_SELF_FLUSH_SWEEP() self_flush_sweep()                                                                 //
#define I_SELF_FLUSH_FINAL() self_flush_final()                                                                 //
#define I_RESULTS() instr_write_results(t_start, t_end)                                                         //
// ==== end instrumentation ===========================================================================

// Every instrumentation clock read goes through this one non-inlined body: get_timestamp's
// volatile locals cost ~30 bytes of code per inlined site, and the diagnostic builds sit within a
// few hundred bytes of the DRISC code-region limit. Functional deadlines keep the inline read.
__attribute__((noinline)) static uint64_t instr_now() { return get_timestamp(); }  //

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
    if constexpr (kNocInit) {
        noc_local_state_init(NOC_INDEX);
        noc_local_state_init(kReadNoc);
    }
    I_NOC_IDX_BEFORE();
    I_NOC_IDX_AFTER();

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    const uint32_t pcie_xy_enc = kPcieEncOverride != 0 ? kPcieEncOverride : sender.d2h.pcie_xy_enc;
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
    static uint8_t seeded[kMaxCores];
    static uint8_t hot[kMaxCores];        // shipped real words last scan; hot + empty scan = publish lag
    static uint8_t ship_list[kMaxCores];  // this sweep's ship set, dense core indices
    // Per-slot frame geometry, written at gather issue and consumed a whole batch later by the
    // ship. Stored rather than recomputed so the two phases cannot diverge. slot_payload[kSelfSlot]
    // is the self frame's constant payload.
    static uint8_t slot_core[kNStage];
    static uint32_t slot_payload[kNStage + 1];
    for (uint32_t i = 0; i < kMaxCores; i++) {
        seeded[i] = 0;
        hot[i] = 0;
    }

    uint64_t total_words = 0;
    uint32_t pages = 0;
    uint32_t frames = 0;
    uint32_t pushes = 0;
    uint32_t sweeps = 0;
    uint32_t overflows = 0;
    uint32_t gap = kGapCycles;  // idle pacing; seeded at 0 so the first sweep runs immediately
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
    uint32_t spool_drops = 0;
    uint32_t credit_timeouts = 0;
    uint32_t dropped_frames = 0;
    uint32_t drain_chunks = 0;  // also the refill sequence number: b_seq ordering reads it

    // ==== instrumentation: counters, histograms and window state (kInstr / kSvcInstr) ============
    uint32_t spool_max = 0;      //
    uint32_t drain_ships = 0;    //
    uint32_t drain_starved = 0;  //
    uint64_t words_at_stop = 0;  //
    uint32_t stop_sweeps = 0;    //
    uint32_t sweep_cyc = 0;      // this sweep's body cycles, written at the sweep-end lap //
    uint64_t s_read0 = 0, s_proc0 = 0, s_rsv0 = 0, s_wr0 = 0, s_bar0 = 0;  // per-sweep phase baselines //
    uint64_t c_scan = 0, c_read = 0, c_issue = 0, c_cv = 0;                //
    uint64_t c_proc = 0, c_reserve = 0, c_write = 0;                       //
    uint64_t c_barrier = 0, c_ph_head = 0;                                 //
    uint64_t c_wr_chunk = 0, c_wr_push = 0, c_wr_notify = 0;               //
    uint64_t c_idle = 0, c_busy = 0, c_pace = 0, c_drain = 0;              //
    uint32_t n_gather_rd = 0, n_cv_rd = 0;                                 //
    uint32_t ship_deferred = 0, max_occ = 0;                               //
    uint64_t cvage_sum = 0, t_cv_cur = 0, t_cv_next = 0;                   //
    uint32_t cvage_n = 0, cvage_max = 0;                                   //
    uint32_t sweeps_idle = 0, max_sweep = 0, max_reserve = 0;              //
    uint32_t ws_read = 0, ws_proc = 0, ws_rsv = 0, ws_wr = 0, ws_bar = 0;  // worst sweep's phase split //
    uint32_t fill_hist[8] = {};                                            //
    uint32_t svc_max = 0;                                                  //
    static uint32_t svc_hist[8];                                           //
    static uint32_t last_ship[kMaxCores];                                  //
    for (uint32_t i = 0; i < 8; i++) {                                     //
        svc_hist[i] = 0;                                                   //
    }                                                                      //
    for (uint32_t i = 0; i < kMaxCores; i++) {                             //
        last_ship[i] = 0;                                                  //
    }                                                                      //
    uint32_t max_bar = 0, max_bar_at = 0;                                  //
    uint32_t max_rdw = 0, max_rdw_at = 0;                                  //
    uint32_t max_cvw = 0, max_cvw_at = 0;                                  //
    uint32_t max_pump = 0, max_pump_at = 0;                                //
    // Workload-window snapshots: lifetime percentages are dominated by idle residence, so the
    // report diffs between the first shipping sweep and the last.
    bool win2_open = false;                                                                     //
    uint64_t w0_t = 0, w1_t = 0, w0_busy = 0, w1_busy = 0, w0_idle = 0, w1_idle = 0;            //
    uint64_t w0_pace = 0, w1_pace = 0, w0_cv = 0, w1_cv = 0, w0_issue = 0, w1_issue = 0;        //
    uint32_t w0_frames = 0, w1_frames = 0, w0_sweeps = 0, w1_sweeps = 0;                        //
    // ==== end instrumentation ===================================================================

    // ==== instrumentation: DRISC self-profiling state and marker path (kSelfZones) ===============
    // The drainer's own zones: 2-word markers plus sticky timers into a 512-word ring in the self
    // slot, published as ordinary RAW span frames through the egress the drainer already owns.
    // Same wall clock as the workers, so nothing needs calibrating.
    NocFpState nf{};                     //
    uint32_t self_head = 0;              // words the host has been shown //
    uint32_t self_tail = 0;              // words written //
    uint32_t self_hi = 0xFFFFFFFFu;      // last wall-clock high half; ~0 forces a first sticky //
    uint32_t self_frames = 0;            //
    uint32_t self_markers = 0;           //
    uint32_t self_dropped = 0;           // markers refused because a publish could not free the ring //
    uint32_t self_sweeps = 0;            //
    uint32_t self_sweeps_work = 0;       //
    uint32_t self_windows = 0;           //
    uint32_t self_words_shipped = 0;     // must end equal to self_tail or trace was lost at teardown //
    uint32_t self_over = 0;              // sweeps uninstrumented because the frame budget was spent //
    uint64_t c_self = 0;                 // cycles spent publishing self frames (the perturbation) //
    uint64_t self_t_sweep0 = 0;          //
    uint64_t self_armed_until = 0;       // instrument every sweep that starts before this //
    bool self_on = false;                //
    bool self_busy = false;              // inside self_publish: suppress markers (re-entrancy) //
    bool self_work = false;              //
    bool self_from_start = false;        // instrumented from the top of the sweep, not armed mid-way //
    // Phase totals over from-the-start sweeps only, so summed zone durations have an exact check.
    uint32_t self_ck_sweeps = 0;         //
    uint64_t self_ck_read = 0, self_ck_proc = 0, self_ck_rsv = 0, self_ck_write = 0, self_ck_bar = 0;  //
    uint32_t sync_seen = 0;              //
    uint32_t sync_events = 0;            //
    uint32_t sync_timeouts = 0;          // must stay 0: a timed-out drainer voids the fiducial //
    uint32_t sync_spin_cyc = 0;          //
    // Sync-event rendezvous words, in the pad behind `stop`: +4 the host's request, +8 our
    // "parked" ack, +12 the release. The ack is what makes it a barrier.
    volatile tt_l1_ptr uint32_t* sync_req = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 4);   //
    volatile tt_l1_ptr uint32_t* sync_ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 8);   //
    volatile tt_l1_ptr uint32_t* sync_go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStopAddr + 12);   //
    volatile tt_l1_ptr uint32_t* self_ring = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(  //
        kStageBase + kSelfSlot * kSlotBytes + (kPrefix + kCtrlWords) * 4u);                   //
    // Publish trampoline: the marker path's ring-full branch must call self_publish, which is
    // defined below it -- a reference cycle no declaration order can express. Wired right after
    // self_publish is defined.
    struct SelfPub {                     //
        void* ctx = nullptr;             //
        void (*fn)(void*) = nullptr;     //
    } self_pub;                          //
    // Same trick for the self frame's egress: self_publish ships through emit_slots, which is
    // functional code defined after this whole region -- the pipeline wires this hook with
    // I_WIRE_SELF_EMIT() once emit_slots exists.
    struct SelfEmit {                                     //
        void* ctx = nullptr;                              //
        void (*fn)(void*, uint32_t, uint32_t) = nullptr;  //
    } self_emit;                                          //
    // Append one marker at a fresh timestamp. If the ring cannot hold a sticky plus a marker,
    // publish first, and drop (counted) only if that could not free room.
    auto self_mark_w0 = [&](uint32_t w0) -> bool {                                          //
        if constexpr (kSelfZones == 0) {                                                    //
            (void)w0;                                                                       //
            return false;                                                                   //
        } else {                                                                            //
            if (!self_on || self_busy) {                                                    //
                return false;                                                               //
            }                                                                               //
            if (self_tail - self_head > kRingWords - 9u) {                                  //
                if (self_pub.fn != nullptr) {                                               //
                    self_pub.fn(self_pub.ctx);                                              //
                }                                                                           //
                if (self_tail - self_head > kRingWords - 9u) {                              //
                    self_dropped++;                                                         //
                    return false;                                                           //
                }                                                                           //
            }                                                                               //
            // Clock read after any publish, so a marker never carries a time from before the
            // publish that made room for it.
            const uint64_t ts = instr_now();                                            //
            const uint32_t hi = static_cast<uint32_t>(ts >> 32);                            //
            if (hi != self_hi) {                                                            //
                self_ring[self_tail % kRingWords] = kernel_profiler::spsc_sticky_timer_w0(hi);  //
                self_tail++;                                                                //
                self_hi = hi;                                                               //
            }                                                                               //
            self_ring[self_tail % kRingWords] = w0;                                         //
            self_tail++;                                                                    //
            self_ring[self_tail % kRingWords] = static_cast<uint32_t>(ts & 0xFFFFFFFFu);    //
            self_tail++;                                                                    //
            self_markers++;                                                                 //
            return true;                                                                    //
        }                                                                                   //
    };                                                                                      //
    // The zone-scope hooks: an uninstrumented sweep pays a flag check, never a call.
    auto self_mark_now = [&](uint32_t w0) -> bool {   //
        if constexpr (kSelfZones == 0) {              //
            (void)w0;                                 //
            return false;                             //
        } else {                                      //
            if (!self_on || self_busy) {              //
                return false;                         //
            }                                         //
            return self_mark_w0(w0);                  //
        }                                             //
    };                                                //
    auto self_mark_phase = [&](uint32_t w0) -> bool {  //
        if constexpr (!kSelfPhases) {                 //
            (void)w0;                                 //
            return false;                             //
        } else {                                      //
            if (!self_on || self_busy) {              //
                return false;                         //
            }                                         //
            return self_mark_w0(w0);                  //
        }                                             //
    };                                                //
    using SelfMarkNow = decltype(self_mark_now);      //
    using SelfMarkPhase = decltype(self_mark_phase);  //
    // One-line zone scopes for the pipeline: constructor marks the begin, destructor the end.
    using ZoneSweep = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SWEEP, SelfMarkNow>;               //
    using ZoneDrain = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_DRAIN, SelfMarkNow>;               //
    using ZonePace = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PACE, SelfMarkNow>;                 //
    using ZoneSync = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_SYNC, SelfMarkNow>;                 //
    using ZoneRead = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ, SelfMarkPhase>;               //
    using ZoneReadWait = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_READ_WAIT, SelfMarkPhase>;      //
    using ZoneProc = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_PROC, SelfMarkPhase>;               //
    using ZoneWrite = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WRITE, SelfMarkPhase>;             //
    using ZoneCreditWait = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_CREDIT_WAIT, SelfMarkPhase>;  //
    using ZoneWrBarrier = kernel_profiler::SpscZoneScope<kernel_profiler::DRISC_ZONE_WR_BARRIER, SelfMarkPhase>;    //
    // Arm mid-sweep: deciding at sweep top would miss the first sweep of every burst. The window
    // is refreshed on every later discovery of work so coverage inside a burst stays contiguous.
    auto self_arm = [&]() {                                  //
        if constexpr (kSelfZones == 0) {                     //
            return;                                          //
        } else {                                             //
            self_work = true;                                //
            self_armed_until = self_t_sweep0 + kSelfHoldCycles;  //
            if (self_on || self_busy) {                      //
                return;                                      //
            }                                                //
            if (self_frames >= kSelfMaxFrames) {             //
                return;                                      //
            }                                                //
            self_on = true;                                  //
            self_windows++;                                  //
        }                                                    //
    };                                                       //

    // Self-frame slot init: the self frame ships RAW (whole rings, decoded circularly against its
    // control vector), so the marker ring never needs packing. Its geometry is constant.
    volatile tt_l1_ptr uint32_t* self_ctrl =                                                                 //
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + kSelfSlot * kSlotBytes + kPrefix * 4u);  //
    if constexpr (kSelfZones != 0) {                                                                         //
        volatile tt_l1_ptr uint32_t* pfx =                                                                   //
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageBase + kSelfSlot * kSlotBytes);             //
        pfx[0] = kernel_profiler::spsc_span_w0() | kernel_profiler::SPSC_SPAN_RAW_FLAG;                      //
        pfx[1] = kSpanWords;                                                                                 //
        slot_payload[kSelfSlot] = kSpanWords;                                                                //
        for (uint32_t k = 2; k < kPrefix; k++) {                                                             //
            pfx[k] = 0;                                                                                      //
        }                                                                                                    //
        // Zero the whole control vector: it ships verbatim and the host reads all five lanes.
        // Rings 1..4 must read head == tail == 0 forever or the decoder would walk garbage.
        for (uint32_t k = 0; k < kCtrlWords; k++) {                                                          //
            self_ctrl[k] = 0;                                                                                //
        }                                                                                                    //
        self_ctrl[kernel_profiler::SPSC_CORE_XY] = kSelfXY;                                                  //
    }                                                                                                        //
    if constexpr (kNocFootprint != 0) {                                                                      //
        // Seed the register mirrors here so the first sweep's delta excludes everything bring-up
        // did (the accumulators start at zero).
        nf_sample_regs(&nf);                                                                                 //
    }                                                                                                        //

    // Recording helpers. The pipeline calls these on single `//`-marked lines; each compiles to
    // nothing when its tier is off.
    auto instr_add32 = [](uint32_t& c, uint32_t v) {  //
        if constexpr (kInstr != 0) {                  //
            c += v;                                   //
        } else {                                      //
            (void)c, (void)v;                         //
        }                                             //
    };                                                //
    auto cvage_stamp = [&](uint64_t& t) {  //
        if constexpr (kInstr != 0) {       //
            t = instr_now();               //
        } else {                           //
            (void)t;                       //
        }                                  //
    };                                     //
    // Age of the tails snapshot backing this generation at the moment its heads go out -- the
    // ring words produced inside this window are invisible to the relief.
    auto cvage_relief = [&]() {                             //
        if constexpr (kInstr != 0) {                        //
            const uint64_t a = instr_now() - t_cv_cur;      //
            cvage_sum += a;                                 //
            cvage_n++;                                      //
            if (a > cvage_max) {                            //
                cvage_max = static_cast<uint32_t>(a);       //
            }                                               //
            t_cv_cur = t_cv_next;                           //
        }                                                   //
    };                                                      //
    auto instr_max32 = [](uint32_t& mx, uint32_t v) {  //
        if constexpr (kInstr != 0) {                   //
            if (v > mx) {                              //
                mx = v;                                //
            }                                          //
        } else {                                       //
            (void)mx, (void)v;                         //
        }                                              //
    };                                                 //
    // Split into an inline constexpr gate (so record builds fold the call away entirely) and a
    // non-inlined body (so diagnostic builds pay one copy, not one per call site).
    auto instr_add64_impl = [](uint64_t& c, uint64_t t0) __attribute__((noinline)) {  //
        c += instr_now() - t0;                                                        //
    };                                                                                //
    auto instr_add64 = [&](uint64_t& c, uint64_t t0) {  // bill now - t0 to a phase counter //
        if constexpr (kInstr != 0) {                    //
            instr_add64_impl(c, t0);                    //
        } else {                                        //
            (void)c, (void)t0;                          //
        }                                               //
    };                                                  //
    auto instr_lap = [](uint64_t& c, uint64_t prev) -> uint64_t {  // bill a segment, return its end //
        if constexpr (kInstr != 0) {                               //
            const uint64_t t = instr_now();                    //
            c += t - prev;                                         //
            return t;                                              //
        } else {                                                   //
            (void)c;                                               //
            return prev;                                           //
        }                                                          //
    };                                                             //
    auto instr_cv_end = [&](uint64_t t_cv0) {            //
        if constexpr (kInstr != 0) {                     //
            const uint64_t d = instr_now() - t_cv0;  //
            c_read += d;                                 //
            c_cv += d;                                   //
        } else {                                         //
            (void)t_cv0;                                 //
        }                                                //
    };                                                   //
    auto instr_batch_end = [&](uint64_t t_batch0, uint64_t t_issue, uint64_t t_after_proc) {  //
        if constexpr (kInstr != 0) {                                                          //
            c_issue += t_issue - t_batch0;                                                    //
            c_read += (t_issue - t_batch0) + (instr_now() - t_after_proc);                //
        } else {                                                                              //
            (void)t_batch0, (void)t_issue, (void)t_after_proc;                                //
        }                                                                                     //
    };                                                                                        //
    auto instr_reserve_end = [&](uint64_t t0) -> uint64_t {      //
        if constexpr (kInstr == 0) {                             //
            return t0;                                           //
        } else {                                                 //
            const uint64_t t1 = instr_now();                 //
            c_reserve += t1 - t0;                                //
            if (static_cast<uint32_t>(t1 - t0) > max_reserve) {  //
                max_reserve = static_cast<uint32_t>(t1 - t0);    //
            }                                                    //
            return t1;                                           //
        }                                                        //
    };                                                           //
    auto instr_wr_end = [&](uint64_t t1, uint64_t t3) {  //
        if constexpr (kInstr != 0) {                     //
            const uint64_t t4 = instr_now();         //
            c_wr_notify += t4 - t3;                      //
            c_write += t4 - t1;                          //
        } else {                                         //
            (void)t1, (void)t3;                          //
        }                                                //
    };                                                   //
    // Worst case since t0 and when it happened; `en` lets a caller veto recording (an idle pump
    // pass is not a worst case).
    auto svc_note_max_impl = [](uint32_t& mx, uint32_t& mx_at, uint64_t t0) __attribute__((noinline)) {  //
        const uint32_t dt = static_cast<uint32_t>(instr_now() - t0);                                     //
        if (dt > mx) {                                                                                   //
            mx = dt;                                                                                     //
            mx_at = static_cast<uint32_t>(t0);                                                           //
        }                                                                                                //
    };                                                                                                   //
    auto svc_note_max = [&](bool en, uint32_t& mx, uint32_t& mx_at, uint64_t t0) {  //
        if constexpr (kSvcInstr != 0) {                                             //
            if (en) {                                                               //
                svc_note_max_impl(mx, mx_at, t0);                                   //
            }                                                                       //
        } else {                                                                    //
            (void)en, (void)mx, (void)mx_at, (void)t0;                              //
        }                                                                           //
    };                                                                              //
    // Per-core service interval (the gap between two ships of the same core) into a log2 histogram
    // starting at 8192 cycles, plus the head-write phase split.
    auto svc_head_begin = [&](uint32_t c) -> uint64_t {                          //
        if constexpr (kSvcInstr != 1) {                                          //
            (void)c;                                                             //
            return 0;                                                            //
        } else {                                                                 //
            const uint64_t t_h0 = instr_now();                               //
            if (last_ship[c] != 0) {                                             //
                const uint32_t dt = static_cast<uint32_t>(t_h0) - last_ship[c];  //
                if (dt > svc_max) {                                              //
                    svc_max = dt;                                                //
                }                                                                //
                uint32_t b = 0;                                                  //
                for (uint32_t q = dt >> 13; q != 0 && b < 7u; q >>= 1) {         //
                    b++;                                                         //
                }                                                                //
                svc_hist[b]++;                                                   //
            }                                                                    //
            last_ship[c] = static_cast<uint32_t>(t_h0);                          //
            return t_h0;                                                         //
        }                                                                        //
    };                                                                           //
    auto svc_head_end = [&](uint64_t t_h0) {      //
        if constexpr (kSvcInstr == 1) {           //
            c_ph_head += instr_now() - t_h0;  //
        } else {                                  //
            (void)t_h0;                           //
        }                                         //
    };                                            //
    auto self_arm_if = [&](bool work) {   //
        if constexpr (kSelfZones != 0) {  //
            if (!self_on && work) {       //
                self_arm();               //
            }                             //
        } else {                          //
            (void)work;                   //
        }                                 //
    };                                    //
    // Sweep bookkeeping: begin snapshots the phase baselines; end returns whether the sweep
    // shipped and folds in the worst-sweep split, busy/idle billing, the fill histogram, the NoC
    // footprint and the workload window's opening snapshot. The window's closing snapshot is taken
    // after the pace gap so it includes this sweep's pace time.
    auto instr_sweep_begin = [&]() -> uint64_t {         //
        if constexpr (kInstr != 0 || kSelfZones != 0) {  //
            s_read0 = c_read, s_proc0 = c_proc;          //
            s_rsv0 = c_reserve, s_wr0 = c_write;         //
            s_bar0 = c_barrier;                          //
            return instr_now();                          //
        } else {                                         //
            return 0;                                    //
        }                                                //
    };                                                   //
    auto instr_sweep_lap = [](uint64_t t_sweep0) -> uint32_t {                                             //
        return (kInstr != 0 || kSelfZones != 0) ? static_cast<uint32_t>(instr_now() - t_sweep0) : 0u;      //
    };                                                                                                     //
    auto instr_sweep_end = [&](uint64_t t_sweep0, uint32_t sweep_cyc, uint32_t frames0, uint32_t peak) -> bool {  //
        const bool worked = frames != frames0;                       //
        if constexpr (kInstr != 0) {                                 //
            if (sweep_cyc > max_sweep) {                             //
                max_sweep = sweep_cyc;                               //
                ws_read = static_cast<uint32_t>(c_read - s_read0);   //
                ws_proc = static_cast<uint32_t>(c_proc - s_proc0);   //
                ws_rsv = static_cast<uint32_t>(c_reserve - s_rsv0);  //
                ws_wr = static_cast<uint32_t>(c_write - s_wr0);      //
                ws_bar = static_cast<uint32_t>(c_barrier - s_bar0);  //
            }                                                        //
            if (worked && !win2_open) {                              //
                win2_open = true;                                    //
                w0_t = t_sweep0;                                     //
                w0_busy = c_busy;                                    //
                w0_idle = c_idle;                                    //
                w0_pace = c_pace;                                    //
                w0_frames = frames0;                                 //
                w0_sweeps = sweeps;                                  //
                if constexpr (kSelfZones == 0) {                     //
                    w0_cv = c_cv;                                    //
                    w0_issue = c_issue;                              //
                }                                                    //
            }                                                        //
            if (!worked) {                                           //
                sweeps_idle++;                                       //
                c_idle += sweep_cyc;                                 //
            } else {                                                 //
                c_busy += sweep_cyc;                                 //
                const uint32_t b = peak / (kRingWords >> 3u);        //
                fill_hist[b > 7u ? 7u : b]++;                        //
            }                                                        //
        }                                                            //
        if constexpr (kNocFootprint != 0) {                          //
            nf_sweep_end(&nf, sweeps, t_sweep0, sweep_cyc, worked);  //
        }                                                            //
        return worked;                                               //
    };                                                               //
    auto instr_win_last = [&](bool worked) {  //
        if constexpr (kInstr != 0) {          //
            if (!worked) {                    //
                return;                       //
            }                                 //
            w1_t = instr_now();           //
            w1_busy = c_busy;                 //
            w1_idle = c_idle;                 //
            w1_pace = c_pace;                 //
            w1_frames = frames;               //
            w1_sweeps = sweeps;               //
            if constexpr (kSelfZones == 0) {  //
                w1_cv = c_cv;                 //
                w1_issue = c_issue;           //
            }                                 //
        } else {                              //
            (void)worked;                     //
        }                                     //
    };                                        //
    // The results block -- every counter the host reads. Called once, at kernel exit.
    auto instr_write_results = [&](uint64_t t_start, uint64_t t_end) {  //
    const uint64_t cycles_v = t_end - t_start;  //
        volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kResultsAddr);  //
        auto out64 = [&](uint32_t i, uint64_t v) {                            //
            out[i] = static_cast<uint32_t>(v & 0xFFFFFFFFu);                  //
            out[i + 1] = static_cast<uint32_t>(v >> 32);                      //
        };                                                                    //
        out64(0, cycles_v);  //
        out64(2, total_words);  //
        out[4] = sweeps;                                                                                  //
        out[5] = pages;                                                                                   //
        out[6] = frames;                                                                                  //
        out[7] = max_occ;                                                                                 //
        out[8] = overflows;                                                                               //
        out[9] = pushes;  // (slots of a disabled tier stay unwritten: the host pre-zeroes the buffer) //                                                                                  //
        if constexpr (kInstr != 0) {  //
        out64(10, c_read);  //
        out64(12, c_proc);  //
        out64(14, c_reserve);  //
        out64(16, c_write);  //
        out64(18, c_barrier);  //
        out[20] = sweeps_idle;                                                                            //
        out64(21, c_idle);  //
        out64(23, c_busy);  //
        out[25] = max_sweep;                                                                              //
        out[26] = max_reserve;                                                                            //
        out64(27, c_wr_chunk);  //
        out64(29, c_wr_push);  //
        out64(31, c_wr_notify);  //
        }  // kInstr != 0 //
        out[33] = credit_timeouts;                                                                        //
        out[34] = dropped_frames;                                                                         //
        out[36] = noc_index_before;                                                                       //
        out[37] = noc_index_after;                                                                        //
        out[38] = NOC_INDEX;                                                                              //
        out[39] = kReadNoc;                                                                               //
        if constexpr (kSvcInstr == 1) {  //
        out64(40, c_ph_head);  //
        }  // kSvcInstr == 1 //
        out[42] = gap;  // where the pacing controller settled //
        if constexpr (kInstr != 0) {  //
        out[43] = ws_read;                                                                                //
        out[44] = ws_proc;                                                                                //
        out[45] = ws_rsv;                                                                                 //
        out[46] = ws_wr;                                                                                  //
        out[47] = ws_bar;                                                                                 //
        }  // kInstr != 0 //
        out[48] = spool_drops;                                                                            //
        out[49] = drain_chunks;                                                                           //
        out[50] = drain_starved;                                                                          //
        out[51] = spool_max;                                                                              //
        out[52] = drain_ships;                                                                            //
        if constexpr (kInstr != 0) {  //
        for (uint32_t k = 0; k < 8u; k++) {                                                               //
            out[53 + k] = fill_hist[k];                                                                   //
        }                                                                                                 //
        out64(61, c_scan);  //
        out[63] = sweeps * num_cores;  // scan visits: every sweep scans every core //
        if constexpr (kSelfZones != 0) {  //
        }  // kInstr != 0 //
        out[64] = self_frames;                                                                            //
        out[65] = self_markers;                                                                           //
        out[66] = self_sweeps;                                                                            //
        out[67] = self_sweeps_work;                                                                       //
        out[68] = self_windows;                                                                           //
        out[69] = self_over;                                                                              //
        out[70] = self_dropped;                                                                           //
        out64(71, c_self);  //
        out[73] = self_tail;                                                                              //
        out[74] = self_ck_sweeps;                                                                         //
        out64(75, self_ck_read);  //
        out64(77, self_ck_proc);  //
        out64(79, self_ck_rsv);  //
        out64(81, self_ck_write);  //
        out64(83, self_ck_bar);  //
        out[85] = kSelfMaxFrames;                                                                         //
        out[86] = kSelfDetail;                                                                            //
        out[87] = self_words_shipped;  // must equal out[73], or trace was lost at teardown //
        }  // kSelfZones != 0 //
        // NoC footprint: out[88..103] lifetime, out[104..119] the workload window, same slot order.
        if constexpr (kNocFootprint != 0) {                                                               //
            // Final sample, so the lifetime block includes the last sweep and the exit drain.
            nf_sample_regs(&nf);                                                                          //
            uint32_t o = 88;                                                                              //
            for (uint32_t i = 0; i < kNfSlots; i++) {                                                     //
                out[o++] = static_cast<uint32_t>(nf.life[i] & 0xFFFFFFFFu);                               //
                out[o++] = static_cast<uint32_t>(nf.life[i] >> 32);                                       //
            }                                                                                             //
            for (uint32_t i = 0; i < kNfSlots; i++) {                                                     //
                const uint64_t v = nf.win_last[i] - nf.win_base[i];                                       //
                out[o++] = static_cast<uint32_t>(v & 0xFFFFFFFFu);                                        //
                out[o++] = static_cast<uint32_t>(v >> 32);                                                //
            }                                                                                             //
            out[120] = nf.win_open ? (nf.win_sweep_last - nf.win_sweep_first + 1u) : 0u;                  //
            const uint64_t win_cyc = nf.win_open ? (nf.win_t1 - nf.win_t0) : 0u;                          //
            out64(121, win_cyc);  //
            out[123] = static_cast<uint32_t>(nf.cost & 0xFFFFFFFFu);  // the instrument's own cost //
            out[124] = static_cast<uint32_t>(nf.cost >> 32);                                              //
            out[125] = 0;                                                                                 //
            out[126] = 0;                                                                                 //
            out[127] = nf.win_open ? 1u : 0u;                                                             //
            out[128] = kNocFootprint;   // echo, so the host never guesses whether this block is valid //
            out[129] = NOC_WORD_BYTES;  // byte scale from the header, so the host never hardcodes it //
        }                                                                                                 //
        if constexpr (kSyncEvent != 0) {  //
        out[130] = sync_events;                                                                           //
        out[131] = sync_timeouts;                                                                         //
        out[132] = sync_spin_cyc;                                                                         //
        }  // kSyncEvent != 0 //
        out[133] = stop_sweeps;                                                                           //
        out[134] = static_cast<uint32_t>(total_words - words_at_stop);                                    //
        out[135] = drain_dead ? 1u : 0u;                                                                  //
        out64(136, c_pace);  //
        if constexpr (kInstr != 0) {  //
        out64(138, c_drain);  //
        }  // kInstr != 0 //
        out64(140, spool_wr);  //
        out[142] = static_cast<uint32_t>(spool_wr - spool_rd);  // stranded; nonzero only with drain_dead //
        if constexpr (kInstr != 0) {  //
        out[170] = ship_deferred;                                                                         //
        out[171] = 0;                                                                                     //
        out64(172, c_issue);  //
        out[174] = n_gather_rd;                                                                           //
        out[175] = n_cv_rd;                                                                               //
        out64(176, c_cv);  //
        {                                                                                                 //
            const uint64_t wc = win2_open ? w1_t - w0_t : 0u;                                             //
            out64(181, wc);  //
            const uint64_t wb = win2_open ? w1_busy - w0_busy : 0u;                                       //
            out64(183, wb);  //
            const uint64_t wi = win2_open ? w1_idle - w0_idle : 0u;                                       //
            out64(185, wi);  //
            const uint64_t wp = win2_open ? w1_pace - w0_pace : 0u;                                       //
            out64(187, wp);  //
            out[189] = win2_open ? w1_frames - w0_frames : 0u;                                            //
            out[190] = win2_open ? w1_sweeps - w0_sweeps : 0u;                                            //
            out[191] = win2_open ? 1u : 0u;                                                               //
            out[192] = num_cores;                                                                         //
        }                                                                                                 //
        // The read-split window's counters only exist when the self-zone build (which trades the same
        // code-region bytes) is off.
        if constexpr (kSelfZones == 0) {                                                                  //
            const uint64_t wcv = win2_open ? w1_cv - w0_cv : 0u;                                          //
            out64(202, wcv);  //
            const uint64_t wis = win2_open ? w1_issue - w0_issue : 0u;                                    //
            out64(204, wis);  //
        }                                                                                                 //
        out[214] = cvage_n;                                                                               //
        out64(215, cvage_sum);  //
        out[217] = cvage_max;                                                                             //
        }  // kInstr != 0 //
        if constexpr (kSvcInstr == 1) {  //
        out[193] = svc_max;                                                                               //
        for (uint32_t i = 0; i < 8; i++) {                                                                //
            out[194 + i] = svc_hist[i];                                                                   //
        }                                                                                                 //
        }  // kSvcInstr == 1 //
        if constexpr (kSvcInstr != 0) {  //
        out[206] = max_bar;                                                                               //
        out[207] = max_bar_at;                                                                            //
        out[208] = max_rdw;                                                                               //
        out[209] = max_rdw_at;                                                                            //
        out[210] = max_cvw;                                                                               //
        out[211] = max_cvw_at;                                                                            //
        out[212] = max_pump;                                                                              //
        out[213] = max_pump_at;                                                                           //
        }  // kSvcInstr != 0 //
        static_assert(                                                                                    //
            kernel_profiler::SPSC_DRAIN_RESULT_WORDS >= 218,                                              //
            "the results block must hold the self-profiling, NoC-footprint and histogram counters");      //
    };  //
    // ==== end instrumentation ====================================================================
    // ==== instrumentation: self-frame publish =======================================================
    // Publish the marker ring as one RAW frame, shipped through the self_emit hook (emit_slots,
    // wired by the pipeline). The staging-reuse barrier sits at the end so the wait lands before
    // the next publish, not on this one's critical path. Phase counters are saved and restored
    // around the egress call so the self frame does not bill itself.
    auto self_publish = [&]() {                                                                     //
        if constexpr (kSelfZones == 0) {                                                            //
            return;                                                                                 //
        } else {                                                                                    //
            if (self_tail == self_head) {                                                           //
                return;                                                                             //
            }                                                                                       //
            const uint64_t t_s0 = instr_now();                                                  //
            self_busy = true;                                                                       //
            self_ctrl[kernel_profiler::SPSC_RING_HEAD_0] = self_head;                               //
            self_ctrl[kernel_profiler::SPSC_RING_TAIL_0] = self_tail;                               //
            asm volatile("fence" ::: "memory");                                                     //
            const uint64_t s_rsv = c_reserve, s_wr = c_write, s_ch = c_wr_chunk;                    //
            const uint64_t s_pu = c_wr_push, s_no = c_wr_notify;                                    //
            const uint32_t s_pages = pages, s_pushes = pushes, s_maxr = max_reserve;                //
            if (self_emit.fn != nullptr) {                                                          //
                self_emit.fn(self_emit.ctx, kSelfSlot, 1u);                                         //
            }                                                                                       //
            c_reserve = s_rsv;                                                                      //
            c_write = s_wr;                                                                         //
            c_wr_chunk = s_ch;                                                                      //
            c_wr_push = s_pu;                                                                       //
            c_wr_notify = s_no;                                                                     //
            pages = s_pages;                                                                        //
            pushes = s_pushes;                                                                      //
            max_reserve = s_maxr;                                                                   //
            if constexpr (kSpool) {                                                                 //
                while (experimental::dma_get_writes_outstanding(kDmaShip) != 0) {                   //
                }                                                                                   //
            } else {                                                                                //
                while (!ncrisc_noc_nonposted_writes_flushed(NOC_INDEX)) {                           //
                }                                                                                   //
            }                                                                                       //
            self_words_shipped += self_tail - self_head;                                            //
            self_head = self_tail;                                                                  //
            self_frames++;                                                                          //
            self_busy = false;                                                                      //
            c_self += instr_now() - t_s0;                                                       //
        }                                                                                           //
    };                                                                                              //
    // Captureless wrapper, so the marker path's trampoline gets a plain function pointer.
    if constexpr (kSelfZones != 0) {                                                                //
        self_pub.ctx = static_cast<void*>(&self_publish);                                           //
        self_pub.fn = [](void* p) { (*static_cast<decltype(self_publish)*>(p))(); };                //
    }                                                                                               //
    // Sweep-top arming and the host-triggered sync rendezvous (kSyncEvent); both can publish,
    // so they live behind self_publish.
    auto self_sweep_arm = [&](uint64_t t_sweep0) {  //
            // Inside an active window every sweep is instrumented -- one compare against a deadline.
            // Only a window's first sweep is partial, armed mid-body when work is discovered.
            if constexpr (kSelfZones != 0) {               //
                self_t_sweep0 = t_sweep0;                  //
                self_work = false;                         //
                self_on = false;                           //
                self_from_start = false;                   //
                if (self_frames >= kSelfMaxFrames) {       //
                    self_over++;                           //
                    // The budget running out is not a reason to discard zones already written.
                    self_publish();                        //
                } else if (t_sweep0 < self_armed_until) {  //
                    self_on = true;                        //
                    self_from_start = true;                //
                }                                          //
            }                                              //
    };  //
    auto self_sync_event = [&]() {  //
            // First in the loop body, before the sweep starts, so the barrier wait is billed to no
            // sweep. The host parks every drainer on the ack, then one release makes them all stamp
            // the same instant.
            if constexpr (kSyncEvent != 0) {                                                                     //
                invalidate_l1_cache();                                                                           //
                const uint32_t req = *sync_req;                                                                  //
                if (req != sync_seen) {                                                                          //
                    sync_seen = req;                                                                             //
                    const uint64_t t_park = instr_now();                                                     //
                    *sync_ack = req;                                                                             //
                    uint64_t t_go = 0;                                                                           //
                    // Bounded, so a host that never releases degrades instead of wedging the workload.
                    uint32_t guard = 0xFFFFFFFFu;                                                                //
                    for (;;) {                                                                                   //
                        invalidate_l1_cache();                                                                   //
                        if (*sync_go == req) {                                                                   //
                            t_go = instr_now();  // the measured instant //
                            break;                                                                               //
                        }                                                                                        //
                        if (*stop != 0 || --guard == 0) {                                                        //
                            break;                                                                               //
                        }                                                                                        //
                    }                                                                                            //
                    if (t_go != 0) {                                                                             //
                        // Forced emission: the sync zone must land whether or not this sweep would have
                        // been instrumented, and it ships immediately.
                        self_on = true;                                                                          //
                        {                                    //
                            ZoneSync z_sync(self_mark_now);  //
                        }                                    //
                        self_publish();                                                                          //
                        sync_events++;                                                                           //
                        sync_spin_cyc = static_cast<uint32_t>(t_go - t_park);                                    //
                    } else {                                                                                     //
                        sync_timeouts++;                                                                         //
                    }                                                                                            //
                }                                                                                                //
            }                                                                                                    //
    };  //
    auto self_sweep_account = [&](uint64_t t_sweep0, uint32_t sweep_cyc, bool busy) {  //
            if constexpr (kSelfZones != 0) {                                    //
                if (busy) {                                                     //
                    self_armed_until = t_sweep0 + sweep_cyc + kSelfHoldCycles;  //
                }                                                               //
                if (self_on) {                                                  //
                    self_sweeps++;                                              //
                    if (busy || self_work) {                                    //
                        self_sweeps_work++;                                     //
                    }                                                           //
                    if (self_from_start) {                                      //
                        self_ck_sweeps++;                                       //
                        self_ck_read += c_read - s_read0;                       //
                        self_ck_proc += c_proc - s_proc0;                       //
                        self_ck_rsv += c_reserve - s_rsv0;                      //
                        self_ck_write += c_write - s_wr0;                       //
                        self_ck_bar += c_barrier - s_bar0;                      //
                    }                                                           //
                }                                                               //
            }                                                                   //
    };  //
    auto self_flush_sweep = [&]() {  //
            // After the pace gap, so the PACE zone rides in the same frame as the sweep it follows.
            if constexpr (kSelfZones != 0) {                    //
                if (self_on) {                                  //
                    if (instr_now() >= self_armed_until) {  //
                        self_publish();                         //
                    }                                           //
                    self_on = false;                            //
                }                                               //
            }                                                   //
    };  //
    auto self_flush_final = [&]() {  //
        // Gated on tail != head, not on self_on -- self_on is cleared at the end of every sweep.
        if constexpr (kSelfZones != 0) {  //
            self_on = false;              //
            self_publish();               //
        }                                 //
    };  //
    // ==== end instrumentation =======================================================================

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
        return occ >= kSpoolBytes / 2u + kSpoolBytes / 8u    ? 3u
               : occ >= kSpoolBytes / 2u                     ? 2u
               : occ >= kSpoolBytes / 4u + kSpoolBytes / 8u  ? 1u
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
            I_PUMP_BEGIN();
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
            if (want_refill && spool_done != spool_wr && static_cast<uint32_t>(spool_done - spool_rd_iss) < kBounceBytes &&
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
                if (nb == 0) {
                    I_STARVED();
                } else {
                    push_fifo(kBounceBase0 + rdy * kBounceBytes + b_off[rdy], sender.write_ptr, nb);
                    socket_push_pages(sender, nb / kPageBytes);
                    notify_pending = true;
                    pages += nb / kPageBytes;
                    pushes++;
                    I_SHIPPED();
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
            I_PUMP_END();
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
                spool_drops += count;
                dropped_frames += count;
                return;
            }
            // The DMA engine reads the control and length words the scalar core staged; Blackhole
            // stores can reach SRAM out of order.
            asm volatile("fence" ::: "memory");
            I_SPOOL_BEGIN();
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
            if (occ > spool_max) {  //
                spool_max = occ;    //
            }                       //
            I_SPOOL_END();
            *phase = kPhWrDone;
            return;
        }
        // Direct push: reserve host FIFO credit, then write the frames straight to the host.
        uint32_t npages = 0;
        for (uint32_t f = 0; f < count; f++) {
            npages += kernel_profiler::spsc_span_frame_words(slot_payload[start + f]) / kPageWords;
        }
        asm volatile("fence" ::: "memory");
        I_WR_BEGIN();
        *phase = kPhaseReserve;
        bool credited;
        {
            // Suppressed while self_publish ships the self frame through this same path, so the
            // self frame's own egress is never a zone.
            I_ZONE_CREDIT();
            credited = reserve_pages(sender, npages, stop);
        }
        *phase = kPhaseWrite;
        if (!credited) {
            // Drop rather than block: the heads for these slots were already written back, so the
            // producers stay unblocked and the workload completes. Capture is best-effort; the
            // workload is not.
            *phase = kPhDropped;
            credit_timeouts++;
            dropped_frames += count;
            I_WR_DROP();
            return;
        }
        I_WR_CREDITED();
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
        I_WR_CHUNKED();
        *phase = kPhWrPush;
        socket_push_pages(sender, npages);
        I_WR_PUSHED();
        *phase = kPhWrNotify;
        notify_host();
        I_WR_DONE();
        *phase = kPhWrDone;
        pages += npages;
        pushes++;
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

    I_WIRE_SELF_EMIT();

    // Main loop. On stop=1, keep sweeping until one whole sweep moves nothing, so markers still in
    // worker rings ship instead of being stranded; exiting on the stop word directly is what
    // silently truncated captures.
    uint64_t stop_seen_at = 0;
    uint32_t frames_at_stop_check = 0;
    const uint64_t t_start = get_timestamp();
    while (sweeps < kMaxSweeps) {
        invalidate_l1_cache();
        if (*stop != 0) {
            if (stop_seen_at == 0) {
                stop_seen_at = get_timestamp();
                I_STOP_WORDS();
            } else if (frames == frames_at_stop_check || get_timestamp() - stop_seen_at > kStopDrainCycles) {
                break;
            }
            frames_at_stop_check = frames;
            I_STOP_SWEEP();
        }
        I_SYNC_EVENT();
        sweeps++;
        *hb = sweeps;
        *phase = kPhasePoll;
        const uint32_t frames_at_sweep_start = frames;
        I_SWEEP_BEGIN();

        I_SELF_SWEEP_ARM();

        uint32_t sweep_peak = 0;
        bool sweep_grew = false;
        {
            // Constructed after the arming block decided self_on, so an armed sweep records its
            // whole body.
            I_ZONE_SWEEP();
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
            {
                I_CV_BEGIN();
                {
                    I_CV_ZONE();
                    // Responses can arrive out of order, so a counted response may belong to a
                    // later core -- a chunk core then scans last sweep's tails, which are stale but
                    // valid (tails are monotonic): it under-ships and catches up next visit.
                    // Only the first chunk's CVs are read here; the rest of the grid's are issued
                    // at the refill pause, mid-sweep, so the late scan sees tails fresh enough to
                    // catch a core that started producing in this very sweep.
                    for (uint32_t i = 0; i < cv_chunk; i++) {
                        noc_async_read<kCvReadBytes>(
                            core_noc[i] + kCvReadSrcOff, kCvBase + i * kCvReadBytes, kCvReadBytes, kReadNoc);
                    }
                    I_CV_ISSUED();
                    while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED) - rd0 < cv_chunk) {
                    }
                    invalidate_l1_cache();
                    I_CV_WAITED();
                    I_CVAGE_WAVE();
                }
                I_CV_END();
            }
            uint32_t scan_lo = 0;
            uint32_t scan_hi = cv_chunk;
            uint32_t cur = 0;
            for (;;) {
                I_SCAN_BEGIN();
                for (uint32_t c = scan_lo; c < scan_hi; c++) {
                    const tt_l1_ptr uint32_t* __restrict tails =
                        reinterpret_cast<const tt_l1_ptr uint32_t*>(kCvBase + c * kCvReadBytes);
                    uint32_t* __restrict mine = &head_mirror[c * kNumRisc];
                    if (!seeded[c]) {
                        // Seed the mirrors from the tails: everything written before this filler
                        // first saw the core predates the workload.
                        for (uint32_t r = 0; r < kNumRisc; r++) {
                            mine[r] = tails[r];
                        }
                        seeded[c] = 1;
                    }
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
                    I_SCAN_PEAK();
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
                        I_SCAN_DEFER();
                        continue;
                    }
                    hot[c] = 1;
                    ship_list[n_ship++] = static_cast<uint8_t>(c);
                }
                I_SCAN_END();
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
                            overflows++;
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
                        I_GATHER_READ();
                        const uint32_t hm = start & (kRingWords - 1u);
                        if (img) {
                            // A near-full wrapping run ships as its whole ring image in one read;
                            // the decoder linearises by head using the same shared predicate. This
                            // is nearly every lane at the saturation boundary, where the wrap
                            // split's second issue is pure loss. One ring per read is also the
                            // measured optimum: coalescing adjacent whole-ring lanes into bigger
                            // reads starves the producer's own L1 port (a clean dose-response, up
                            // to ~70x the stall floor at five rings per read).
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(                 kReadNoc, read_cmd_buf, ring_src, slot + off * 4u, kRingWords * 4u);
                            off += kRingWords;
                        } else if (hm + take > kRingWords) {
                            // A small wrapping run ships as the two-piece split, byte-exact: at
                            // sustained rates the image's dead remainder is most of the ring, and
                            // there the drain, not the sweep, is the binding resource.
                            const uint32_t first = kRingWords - hm;
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(                 kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, first * 4u);
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(                 kReadNoc, read_cmd_buf, ring_src, slot + (off + first) * 4u, (take - first) * 4u);
                            I_GATHER_READ();
                            off += take;
                        } else {
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(                 kReadNoc, read_cmd_buf, ring_src + hm * 4u, slot + off * 4u, take * 4u);
                            off += take;
                        }
                    }
                    cv[kernel_profiler::SPSC_WIRE_XY] = xy;
                    total_words += live;
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
                    I_PROC_BEGIN();
                    for (uint32_t i = 0; i < n; i++) {
                        const uint32_t sl = g * kGenSlots + i;
                        const uint32_t c = slot_core[sl];
                        I_HEAD_BEGIN();
                        const uint32_t sc = kHeadScratch + c * 32u;
                        // Posted (the barriers protect staging reuse, which a head write never
                        // touches; scratch reuse is safe on the slot rotation) and on the read NoC:
                        // on the egress NoC this small packet queues behind frame data, so head
                        // visibility inherited the PCIe tile's acceptance jitter.
                        noc_async_write_one_packet<true, true>(
                            sc, core_noc[c] + kernel_profiler::SPSC_RING_HEAD_0 * 4u, kNumRisc * 4u, kReadNoc);
                        I_HEAD_END();
                        frames++;
                    }
                    I_PROC_END();
                };

                auto ship_frames = [&](uint32_t n, uint32_t g) {
                    // A listed core is live by construction, so a batch with cores is the work
                    // signal that opens the self-zone window.
                    I_SELF_ARM();
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
                        I_BAR_BEGIN();
                        *phase = kPhBar1;
                        {
                            I_ZONE_BAR();
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
                        }
                        I_BAR_END();
                        gen_shipped[gen] = false;
                    }

                    I_BATCH_BEGIN();
                    uint32_t n = 0;
                    {
                        I_ZONE_ISSUE();
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
                            ncrisc_noc_read_set_state<DM_DEDICATED_NOC, false, false>(
                                kReadNoc, read_cmd_buf, core_noc[c]);
                            ncrisc_noc_read_with_state<DM_DEDICATED_NOC, true, false>(
                                kReadNoc, read_cmd_buf, cv_src + kCvReadSrcOff, kCvBase + c * kCvReadBytes,
                                kCvReadBytes);
                        }
                        I_BATCH_CVS();
                        I_CVAGE_PREF();
                    }
                    I_ISSUE_MARK();

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

                    I_PROC_MARK();
                    I_WAIT_BEGIN();
                    {
                        I_ZONE_WAIT();
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
                    }
                    I_WAIT_END();
                    I_BATCH_END();
                    advance_heads(n, gen);
                    I_CVAGE_RELIEF();

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
                for (uint32_t i = scan_hi; i < num_cores; i++) {
                    noc_async_read<kCvReadBytes>(
                        core_noc[i] + kCvReadSrcOff, kCvBase + i * kCvReadBytes, kCvReadBytes, kReadNoc);
                }
                I_LATE_CVS();
                // Wait on the response count, not a full barrier: gather responses also bump it,
                // which only ever lets a scan see stale-but-valid tails (monotonic) -- a benign
                // under-ship.
                while (NOC_STATUS_READ_REG(kReadNoc, NIU_MST_RD_RESP_RECEIVED) - rd0 < num_cores) {
                }
                invalidate_l1_cache();
                I_CVAGE_WAVE();
                scan_hi = num_cores;
            }

            I_SWEEP_LAP();
        }
        I_SWEEP_END();
        // Post-sweep pump, after sweep_cyc is captured so drain time is never billed to the sweep
        // it trails. Busy sweeps below the first band skip the pump entirely: the spool is the
        // burst absorber, and a capture that fits in it deserves pure gather.
        if constexpr (kSpool) {
            if (pump_level >= 2u || (pump_level == 1u && (sweeps & 1u) != 0) || fresh_boost ||
                frames == frames_at_sweep_start) {
                I_DRAIN_BEGIN();
                drain_pump();
                drain_notify();
                I_DRAIN_END();
            }
        }
        I_SELF_ACCOUNT();

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
            const uint64_t t_g0 = get_timestamp();
            {
                // Its own zone, a sibling of the sweep that closed above, never its child.
                I_ZONE_PACE();
                const uint64_t until = t_g0 + gap;
                while (get_timestamp() < until) {
                    if constexpr (kSpool) {
                        drain_pump();  // idle time is drain time
                    }
                }
            }
            I_PACE_END();
        }
        I_WIN_LAST();
        I_SELF_FLUSH_SWEEP();
    }

    I_SELF_FLUSH_FINAL();

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
    {
        const uint64_t t_ps = get_timestamp() + 1350000u;
        while (!(ncrisc_noc_posted_writes_sent(NOC_INDEX) && ncrisc_noc_posted_writes_sent(kReadNoc)) &&
               get_timestamp() < t_ps) {
        }
    }
    const uint64_t t_end = get_timestamp();

    I_RESULTS();

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
    for (uint32_t spins = 0; spins < 200000000u && *stop != 2u; spins++) {
        invalidate_l1_cache();
    }
    experimental::drisc_set_noc2axi_mode_all();
}
