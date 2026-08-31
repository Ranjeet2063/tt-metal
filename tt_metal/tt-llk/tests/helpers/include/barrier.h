// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// The one cross-thread rendezvous for the perf harness: same name, same signature and the same
// release semantics on every arch, outside every build guard so both builds compile the same barrier.

namespace llk_barrier
{

// counters.h includes this header before its own build guard, so it is also compiled for BRISC, which
// has no thread identity and never rendezvouses. Everything thread-specific is gated on that.
#if defined(LLK_TRISC_UNPACK) || defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK) || defined(LLK_TRISC_ISOLATE_SFPU)
#define LLK_BARRIER_ON_TRISC 1
#endif

// Quasar adds an SFPU thread.
#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_THREADS = 4; // unpack, math, pack, sfpu
#else
constexpr std::uint32_t NUM_THREADS = 3; // unpack, math, pack
#endif

#if defined(LLK_BARRIER_ON_TRISC)

// Thread identity lives here because the barrier needs it before the profiler does; profiler.h
// aliases TRISC_ID onto it so there is one mapping.
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t THREAD_ID = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t THREAD_ID = 1;
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t THREAD_ID = 2;
#else
constexpr std::uint32_t THREAD_ID = 3;
#endif

// Fixed rather than per-run-type: letting it vary is how the two builds ended up releasing from
// different threads. Pack exists on every arch, Quasar included.
constexpr bool is_action_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

#if !defined(ARCH_QUASAR)

// ARRIVE is the perf-designated semaphore; the fuser also posts it, so arrivals are drained every
// round rather than left to accumulate. RELEASE is unused by the ops and by every generated kernel.
constexpr std::uint8_t ARRIVE_SEM  = ckernel::semaphore::PACK_DONE;
constexpr std::uint8_t RELEASE_SEM = ckernel::semaphore::UNPACK_OPERAND_SYNC;

// Peers announce on ARRIVE and wait for a token on RELEASE; the action thread collects the arrivals,
// runs action(), then hands out one token per peer. The release is a token each peer consumes, not a
// level it has to observe, so a peer that samples late still finds its token: no timing assumption.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    if (is_action_thread)
    {
        while (ckernel::semaphore_read(ARRIVE_SEM) < NUM_THREADS - 1)
        {
        }
        while (ckernel::semaphore_read(ARRIVE_SEM) != 0)
        {
            ckernel::semaphore_get(ARRIVE_SEM);
        }

        action();

        for (std::uint32_t i = 0; i < NUM_THREADS - 1; ++i)
        {
            ckernel::semaphore_post(RELEASE_SEM);
        }
    }
    else
    {
        ckernel::semaphore_post(ARRIVE_SEM);
        while (ckernel::semaphore_read(RELEASE_SEM) == 0)
        {
        }
        ckernel::semaphore_get(RELEASE_SEM);
    }

    ckernel::fence_compiler();
}

#else // ARCH_QUASAR

// Quasar names only MATH_PACK, UNPACK_MATH and PACK_UNPACK, all owned by the ops, and its
// PC_BUF_SEMAPHORE_BASE carries a FIXME about tracking the hardware SEM_COUNT, so claiming an unnamed
// index is not safe from here. It gets an L1 rendezvous instead, with the same release guarantee.
// NUM_THREADS words, set by trisc.cpp from the profiler's L1 map (which cannot be included here).
extern volatile std::uint32_t* barrier_slots;

// Two generation rounds: everyone arrives, the action thread runs action(), then everyone waits for
// the release round. Peers block on the action thread's second bump, so they cannot leave before
// action() completes. Generations only ever increase, so a late thread still sees the round it missed
// and nothing depends on catching a transient value. Compared with >= for the same reason.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    volatile std::uint32_t* slots = barrier_slots;

    const std::uint32_t arrive_gen = slots[THREAD_ID] + 1;
    slots[THREAD_ID]               = arrive_gen;
    ckernel::invalidate_data_cache();
    for (std::uint32_t i = 0; i < NUM_THREADS; ++i)
    {
        while (i != THREAD_ID && slots[i] < arrive_gen)
        {
            ckernel::invalidate_data_cache();
        }
    }

    if (is_action_thread)
    {
        action();
    }

    const std::uint32_t release_gen = arrive_gen + 1;
    slots[THREAD_ID]                = release_gen;
    ckernel::invalidate_data_cache();
    for (std::uint32_t i = 0; i < NUM_THREADS; ++i)
    {
        while (i != THREAD_ID && slots[i] < release_gen)
        {
            ckernel::invalidate_data_cache();
        }
    }

    ckernel::fence_compiler();
}

#endif // !ARCH_QUASAR

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

#endif // LLK_BARRIER_ON_TRISC

} // namespace llk_barrier
