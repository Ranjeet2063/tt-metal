// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// The one cross-thread rendezvous for the perf harness, outside every build guard so both builds
// compile the same barrier. On semaphores, not L1: no traffic on what is being measured.

namespace llk_barrier
{

// Quasar has no semaphore to spare (MATH_PACK, UNPACK_MATH and PACK_UNPACK all belong to the ops) and
// its counter path is excluded by an #error in counters.h, so it keeps the L1 rendezvous in profiler.h.
#if !defined(ARCH_QUASAR)

constexpr std::uint32_t NUM_THREADS = 3; // unpack, math, pack

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

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

// Fixed rather than per-run-type: letting it vary is how the two builds ended up releasing from
// different threads.
constexpr bool is_action_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

#endif // !ARCH_QUASAR

} // namespace llk_barrier
