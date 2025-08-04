/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

__inline__ __device__ uint64_t get_flat_wid() {
	uint64_t bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id 
    int warpSize = 32;
    uint64_t warps_per_block = (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize;
	return get_warpid() + bid * warps_per_block;
}

extern "C" __device__ __noinline__ void count_bbv(int count_warp_level,
                                                    uint64_t bid,
                                                    uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
    * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(active_mask);

    unsigned long long* bb_counters = **(unsigned long long***)pcounter;

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            atomicAdd(bb_counters + bid, 1);
        } else {
            atomicAdd(bb_counters + bid, num_threads);
        }
    }
}

extern "C" __device__ __noinline__ void count_bbv_per_warp(int count_warp_level,
                                                     uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(active_mask);

    const uint64_t wid = get_global_warp_id();

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            atomicAdd((unsigned long long*)pcounter + wid, 1);
        } else {
            atomicAdd((unsigned long long*)pcounter + wid, num_threads);
        }
    }
}

extern "C" __device__ __noinline__ void count_pred_off(int predicate,
                                                       int count_warp_level,
                                                       uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* get predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);

    /* get mask of threads that have their predicate off */
    const int mask_off = active_mask ^ predicate_mask;

    /* count the number of threads that have their predicate off */
    const int num_threads_off = __popc(mask_off);

    /* only the first active thread updates the counter of predicated off
     * threads */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            if (predicate_mask == 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads_off);
        }
    }
}
