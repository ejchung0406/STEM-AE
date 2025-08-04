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

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

#include <random>
#include <cmath>
#include <fstream>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU threads */
__managed__ uint64_t counter = 0;
__managed__ uint64_t counter_pred_off = 0;
__managed__ uint64_t **ptr_to_bb_counters = nullptr;  // Pointer to the basic block counters
__managed__ uint64_t *bb_counters = nullptr;  // Flattened array: [bb_idx * num_warps + warp_id]
// __managed__ uint64_t *function_offsets = nullptr;  // Array to store starting index for each function's counters

/* global control variables for this tool */
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;
int active_from_start = 1;
bool mangled = false;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* Path to save BBV csv file */
std::string bbv_file_path = "./basic_block_counters.csv";

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_STR(bbv_file_path, "BBV_FILE_PATH",
                "Path to save the basic block counters CSV file");
    if (active_from_start == 0) {
        active_region = false;
    }

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    uint64_t function_id = 0;
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Get the static control flow graph of instruction */
        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) {
            printf(
                "Warning: Function %s is degenerated, we can't compute basic "
                "blocks statically",
                nvbit_get_func_name(ctx, f));
        }

        if (verbose) {
            printf("Function %s\n", nvbit_get_func_name(ctx, f));
            /* print */
            uint64_t cnt = 0;
            for (auto &bb : cfg.bbs) {
                printf("Basic block id %lu - num instructions %ld\n", cnt++,
                       bb->instrs.size());
                for (auto &i : bb->instrs) {
                    i->print(" ");
                }
            }
        }

        if (verbose) {
            printf("inspecting %s - number basic blocks %ld\n",
                   nvbit_get_func_name(ctx, f), cfg.bbs.size());
        }

        /* Iterate on basic block and inject the first instruction */
        uint64_t bb_id = 0;
        for (auto &bb : cfg.bbs) {
            Instr *i = bb->instrs[0];

            uint64_t counter_index = bb_id;
            
            /* inject device function */
            nvbit_insert_call(i, "count_bbv", IPOINT_BEFORE);
            /* add count warp level option */
            nvbit_add_call_arg_const_val32(i, count_warp_level);
            /* BB id */
            nvbit_add_call_arg_const_val64(i, counter_index);
            /* add pointer to counter location */
            nvbit_add_call_arg_const_val64(i, (uint64_t)&ptr_to_bb_counters);
            if (verbose) {
                printf("Injecting count_bbv for Function %lu BB %lu with counter at %p, %p, %p\n", 
                       function_id, bb_id, ptr_to_bb_counters, *ptr_to_bb_counters, &bb_counters[counter_index]);
                i->print("Inject count_instr before - ");

                unsigned long long* bbc = *(unsigned long long**)(uint64_t)&ptr_to_bb_counters;
                printf("%p, %p\n", bbc, bbc+bb_id);
            }

            // uint64_t counter_index = (function_offsets[function_id] + bb_id) * num_of_warps;
            
            // /* inject device function */
            // nvbit_insert_call(i, "count_bbv_per_warp", IPOINT_BEFORE);
            // /* add count warp level option */
            // nvbit_add_call_arg_const_val32(i, count_warp_level);
            // /* add pointer to counter location */
            // nvbit_add_call_arg_const_val64(i, (uint64_t)&bb_counters[counter_index]);
            // if (verbose) {
            //     printf("Injecting count_bbv for Function %lu BB %lu with counter at %p\n", 
            //            function_id, bb_id, &bb_counters[counter_index]);
            //     i->print("Inject count_instr before - ");
            // }
            bb_id++;
        }
        function_id++;
    }
}

std::vector<uint64_t> warp_sampling(uint64_t num_of_warps, double chance) {
    uint64_t sample_size = std::ceil(num_of_warps * chance);  // Compute ceil(n/100)
    std::unordered_set<uint64_t> unique_samples;  // Set to ensure no duplicates
    
    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_of_warps - 1);

    // Keep generating random numbers until we have the desired number of unique samples
    while (unique_samples.size() < sample_size) {
        unique_samples.insert((uint64_t)dis(gen));  // Insert random number into set
    }

    // Convert set to vector for return
    std::vector<uint64_t> sampled_numbers(unique_samples.begin(), unique_samples.end());
    
    return sampled_numbers;
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    
    
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        uint64_t num_of_warps;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
            // Calculate total number of warps
            // Each block has (blockDimX * blockDimY * blockDimZ) / 32 warps
            uint64_t threads_per_block = p->config->blockDimX * p->config->blockDimY * p->config->blockDimZ;
            uint64_t warps_per_block = (threads_per_block + 31) / 32;  // Round up division
            num_of_warps = warps_per_block * p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
            // Calculate total number of warps
            uint64_t threads_per_block = p->blockDimX * p->blockDimY * p->blockDimZ;
            uint64_t warps_per_block = (threads_per_block + 31) / 32;  // Round up division
            num_of_warps = warps_per_block * p->gridDimX * p->gridDimY * p->gridDimZ;
        }

        if (!is_exit) {
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Instrument the function if needed
             * 3. Select if we want to run the instrumented or original
             * version of the kernel
             * 4. Reset the kernel instruction counter */

            pthread_mutex_lock(&mutex);

            // First pass: count total basic blocks and allocate memory if needed
            uint64_t total_basic_blocks = 0;

            std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
            related_functions.push_back(func);
            uint64_t num_functions = related_functions.size();
            for (auto f : related_functions) {
                const CFG_t &cfg = nvbit_get_CFG(ctx, f);
                total_basic_blocks += cfg.bbs.size();
            }

            // Allocate memory for counters (flattened) and offsets
            // uint64_t total_counters = total_basic_blocks * num_of_warps;
            uint64_t total_counters = total_basic_blocks;
            
            // CUDA_SAFECALL(cudaMallocManaged(&bb_counters, total_counters * sizeof(uint64_t), cudaMemAttachGlobal));
            CUDA_SAFECALL(cudaMallocManaged(&bb_counters, total_counters * sizeof(uint64_t), cudaMemAttachHost));
            CUDA_SAFECALL(cudaMemset(bb_counters, 0, total_counters * sizeof(uint64_t)));

            ptr_to_bb_counters = &bb_counters;
            
            CUDA_SAFECALL(cudaDeviceSynchronize());
            instrument_function_if_needed(ctx, func);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }

            counter = 0;
            counter_pred_off = 0;

        } else {
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());
            uint64_t kernel_instrs = counter - counter_pred_off;
            tot_app_instrs += kernel_instrs;
            int num_ctas = 0;
            uint64_t num_threads = 0;
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
                num_threads = num_ctas * p2->blockDimX * p2->blockDimY * p2->blockDimZ;
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params *p2 = (cuLaunchKernelEx_params *)params;
                num_ctas = p2->config->gridDimX * p2->config->gridDimY *
                    p2->config->gridDimZ;
                num_threads = num_ctas * p2->config->blockDimX * p2->config->blockDimY * p2->config->blockDimZ;
            }
            printf(
                "kernel %d - %s - #thread-blocks %d, #warps %lu, #threads %lu\n", 
                kernel_id, nvbit_get_func_name(ctx, func, mangled), num_ctas, 
                num_of_warps, num_threads);

            // Print final values of basic block counters
            if (bb_counters != nullptr) {
                std::vector<uint64_t> sampled_warps = warp_sampling(num_of_warps, 0.01);
                // if (verbose) {
                //     printf("\nBasic Block Counters for kernel %d:\n", kernel_id);
                //     for (const w: sampled_warps) {
                //         printf("  BBV for warp %ld: ", (uint64_t)w);
                //         for (uint64_t bb = 0; bb < total_basic_blocks; bb++) {
                //             printf("%ld ", bb_counters[bb * num_of_warps + w]);
                //         }
                //         printf("\n");
                //     }
                // }

                std::ofstream ofs(bbv_file_path, std::ios::app);
                if (!ofs.is_open()) {
                    std::cerr << "Failed to open basic_block_counters.csv" << std::endl;
                    exit(EXIT_FAILURE);
                }

                std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
                related_functions.push_back(func);
                uint64_t total_basic_blocks = 0;
                uint64_t num_functions = related_functions.size();
                for (auto f : related_functions) {
                    const CFG_t &cfg = nvbit_get_CFG(ctx, f);
                    total_basic_blocks += cfg.bbs.size();
                }

                if (kernel_id == 0)
                    ofs << "Warp ID," << "Basic Block Counts" << std::endl;
                ofs << "Kernel" << kernel_id << "," << total_basic_blocks << "," << num_of_warps << "," << nvbit_get_func_name(ctx, func, mangled) << std::endl;

                for (uint64_t bb = 0; bb < total_basic_blocks; bb++) {
                    ofs << bb_counters[bb] << ",";
                }
                ofs << std::endl;    

                // for (const w: sampled_warps) {
                //     ofs << w << ",";
                //     for (uint64_t bb = 0; bb < total_basic_blocks; bb++) {
                //         ofs << bb_counters[bb * num_of_warps + w] << ",";
                //     }
                //     ofs << std::endl;
                // }
                ofs.close();

                // Free the allocated memory using CUDA free
                CUDA_SAFECALL(cudaFree(bb_counters));
                // CUDA_SAFECALL(cudaFree(function_offsets));
                bb_counters = nullptr;
                // function_offsets = nullptr;
                total_basic_blocks = 0;
            }

            pthread_mutex_unlock(&mutex);
            kernel_id++;
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    }
    
}

void nvbit_at_term() {
    // printf("Total app instructions: %ld\n", tot_app_instrs);
}
