from cuda import cuda
import sqlite3
import pandas as pd
import numpy as np
import sys
import os
import math
import argparse
import csv
import copy

import matplotlib.pyplot as plt

current_path = os.getcwd()
sqlite_file_path = f'{current_path}/data/sqlite'
sqlite_files = os.listdir(sqlite_file_path)

for file in sqlite_files:
    conn = sqlite3.connect(f'{sqlite_file_path}/{file}')
    cur = conn.cursor()
    try:
        cur.executescript("""
            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD "duration[ns]" TEXT;
            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD kernelName TEXT;

            UPDATE CUPTI_ACTIVITY_KIND_KERNEL
            SET "duration[ns]" = end - start;

            UPDATE CUPTI_ACTIVITY_KIND_KERNEL
            SET kernelName = (
                SELECT StringIds.value AS shortName
                FROM StringIds
                WHERE CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id
            )
            WHERE CUPTI_ACTIVITY_KIND_KERNEL.gridId IN (
                SELECT CUPTI_ACTIVITY_KIND_KERNEL.gridId
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.shortName = StringIds.id);


            ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN nvtxRange TEXT;

            CREATE INDEX nvtx_start ON NVTX_EVENTS (start);

            UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET nvtxRange = (
                SELECT NVTX_EVENTS.text
                FROM NVTX_EVENTS JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON
                    NVTX_EVENTS.eventType == 59 AND
                    NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AND
                    NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start AND
                    NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end
                WHERE
                    CUPTI_ACTIVITY_KIND_KERNEL.correlationId == CUPTI_ACTIVITY_KIND_RUNTIME.correlationId
                ORDER BY NVTX_EVENTS.start DESC LIMIT 1
            );     
        """)
        
    except sqlite3.Error as err:
        if "duplicate" in ' '.join(err.args):
            pass
        else:
            print(' '.join(err.args))
            
    df = pd.read_sql("""
            SELECT kernelName, "duration[ns]", nvtxRange, gridX, gridY, gridZ, blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory, registersPerThread 
            FROM CUPTI_ACTIVITY_KIND_KERNEL;                     
        """, conn)
    
    df = df.astype({'duration[ns]': 'float'})
    df = df.rename(columns={'duration[ns]': 'duration[us]'})
    df['duration[us]'] = df['duration[us]']/1000
    
    conn.close()
    
    err, = cuda.cuInit(0)
    err, cuDevice = cuda.cuDeviceGet(0)
    err, context = cuda.cuCtxCreate(0, cuDevice)

    _, hw_num_sm = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice)
    _, hw_threads_per_warp = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice)
    _, hw_max_block_per_sm = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, cuDevice)
    _, hw_max_threads_per_sm = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice)
    _, hw_max_register_per_sm = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cuDevice)
    _, hw_max_register_per_block = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, cuDevice)
    hw_max_warps_per_sm = hw_max_threads_per_sm / hw_threads_per_warp
    hw_max_register_per_thread = 255
    hw_warp_alloc_granularity = 4
    hw_register_alloc_unit_size = 256
    hw_shmem_per_sm_config = 0

    num_blocks = (df['gridX'] * df['gridY'] * df['gridZ']).to_numpy()
    num_threads_per_block = (df['blockX'] * df['blockY'] * df['blockZ']).to_numpy()
    block_limit_warps = hw_max_threads_per_sm // num_threads_per_block

    block_limit_register = np.array([hw_max_block_per_sm] * len(df))
    register_per_threads = df['registersPerThread'].to_numpy()
    block_limit_register[register_per_threads > 0] = np.floor(np.floor(hw_max_register_per_block / (np.ceil(register_per_threads[register_per_threads > 0] * hw_threads_per_warp / hw_register_alloc_unit_size) * hw_register_alloc_unit_size) / hw_warp_alloc_granularity) * hw_warp_alloc_granularity / (hw_max_warps_per_sm / block_limit_warps))
    block_limit_register[register_per_threads > hw_max_register_per_thread] = 0

    total_shared_mem = df['staticSharedMemory'] + df['dynamicSharedMemory']
    block_limit_shmem = np.array([hw_max_block_per_sm] * len(df))
    if hw_shmem_per_sm_config > 0:
        block_limit_shmem[total_shared_mem > 0] = min(hw_shmem_per_sm_config // total_shared_mem[total_shared_mem > 0], hw_max_block_per_sm)

    block_limit_per_sm = np.minimum(np.minimum(block_limit_warps, block_limit_register), block_limit_shmem)

    df['wave'] = num_blocks / (hw_num_sm * block_limit_per_sm)
    
    df = df[['kernelName', 'nvtxRange', 'duration[us]', 'wave']]
    df[['kernelType']] = None
    
    spin_kernel = df['kernelName'].str.contains('spin')
    spin_kernel_idx = df[spin_kernel].index
    df = df[spin_kernel_idx[0]+1:]
    df =df.reset_index(drop=True)
    
    same_nvtx_list = []
    main_kernel_list = ['Winograd', 'scudnn', 'gemm', 'wgrad', 'dgrad', 'bn', 'scalePackedTensor', 'max_pool', 'launch_clamp_scalar', 'hardswish', 'hardsigmoid']
    for index, row in df.iloc[:].iterrows():
        if len(same_nvtx_list) == 0:
            same_nvtx_list.append(index)
        else:
            if df.loc[same_nvtx_list[-1], 'nvtxRange'] == df.loc[index, 'nvtxRange']:
                same_nvtx_list.append(index)
            else:
                if len(same_nvtx_list) == 1:
                    df.loc[same_nvtx_list[0], 'kernelType'] = 'main'
                else:
                    for j in same_nvtx_list:
                        df.loc[j,'kernelType'] = 'sub'
                        for keyword in main_kernel_list:
                            if keyword in df.loc[j,'kernelName']:
                                df.loc[j,'kernelType'] = 'main'
                same_nvtx_list = []
                same_nvtx_list.append(index)
    df.loc[df['kernelName'].str.contains('|'.join(['OnSelf', 'Offset'])), 'kernelType'] = 'sub'
                
    df.to_csv(f"{sqlite_file_path}/../csv/{file[:-7]}.csv", index=False)
    # import pdb; pdb.set_trace()