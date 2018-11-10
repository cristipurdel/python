import cpuinfo

import math
import numba
import os
import pdb
import pkg_resources
import time

import tensorflow as tf

try:
    import cupy as cp
except ImportError:
    cupy = None
import numpy as np


import matplotlib
import matplotlib.pyplot as plt

from numba import guvectorize, int32, float32, cuda, float64, void, jit, njit, prange, threading_layer
from timeit import default_timer as timer

cpuID = cpuinfo.get_cpu_info()['brand']
print(cpuID)
cpuID = cpuID[:30]
cudaID = ''

''' MIGHT NEED SMTH EXTRA DEPENDING ON CUDATOOLKIT'''
if numba.cuda.is_available() == True:
    booCUDA = True
    print (numba.cuda.detect())
    cudaID =  str(numba.cuda.get_current_device().name)
    cudaID = cudaID[2:-1]
    RunTime = {'cp_cuda':[], 'np_cpuMT':[], '/cpu:0':[], '/gpu:0':[]} 
    GFLOPS = {'cp_cuda':[], 'np_cpuMT':[], '/cpu:0':[], '/gpu:0':[]}    
    targetOpt = 'cuda'
#    targetOpt = 'parallel'  
#    targetOpt = 'cpu'
else:
    booCUDA = False
    print ('No CUDA detected, parallel will be used')
    RunTime = {'np_cpuMT':[], '/cpu:0':[]} 
    GFLOPS = {'np_cpuMT':[], '/cpu:0':[]} 
    targetOpt = 'parallel'
#    targetOpt = 'cpu'



step = 500
count = 14
npdtype = np.float32
tfdtype = tf.float32

config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

print('A X = B for variable size N using float32')
matrix_sizes = range(step, step * count + 1, step)
for N in matrix_sizes:
    
    print('N = ' + str(N))
    N_ops = N**3

    
        
    ''' numpy cpuMT'''
    if True:    
        tR = timer()
        A_np = np.random.randn(N**2).reshape(N,N).astype(npdtype)
        B_np = np.random.randn(N).astype(npdtype)           
        tG = timer()
        X_np = np.linalg.tensorsolve(A_np, B_np)
        tG = timer() - tG
        tG = round(N_ops/tG/10**9, 1)
        tR = round(timer() - tR, 3)
        
        GFLOPS['np_cpuMT'].append(tG)
        RunTime['np_cpuMT'].append(tR)

#        if np.allclose(np.matmul(A_np,X_np), B_np, rtol=0, atol=1e-03, equal_nan=False) == False:
#            print(' numpy cpuMT is False for N =  ' + str(N) )



    ''' tensorflow cpu'''
    if True:    
        tR = timer()        
        with tf.device('/cpu:0'):
            A_tf_cpu = tf.random_uniform(shape=(N,N), minval=0, maxval=1, dtype=tfdtype)
            B_tf_cpu = tf.random_uniform(shape=(N,1), minval=0, maxval=1, dtype=tfdtype)
            tf_op_cpu = tf.linalg.solve(A_tf_cpu, B_tf_cpu, adjoint=False, name=None)

        with tf.Session(config=config) as session:
            tG = timer()
            X_tf_cpu = session.run(tf_op_cpu)
            tG = timer() - tG
            tG = round(N_ops/tG/10**9, 1)
            session.close()
            tR = round(timer() - tR, 3)
            
            GFLOPS['/cpu:0'].append(tG)
            RunTime['/cpu:0'].append(tR)



    ''' tensorflow gpu'''
    if targetOpt == 'cuda': 
        tR = timer()        
        with tf.device('/device:GPU:0'):
            A_tf_gpu = tf.random_uniform(shape=(N,N), minval=0, maxval=1, dtype=tfdtype)
            B_tf_gpu = tf.random_uniform(shape=(N,1), minval=0, maxval=1, dtype=tfdtype)
            tf_op_gpu = tf.linalg.solve(A_tf_gpu, B_tf_gpu, adjoint=False, name=None)

        with tf.Session(config=config) as session:
            tG = timer()
            X_tf_gpu = session.run(tf_op_gpu)
            tG = timer() - tG
            tG = round(N_ops/tG/10**9, 1)
            session.close()
            tR = round(timer() - tR, 3)
            
            GFLOPS['/gpu:0'].append(tG)
            RunTime['/gpu:0'].append(tR)                


    
    ''' cupy cuda'''    
    if targetOpt == 'cuda':

        tR = timer()       
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()       
        A_cp = cp.random.randn(N**2).reshape(N,N).astype(npdtype)
        B_cp = cp.random.randn(N).astype(npdtype)       
        tG = timer()        
        X_cp = cp.linalg.tensorsolve(A_cp, B_cp, axes=None)
        tG = timer() - tG
        tG = round(N_ops/tG/10**9, 1)      
        A_cp = None
        B_cp = None
        X_cp = None  
        mempool.free_all_blocks()
        tR = round(timer() - tR, 3)
        
        GFLOPS['cp_cuda'].append(tG)
        RunTime['cp_cuda'].append(tR)
        
       



''' Times '''
np_RunTime = RunTime['np_cpuMT']
plt.plot(matrix_sizes[:len(np_RunTime)], np_RunTime, 'bo-') 
tf_cpu_RunTime = RunTime['/cpu:0']
plt.plot(matrix_sizes[:len(tf_cpu_RunTime)], tf_cpu_RunTime, 'bo--') 
if numba.cuda.is_available():
    cp_RunTime = RunTime['cp_cuda']
    plt.plot(matrix_sizes[:len(cp_RunTime)], cp_RunTime, 'r^-')
    tf_gpu_RunTime = RunTime['/gpu:0']
    plt.plot(matrix_sizes[:len(tf_gpu_RunTime)], tf_gpu_RunTime, 'r^--')    
plt.title('RunTime vs Matrix size using float32')
plt.legend(('np ' + cpuID, 'tf ' + cpuID, 'cp ' + cudaID, 'tf ' + cudaID), loc='upper left')
plt.ylabel('Time')
plt.xlabel('Matrix size')
plt.show()



''' GFLOPS '''
np_GFLOPS = GFLOPS['np_cpuMT']
plt.plot(matrix_sizes[:len(np_GFLOPS)], np_GFLOPS, 'bo-')  
tf_cpu_GFLOPS = GFLOPS['/cpu:0']
plt.plot(matrix_sizes[:len(tf_cpu_GFLOPS)], tf_cpu_GFLOPS, 'bo--') 
if numba.cuda.is_available():
    cp_GFLOPS = GFLOPS['cp_cuda']
    plt.plot(matrix_sizes[:len(cp_GFLOPS)], cp_GFLOPS, 'r^-') 
    tf_gpu_GFLOPS = GFLOPS['/gpu:0']
    plt.plot(matrix_sizes[:len(tf_gpu_GFLOPS)], tf_gpu_GFLOPS, 'r^--')     
    
plt.title('GFLOPS vs Matrix size using float32')
plt.legend(('np ' + cpuID, 'tf ' + cpuID, 'cp ' + cudaID, 'tf ' + cudaID), loc='upper left')
plt.ylabel('GFLOPS')
plt.xlabel('Matrix size')
plt.show()
