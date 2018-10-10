import numba
import os
import timeit

import numpy as np
import pkg_resources

from numba import vectorize, guvectorize, float32, void, cuda, njit, prange


try:
	print('NUMBA NUM THREADS = ' + str(os.environ['NUMBA_NUM_THREADS']))
except:
	print('NUMBA NUM THREADS = ' + str(numba.config.NUMBA_DEFAULT_NUM_THREADS))

print('numba version used: ' + str(pkg_resources.get_distribution("numba").version))
print('numpy version used: ' + str(pkg_resources.get_distribution("numpy").version))



import cpuinfo
print(cpuinfo.get_cpu_info()['brand'])

if cuda.is_available():  
    cuda.detect()


count = 4*256




# for numpy
def npy_add(a, b):  
    c = 0    
    for i in range(count):
        c += np.add(a,b)
    return c

# for vectorize
def add_array0D0D_return(a, b):
    c = float32(0.)
    for i in range(count):
        c += float32(a + b)
    return c

#""" DOES NOT WORK with vectors for vectorize"""
## for vectorize
#def add_array1D0D_return(a, b):
#    c = 0.
#    for i in range(a.shape[0]):
#        for l in range(count):
#            c[i] += float32(a[i] + b)
#    return c
    


# for guvectorize
def add_array0D0D_noreturn(a, b, c):
    for i in range(count):
        c += float32(a + b)

# for guvectorize without prange
def add_array1D1D_noreturn(A, B, out):            
    for i in range(out.shape[0]):
        for c in range(count):
            out[i] += float32(A[i] + B[i])

# for guvectorize without prange
def add_array2D2D_noreturn(A, B, out):            
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for c in range(count):
                out[i,j] += float32(A[i,j] + B[i,j])
  



def printbreak():
    print('!!!!!!!!!!!   FAIL CHECKSUM   !!!!!!!!!!!!! ')
#    stop



if True:

    
    add_c11_vec_cpu = vectorize(['float32(float32, float32)'], target='cpu')(add_array0D0D_return) 
#    add_c12_vec_parallel = vectorize(['float32(float32, float32)'], target='parallel')(add_array0D0D_return) 

    add_c14_guv_cpu = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cpu')(add_array1D1D_noreturn) 
    add_c15_guv_parallel = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='parallel')(add_array1D1D_noreturn) 



    add_c22_vec_parallel = vectorize(['float32(float32, float32)'], target='parallel')(add_array0D0D_return) 
    add_c24_r1_guv_cpu = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cpu')(add_array1D1D_noreturn) 
    add_c25_r1_guv_parallel = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='parallel')(add_array1D1D_noreturn) 

    add_c34_r1_guv_cpu = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cpu')(add_array1D1D_noreturn) 
    add_c34_r2_guv_cpu = guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(m,m),(m,m)->(m,m)', target='cpu')(add_array2D2D_noreturn) 
#    add_c35_r0_guv_parallel = guvectorize([(float32, float32, float32)], '(),()->()', target='parallel')(add_array0D0D_noreturn) 
#    add_c35_r1_guv_parallel = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='parallel')(add_array1D1D_noreturn) 
    add_c35_r2_guv_parallel = guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(m,m),(m,m)->(m,m)', target='parallel')(add_array2D2D_noreturn) 

        
if numba.cuda.is_available():
#
    add_c13_vec_cuda = vectorize(['float32(float32, float32)'], target='cuda')(add_array0D0D_return) 
    add_c16_guv_cuda = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cuda')(add_array1D1D_noreturn) 
##    add_c6_guv_cuda = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cuda')(add_array1D_noreturn_cuda) 
#    add_c23_vec_cuda = vectorize(['float32(float32, float32)'], target='cuda')(add_array0D0D_return) 
    add_c26_r1_guv_cuda = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cuda')(add_array1D1D_noreturn) 
    add_c36_r1_guv_cuda = guvectorize([(float32[:], float32[:], float32[:])], '(m),(m)->(m)', target='cuda')(add_array1D1D_noreturn) 
    add_c36_r2_guv_cuda = guvectorize([(float32[:,:], float32[:,:], float32[:,:])], '(m,m),(m,m)->(m,m)', target='cuda')(add_array2D2D_noreturn) 



    
    
for N in [16, ]:
   


    x = N * 1024
    y = 32
    z = 32
    N1 = x * y * z
    N2 = int(N1**0.5)


    M123 = np.arange(24).reshape((2, 3, 4))
    M12 = M123.reshape((6,4))
    M123b = M12.reshape((2,3,4))
    M12b = M123b.reshape((6,4))
    M123c = M12.reshape((3,2,4))
    

    

    blockdim = 1024
    griddim = int(np.ceil(N / blockdim))

    B0 = float32(0.)

    A1 = np.ones(N1).astype(np.float32)
    B1 = np.ones(N1).astype(np.float32)  
    
    A2 = np.ones((N2,N2)).astype(np.float32)
    B2 = np.ones((N2,N2)).astype(np.float32)    

    A3 = np.ones((x,y,z)).astype(np.float32)
    B3 = np.ones((x,y,z)).astype(np.float32)   

    
    print('############### size of one matrix is ' + str(round(A1.nbytes / 2 ** 20,0)) + ' MBs ######################')



    if True:    

   
        t_start = timeit.default_timer()
        out_c10_npy_cpu = npy_add(A1, B1)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('npy_ cpu      1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
        if np.sum(out_c10_npy_cpu) != 2*N1*count: printbreak()
        
        out_c11_vec_cpu = np.zeros(N1).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c11_vec_cpu = add_c11_vec_cpu(A1, B1)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('vec_ cpu      1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
        if np.sum(out_c11_vec_cpu) != 2*N1*count: printbreak()

#        """ DOES NOT WORK """
#        out_c12_vec_parallel = np.zeros(N1).astype(np.float32) 
#        t_start = timeit.default_timer()
#        out_c12_vec_parallel = add_c12_vec_parallel(A1, B1)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('vec_ parallel 1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
#        if np.sum(out_c12_vec_parallel) != 2*N1*count: printbreak()

        out_c14_guv_cpu = np.zeros(N1).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c14_guv_cpu = add_c14_guv_cpu(A1, B1, out_c14_guv_cpu)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cpu      1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
        if np.sum(out_c14_guv_cpu) != 2*N1*count: printbreak()

        out_c15_guv_parallel = np.zeros(N1).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c15_guv_parallel = add_c15_guv_parallel(A1, B1, out_c15_guv_parallel)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ parallel 1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
        if np.sum(out_c15_guv_parallel) != 2*N1*count: printbreak()       
        

        t_start = timeit.default_timer()
        out_c20_npy_cpu = npy_add(A2, B2)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('npy_ cpu      2D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c20_npy_cpu) != 2*N2*N2*count: printbreak()

        t_start = timeit.default_timer()
        out_c21_vec_cpu = add_c11_vec_cpu(A2, B2)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('vec_ cpu      2D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c21_vec_cpu) != 2*N2*N2*count: printbreak()       

###        """ DOES NOT WORK """
#        t_start = timeit.default_timer()
#        out_c22_vec_parallel = add_c22_vec_parallel(A2, B2)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('vec_ parallel 2D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
#        if np.sum(out_c22_vec_parallel) != 2*N2*N2*count: printbreak()       

        out_c24_r1_guv_cpu = np.zeros((N2,N2)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c24_r1_guv_cpu = add_c24_r1_guv_cpu(A2, B2, out_c24_r1_guv_cpu)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cpu      2D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c24_r1_guv_cpu) != 2*N2*N2*count: printbreak()

        out_c25_r1_guv_parallel = np.zeros((N2,N2)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c25_r1_guv_parallel = add_c25_r1_guv_parallel(A2, B2, out_c25_r1_guv_parallel)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ parallel 2D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c25_r1_guv_parallel) != 2*N2*N2*count: printbreak()
        

###        """ DOES NOT WORK """
#        t_start = timeit.default_timer()
#        out_c30_npy_cpu = npy_add(A3, B3)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('npy_ cpu      3D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
#        if np.sum(out_c30_npy_cpu) != 2*x*y*z*count: printbreak()
        
        t_start = timeit.default_timer()
        out_c31_vec_cpu = add_c11_vec_cpu(A3, A3)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('vec_ cpu      3D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c31_vec_cpu) != 2*x*y*z*count: printbreak()   
#        
###        """ DOES NOT WORK """
#        t_start = timeit.default_timer()
#        out_c32_vec_parallel = add_c22_vec_parallel(A3, B3)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('vec_ parallel 3D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
#        if np.sum(out_c32_vec_parallel) != 2*x*y*z*count: printbreak()    


        out_c34_r1_guv_cpu = np.zeros((x,y,z)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c34_r1_guv_cpu = add_c34_r1_guv_cpu(A3, B3, out_c34_r1_guv_cpu)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cpu      3D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c34_r1_guv_cpu) != 2*x*y*z*count: printbreak()

        out_c34_r2_guv_cpu = np.zeros((x,y,z)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c34_r2_guv_cpu = add_c34_r2_guv_cpu(A3, B3, out_c34_r2_guv_cpu)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cpu      3D r2    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c34_r2_guv_cpu) != 2*x*y*z*count: printbreak()

###        """ DOES NOT WORK """
#        out_c35_r1_guv_parallel = np.zeros((x,y,z)).astype(np.float32) 
#        t_start = timeit.default_timer()
#        out_c35_r1_guv_parallel = add_c35_r1_guv_parallel(A3, B3, out_c35_r1_guv_parallel)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('guv_ parallel 3D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
#        if np.sum(out_c35_r1_guv_parallel) != 2*x*y*z*count: printbreak()

        out_c35_r2_guv_parallel = np.zeros((x,y,z)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c35_r2_guv_parallel = add_c35_r2_guv_parallel(A3, B3, out_c35_r2_guv_parallel)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ parallel 3D r2    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c35_r2_guv_parallel) != 2*x*y*z*count: printbreak()

    if numba.cuda.is_available():  

        t_start = timeit.default_timer()
        out_c13_vec_cuda = add_c13_vec_cuda(A1, B1)
        t_end = round(timeit.default_timer() - t_start,3)
        print('vec_ cuda     1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
        if np.sum(out_c13_vec_cuda) != 2*N1*count: printbreak()

###        """ TOO SLOW """ 
#        out_c16_guv_cuda = np.zeros(N1).astype(np.float32) 
#        t_start = timeit.default_timer()
#        out_c16_guv_cuda = add_c16_guv_cuda(A1, B1, out_c16_guv_cuda)
#        t_end = round(timeit.default_timer() - t_start + 0.001,3)
#        print('guv_ cuda     1D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N1*count/t_end/10**9,1)))
#        if np.sum(out_c16_guv_cuda) != 2*N1*count: printbreak()    

      
 
        t_start = timeit.default_timer()
        out_c23_vec_cuda = add_c13_vec_cuda(A2, B2)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('vec_ cuda     2D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c23_vec_cuda) != 2*N2*N2*count: printbreak()         

        out_c26_r1_guv_cuda = np.zeros((N2,N2)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c26_r1_guv_cuda = add_c26_r1_guv_cuda(A2, B2, out_c26_r1_guv_cuda)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cuda     2D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(N2*N2*count/t_end/10**9,1)))
        if np.sum(out_c26_r1_guv_cuda) != 2*N2*N2*count: printbreak()



        t_start = timeit.default_timer()
        out_c33_vec_cuda = add_c13_vec_cuda(A3, A3)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('vec_ cuda     3D       took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c33_vec_cuda) != 2*x*y*z*count: printbreak()  

        out_c36_r1_guv_cuda = np.zeros((x,y,z)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c36_r1_guv_cuda = add_c36_r1_guv_cuda(A3, B3, out_c36_r1_guv_cuda)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cuda     3D r1    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c36_r1_guv_cuda) != 2*x*y*z*count: printbreak()

        out_c36_r2_guv_cuda = np.zeros((x,y,z)).astype(np.float32) 
        t_start = timeit.default_timer()
        out_c36_r2_guv_cuda = add_c36_r2_guv_cuda(A3, B3, out_c36_r2_guv_cuda)
        t_end = round(timeit.default_timer() - t_start + 0.001,3)
        print('guv_ cuda     3D r2    took ' + str(t_end) + ' seconds, GFLOPS = ' + str(round(x*y*z*count/t_end/10**9,1)))
        if np.sum(out_c36_r2_guv_cuda) != 2*x*y*z*count: printbreak()