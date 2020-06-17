#ifndef _CUDA_HPP
#define _CUDA_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#ifndef EXTERN_CUDA
 #define EXTERN_CUDA extern
#define INIT_CUDA_TO(...)
#else
 #define INIT_CUDA_TO(...) (__VA_ARGS__)
#endif

#include <base/inliner.hpp>
#include <base/logger.hpp>

namespace ciccios
{
#ifdef USE_CUDA
 #define DEVICE __device__
 #define GLOBAL __global__
 #define HOST __host__
#else
 #define DEVICE
 #define HOST
 #define GLOBAL
#endif
  
#ifdef __CUDA_ARCH__
  #define COMPILING_FOR_DEVICE
#endif
  
#ifdef USE_CUDA
  namespace Gpu
  {
    /// Number of threads for cuda, to be generalized
    EXTERN_CUDA int nCudaThreads INIT_CUDA_TO(128);
    
    /// Body of a generic kernel which runs the passed function
    template <typename IMin,
	      typename IMax,
	      typename F>
    GLOBAL
    INLINE_FUNCTION
    void cudaGenericKernel(const IMin min,
			   const IMax max,
			   F f)
    {
      const auto i=min+blockIdx.x*blockDim.x+threadIdx.x;
      
      if(i<max)
	f(i);
    }
    
    /// Imposes a barrier on the kernel
    INLINE_FUNCTION
    void cudaBarrier()
    {
      cudaDeviceSynchronize();
    }
    
    /// Issue a kernel on the basis of the passed range and function
    template <typename IMin,
	      typename IMax,
	      typename F>
    INLINE_FUNCTION
    void cudaParallel(const IMin min,
		      const IMax max,
		      F f)
    {
      const auto length=(max-min);
      const dim3 block_dimension(nCudaThreads);
      const dim3 grid_dimension((length+block_dimension.x-1)/block_dimension.x);
      
      VERB_LOGGER(2)<<"launching kernel on loop ["<<min<<","<<max<<") using blocks of size "<<block_dimension.x<<" and grid of size "<<grid_dimension.x<<endl;
      
      cudaGenericKernel<<<grid_dimension,block_dimension>>>(min,max,std::forward<F>(f));
      cudaBarrier();
      VERB_LOGGER(2)<<" finished"<<endl;
    }
  }
#endif
  
  void initCuda();
}

#endif

