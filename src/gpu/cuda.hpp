#ifndef _CUDA_HPP
#define _CUDA_HPP

/// \file cuda.hpp
///
/// \brief Implements an intermediate layout in front of cuda

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#ifndef EXTERN_CUDA

 /// Make external if put in front of a variable
 ///
 /// Actual allocation is done in the cpp file
# define EXTERN_CUDA extern
# define INIT_CUDA_TO(...)

#else

# define INIT_CUDA_TO(...) (__VA_ARGS__)

#endif

#include <base/inliner.hpp>
#include <base/logger.hpp>

#include <gpu/cudaMacros.hpp>

namespace ciccios
{
#ifdef USE_CUDA
  /// Internal implementation of GPU functionalities
  namespace Gpu
  {
    /// Number of threads for cuda, to be generalized
    EXTERN_CUDA int nCudaThreads INIT_CUDA_TO(64);
    
    /// Body of a generic kernel which runs the passed function
    template <typename IMin,
	      typename IMax,
	      typename F>
    CUDA_GLOBAL
    void cudaGenericKernel(const IMin min,
			   const IMax max,
			   F f)
    {
      /// Loop iteration
      const auto i=
	min+blockIdx.x*blockDim.x+threadIdx.x;
      
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
		      F&& f)
    {
      /// Length of the loop
      const auto length=
	max-min;
      
      /// Dimension of the block
      const dim3 blockDimension(nCudaThreads);
      
      /// Dimension of the grid
      const dim3 gridDimension((length+blockDimension.x-1)/blockDimension.x);
      
      VERB_LOGGER(2)<<"launching kernel on loop ["<<min<<","<<max<<") using blocks of size "<<blockDimension.x<<" and grid of size "<<gridDimension.x<<endl;
      
      cudaGenericKernel<<<gridDimension,blockDimension>>>(min,max,std::forward<F>(f));
      cudaBarrier();
      VERB_LOGGER(2)<<" finished"<<endl;
    }
  }
#endif
  
  void initCuda();
}

#endif

