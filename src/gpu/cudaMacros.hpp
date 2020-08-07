#ifndef _CUDAMACROS_HPP
#define _CUDAMACROS_HPP

/// \file cudaMacros.hpp
///
/// \brief Change CUDA_DEVICE and similars in terms of whether we compile
/// with nvcc or not

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#ifdef USE_CUDA

 /// CUDA_DEVICE is actually the cuda attribute
# define CUDA_DEVICE __device__
 
 /// CUDA_GLOBAL is actually the cuda attribute
# define CUDA_GLOBAL __global__
 
 /// CUDA_HOST is actually the cuda attribute
# define CUDA_HOST __host__
 
#else
 
 /// CUDA_DEVICE is a dummy macro
# define CUDA_DEVICE
 
 /// CUDA_HOST is a dummy macro
# define CUDA_HOST
 
 /// CUDA_GLOBAL is a dummy macro
# define CUDA_GLOBAL
 
#endif

/// Put together CUDA_HOST and CUDA_DEVICE
#define CUDA_HOST_DEVICE CUDA_HOST CUDA_DEVICE

#ifdef __CUDA_ARCH__
 
 /// A convenient macro to detect that we are compiling on device
# define COMPILING_FOR_DEVICE
 
#endif

/// True or false depending on whether we are compiling on device
[[ maybe_unused ]]
constexpr bool CompilingForDevice=
#ifdef COMPILING_FOR_DEVICE
	    true
#else
	    false
#endif
	    ;
 


#endif
