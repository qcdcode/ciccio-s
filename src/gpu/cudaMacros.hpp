#ifndef _CUDAMACROS_HPP
#define _CUDAMACROS_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#ifdef USE_CUDA

 /// DEVICE is actually the cuda attribute
 #define DEVICE __device__
 
 /// GLOBAL is actually the cuda attribute
 #define GLOBAL __global__
 
 /// HOST is actually the cuda attribute
 #define HOST __host__
 
#else
 
 /// DEVICE is a dummy macro
 #define DEVICE
 
 /// HOST is a dummy macro
 #define HOST
 
 /// GLOBAL is a dummy macro
 #define GLOBAL
 
#endif

#ifdef __CUDA_ARCH__
 
 /// A convenient macro to detect that we are compiling on device
 #define COMPILING_FOR_DEVICE
 
#endif

#endif
