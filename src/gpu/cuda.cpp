#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

/// \file cuda.cpp
///
/// \brief Implements the main initialization of cuda

#define EXTERN_CUDA
 #include <gpu/cuda.hpp>

#include <base/debug.hpp>

namespace ciccios
{
  void initCuda()
  {
#ifdef USE_CUDA
    int nDevices;
    if(cudaGetDeviceCount(&nDevices)!=cudaSuccess)
      CRASHER<<"no CUDA enabled device"<<endl;
    
    LOGGER<<"Number of CUDA enabled devices: "<<nDevices<<endl;
    for(int i=0;i<nDevices;i++)
      {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp,i);
	LOGGER<<" CUDA Enabled device "<<i<<"/"<<nDevices<<": "<<deviceProp.major<<"."<<deviceProp.minor<<endl;
      }
    //assumes that if we are seeing multiple gpus, there are nDevices ranks to attach to each of it
    if(nDevices!=1)
      {
	const int iCudaDevice=rank()%nDevices;
	DECRYPT_CUDA_ERROR(cudaSetDevice(iCudaDevice),"Unable to set device %d",iCudaDevice);
      }
#endif
  }
}
