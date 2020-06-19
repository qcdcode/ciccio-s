#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <Base.hpp>
#include <Gpu.hpp>
#include <DataTypes.hpp>
#include <Threads.hpp>

namespace ciccios
{
  namespace resources
  {
    /// Holds the time at which the program has been compiled
    char PROG_COMPILE_TIME[]=__TIME__;
    /// Holds the date at which the program has been compiled
    char PROG_COMPILE_DATE[]=__DATE__;
  }
  
  /// Initialize the library
  template <typename F>
  inline void initCiccios(F&& f,int& narg,char **&arg)
  {
    initRanks(narg,arg);
    
    printBanner();
    
    readAllFlags();
    
    printVersionAndCompileFlags(LOGGER);
    
    possiblyWaitToAttachDebugger();
    
    cpuMemoryManager=new CPUMemoryManager;
    //cpuMemoryManager->disableCache();
    
    initCuda();
    
#ifdef USE_CUDA
    gpuMemoryManager=new GPUMemoryManager;
#endif
    
    ThreadPool::poolStart(f,narg,arg);
  }
  
  /// Finalizes
  inline void finalizeCiccios()
  {
    ThreadPool::poolStop();
    
    delete cpuMemoryManager;
    
#ifdef USE_CUDA
    delete gpuMemoryManager;
#endif
    
    LOGGER<<endl<<"Ariciao!"<<endl<<endl;
    
    finalizeRanks();
  }
}

#endif
