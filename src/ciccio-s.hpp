#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

/// \file ciccio-s.hpp
///
/// \brief Main header for the library

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <Base.hpp>
#include <Expr.hpp>
#include <Gpu.hpp>
#include <DataTypes.hpp>
#include <Fields.hpp>
#include <Tensors.hpp>
#include <Threads.hpp>

/// Main namespace of the library
namespace ciccios
{
  /// Hides internal implementation not to be used directly
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
