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
  inline void initCiccios(int& narg,char **&arg)
  {
    initRanks(narg,arg);
    
    printBanner();
    
    printVersionAndCompileFlags(LOGGER);
    
    possiblyWaitToAttachDebugger();
    
    cpuMemoryManager=new CPUMemoryManager;
    //cpuMemoryManager->disableCache();
    
    ThreadPool::poolStart();
  }
  
  /// Finalizes
  inline void finalizeCiccios()
  {
    ThreadPool::poolStop();
    
    delete cpuMemoryManager;
    
    LOGGER<<endl<<"Ariciao!"<<endl<<endl;
    
    finalizeRanks();
  }
}

#endif
