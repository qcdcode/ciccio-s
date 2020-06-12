#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <thread>

#include "Base.hpp"
#include "DataTypes.hpp"
#include "Threads.hpp"

namespace ciccios
{
  namespace resources
  {
    /// Holds the time at which the program has been compiled
    char PROG_COMPILE_TIME[]=__TIME__;
    /// Holds the date at which the program has been compiled
    char PROG_COMPILE_DATE[]=__DATE__;
  }
  
  /// Initialize the library and jump to f
  ///
  /// \c f will be used to initialize thread pool
  inline void initCiccios(void(*replacementMain)(int narg,char **arg),int& narg,char **&arg)
  {
    initRanks(narg,arg);
    
    printBanner();
    
    printVersionAndCompileFlags(LOGGER);
    
    possiblyWaitToAttachDebugger();
    
    cpuMemoryManager=new CPUMemoryManager;
    cpuMemoryManager->disableCache();
    
    ThreadPool::poolThread(replacementMain,narg,arg);
  }
  
  /// Finalizes
  inline void finalizeCiccios()
  {
    delete cpuMemoryManager;
    
    LOGGER<<endl<<"Ariciao!"<<endl<<endl;
    
    finalizeRanks();
  }
}

#endif
