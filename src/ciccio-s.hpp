#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "base.hpp"
#include "dataTypes.hpp"

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
  template <typename F>
  void initCiccios(int& narg,char **&arg,const F& inMain)
  {
    initRanks(narg,arg);
    
    printBanner();
    
    printVersionAndCompileFlags(LOGGER);
    
    possiblyWaitToAttachDebugger();
    
    cpuMemoryManager=new CPUMemoryManager;
    cpuMemoryManager->disableCache();
    
    inMain();
  }
  
  /// Finalizes
  void finalizeCiccios()
  {
    delete cpuMemoryManager;
    
    LOGGER<<endl<<"Ariciao!"<<endl<<endl;
    
    finalizeRanks();
  }
}

#endif
