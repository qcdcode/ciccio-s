#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

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
  template <typename F>
  void initCiccios(int& narg,char **&arg,const F& inMain)
  {
    initRanks(narg,arg);
    
    printBanner();
    
    printVersionAndCompileFlags(LOGGER);
    
    possiblyWaitToAttachDebugger();
    
    cpuMemoryManager=new CPUMemoryManager;
    cpuMemoryManager->disableCache();
    
    /// Tag to be used for setting nThreads
    const char* numThreadsTag="CICCIOS_NUM_THREADS";
    
    /// Capture environment variable
    const char* nThreadsStr=getenv(numThreadsTag);
    
    /// Try to convert from environment variable
    const int nEnvThreads=(nThreadsStr!=nullptr)?atoi(nThreadsStr):-1;
    
    /// Get the hardware number of threads
    const int nHwThreads=std::thread::hardware_concurrency();
    
    /// Decide whether to use hardware number of threads
    const bool useEnvThreads=nEnvThreads>0 and nEnvThreads<nHwThreads;
    
    /// Set nThreads according to source
    int nThreads=useEnvThreads?nEnvThreads:nHwThreads;
    LOGGER<<"Using "<<(useEnvThreads?"environment ":"hardware ")<<"number of threads, "<<nThreads<<endl;
    
    threadPool=new ThreadPool(nThreads);
    
    inMain();
  }
  
  /// Finalizes
  void finalizeCiccios()
  {
    delete threadPool;
    
    delete cpuMemoryManager;
    
    LOGGER<<endl<<"Ariciao!"<<endl<<endl;
    
    finalizeRanks();
  }
}

#endif
