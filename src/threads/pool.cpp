#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#define EXTERN_POOL
 #include "threads/pool.hpp"

namespace ciccios
{
  void ThreadPool::fill(const pthread_attr_t* attr)
  {
    {
      // ALLOWS_ALL_THREADS_TO_PRINT_FOR_THIS_SCOPE(runLog);
      
      LOGGER<<"Filling the thread pool with "<<nThreads<<" threads"<<endl;
      
      // Checks that the pool is not filled
      if(isFilled)
	CRASHER<<"Cannot fill again the pool!"<<endl;
      
      // Resize the pool to contain all threads
      pool.resize(nThreads,0);
      
      // Marks the pool as filled, even if we are still filling it, this will keep the threads swimming
      isFilled=true;
      
      for(int threadId=1;threadId<nThreads;threadId++)
	{
	  //runLog()<<"thread of id "<<threadId<<" spwawned\n";
	  
	  // Allocates the parameters of the thread
	  ThreadPars* pars=
	    new ThreadPars{this,threadId};
	  
	  if(pthread_create(&pool[threadId],attr,threadPoolSwim,pars)!=0)
	    CRASHER<<"creating the thread "<<threadId<<endl;
	}
      
      waitPoolToBeFilled(masterThreadId);
    }
    
    // Marks the pool is waiting for job to be done
    isWaitingForWork=true;
  }
}
