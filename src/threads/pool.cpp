#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

/// \file pool.cpp
///
/// \brief Implements the start and stop, and pool loop

#define EXTERN_POOL
 #include "threads/pool.hpp"

namespace ciccios
{
  namespace ThreadPool
  {
    void* poolWorkerLoop(void* _pars)
    {
      /// Decrypt the pars
      int* pars=
	static_cast<int*>(_pars);
      
      /// Gets the thread id from the pars
      const int threadId=
	*pars;
      
      // Delete the pars, which have been passed as new int
      delete pars;
      
      do
	{
	  resources::waitForWork();
	  
	  work(threadId);
	}
      while(poolIsStarted);
      
      return nullptr;
    }
    
    void poolStop()
    {
      // Gives all worker a trivial work: mark the pool as not started
      parallel([](const int&)
	       {
		 poolIsStarted=false;
	       });
      
      if(useDetachedPool)
	{
	  // Join threads
	  for(int threadId=1;threadId<nThreads;threadId++)
	    if(pthread_join(resources::pool[threadId],nullptr)!=0)
	      CRASHER<<"joining thread "<<threadId<<endl;
	  
	  // Remove all pthreads
	  resources::pool.resize(0);
	}
    }
  }
}
