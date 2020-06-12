#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#define EXTERN_POOL
 #include "threads/pool.hpp"

namespace ciccios
{
  namespace ThreadPool
  {
    void poolThread(void(*replacementMain)(int narg,char **arg),int narg,char **arg)
    {
      // ALLOWS_ALL_THREADS_TO_PRINT_FOR_THIS_SCOPE(runLog);
      
      // Checks that the pool is not filled, to avoid recursive call
      if(poolIsStarted)
	CRASHER<<"Cannot fill again the pool!"<<endl;
      else
	poolIsStarted=true;
      
      nThreads=getNThreads();
      
      LOGGER<<"Filling the thread pool with "<<nThreads<<" threads"<<endl;
      
#pragma omp parallel
	{
	  const int threadId=getThreadId();
	  
	  if(isMasterThread(threadId))
	    {
	      replacementMain(narg,arg);
	      
	      waitAllButMasterWaitForWork();
	      
	      poolIsStarted=false;
	      
	      workOn([](const int&){});
	    }
	  else
	    {
	      do
		{
		  const int prevNWorkAssigned=nWorksAssigned;
		  
		  //int pre=
		    nThreadsWaitingForWork.fetch_add(1);
		  //printf("Waiting for assignemnt %d %d->%d\n",threadId,pre,nThreadsWaitingForWork.load());
		  while(nWorksAssigned.load(std::memory_order_relaxed)==prevNWorkAssigned)
		    //printf("Thread %d waiting for assignment (waiting: %d)\n",threadId,nThreadsWaitingForWork.load())
		    ;
		  std::atomic_thread_fence(std::memory_order_acquire);
		  
		  //printf("Work assigned %d\n",threadId);
		  work(threadId);
		  //printf("Work finished %d\n",threadId);
		}
	      while(poolIsStarted);
	      
	      printf("Exiting the pool, thread %d\n",poolIsStarted);
	    }
	}
    }
  }
}
