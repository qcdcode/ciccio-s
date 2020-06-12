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
	      
	      work=[](const int&){};
	      nThreadsWaitingForJob=0;
	    }
	  else
	    do
	      {
		nThreadsWaitingForJob.fetch_add(1);
		//printf("Waiting for assignemnt %d %d\n",threadId,nThreadsWaitingForJob.load());
		while(nThreadsWaitingForJob!=0);
		//printf("Work assigned %d\n",threadId);
		work(threadId);
		//printf("Work finished %d\n",threadId);
	      }
	    while(poolIsStarted);
	}
    }
  }
}
