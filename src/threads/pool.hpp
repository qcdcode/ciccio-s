#ifndef _POOL_HPP
#define _POOL_HPP

#include <atomic>
#include <functional>
#include <omp.h>
#include <tuple>
#include <vector>

//#include <threads/mutex.hpp>
//#include <threads/tBarrier.hpp>

#include <base/debug.hpp>

//#include <external/inplace_function.h>

#ifndef EXTERN_POOL
 #define EXTERN_POOL extern
#define INIT_POOL_TO(...)
#else
 #define INIT_POOL_TO(...) (__VA_ARGS__)
#endif

namespace ciccios
{
  /// Thread id of master thread
  [[ maybe_unused ]]
  constexpr int masterThreadId=0;
  
  /// Contains a thread pool
  namespace ThreadPool
  {
    /// Maximal size of the stack used for thw work
    //static constexpr int MAX_POOL_FUNCTION_SIZE=128;
    
    /// Number of threads waiting for work
    EXTERN_POOL std::atomic<int> nThreadsWaitingForWork INIT_POOL_TO(0);
    
    EXTERN_POOL std::atomic<int> nWorksAssigned INIT_POOL_TO(0);
    
    /// States if the pool is started
    EXTERN_POOL bool poolIsStarted INIT_POOL_TO(false);
    
    /// Type to encapsulate the work to be done
    using Work=
      std::function<void(int)>;
    //stdext::inplace_function<void(int),MAX_POOL_FUNCTION_SIZE>;
    
    /// Work to be done in the pool
    ///
    /// This incapsulates a function returning void, and getting an
    /// integer as an argument, corresponding to the thread
    EXTERN_POOL Work work;
    
    /// Number of threads
    EXTERN_POOL int nThreads;
    
    /// Assert that only the pool is accessing
    inline void assertPoolOnly(const int& threadId) ///< Calling thread
    {
      if(threadId==masterThreadId)
	CRASHER<<"Only pool threads are allowed"<<endl;
    }
    
    /// Assert that only the master thread is accessing
    inline void assertMasterOnly(const int& threadId) ///< Calling thread
    {
      if(threadId!=masterThreadId)
	CRASHER<<"Only master thread is allowed, but thread "<<threadId<<" is trying to act"<<endl;
    }
    
    /// Get the total number of threads
    inline int getNThreads()
    {
      int res;
      
      #pragma omp parallel
      res=omp_get_num_threads();
      
      return res;
    }
    
    /// Get the thread id of the current thread
    inline int getThreadId()
    {
      return omp_get_thread_num();
    }
    
    /// Compares the thread tag with the master one
    inline bool isMasterThread(const int& threadId)
    {
      return (threadId==masterThreadId);
    }
    
    inline void waitAllButMasterWaitForWork()
    {
      while(nThreadsWaitingForWork!=nThreads-1) // printf("NWaiting: %d/%d\n",nThreadsWaitingForWork.load(),nThreads-1)
						  ;
    }
    
    /// Gives to all threads some work to be done
    ///
    /// The object \c f must be callable, returning void and getting
    /// an integer as a parameter, representing the thread id
    //void workOn(Work&& f) ///< Function embedding the work
    template <typename F>
    INLINE_FUNCTION void workOn(F&& f) ///< Function embedding the work
    {
      waitAllButMasterWaitForWork();
      //printf("All works waiting\n");
      work=std::move(f);
      
      nThreadsWaitingForWork=0;
      nWorksAssigned.store(nWorksAssigned+1,std::memory_order_release);
      
      work(masterThreadId);
    }
    
    /// Split a loop into \c nTrheads chunks, giving each chunk as a work for a corresponding thread
    template <typename Size,           // Type for the range of the loop
	      typename F>              // Type of the function
    INLINE_FUNCTION
    void loopSplit(const Size& beg,  ///< Beginning of the loop
		   const Size& end,  ///< End of the loop
		   const F& f)       ///< Function to be called, accepting two integers: the first is the thread id, the second the loop argument
    {
      asm("#sending workon");
      workOn([beg,end,nPieces=nThreads,&f](const int& threadId) INLINE_ATTRIBUTE
	     {
	       /// Workload for each thread, taking into account the remainder
	       const Size threadLoad=
		 (end-beg+nPieces-1)/nPieces;
	       
	       /// Beginning of the chunk
	       const Size threadBeg=
		 threadLoad*threadId;
	       
	       /// End of the chunk
	       const Size threadEnd=
		 std::min(end,threadBeg+threadLoad);
	       
	       for(Size i=threadBeg;i<threadEnd;i++)
		 f(threadId,i);
	     });
      asm("#sent workon");
    }
    
    void poolThread(void(*replacementMain)(int narg,char **arg),int narg,char **arg);
  }
}

#undef EXTERN_POOL
#undef INIT_POOL_TO

#endif
