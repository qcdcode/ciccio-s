#ifndef _POOL_HPP
#define _POOL_HPP

#include <functional>
#include <pthread.h>
#include <thread>
#include <tuple>
#include <vector>

#include <threads/mutex.hpp>
#include <threads/tBarrier.hpp>

#include <external/inplace_function.h>

#ifndef EXTERN_POOL
 #define EXTERN_POOL extern
#define INIT_POOL_TO(...)
#else
 #define INIT_POOL_TO(...) (__VA_ARGS__)
#endif

namespace ciccios
{
  // EXTERN_POOL int _beg,_end,_nPieces;
  // EXTERN_POOL void(*_f)(const int&,const int&);
  
  /// Thread id of master thread
  [[ maybe_unused ]]
  constexpr int masterThreadId=
    0;
  
  /// Contains a thread pool
  struct ThreadPool
  {
    /// Maximal size of the stack used for thw work
    static constexpr int MAX_POOL_FUNCTION_SIZE=128;
    
    /// States if the pool is waiting for work
    bool isWaitingForWork{false};
    
    /// States if the pool is filled
    bool isFilled{false};
    
    /// Thread id of master thread
    const pthread_t masterThreadTag{getThreadTag()};
    
    /// Type to encapsulate the work to be done
    using Work=
      // void(*)(const int&);
      //std::function<void(int)>;
    stdext::inplace_function<void(int),MAX_POOL_FUNCTION_SIZE>;
    
    /// Work to be done in the pool
    ///
    /// This incapsulates a function returning void, and getting an
    /// integer as an argument, corresponding to the thread
    Work work;
    
    /// Incapsulate the threads
    ///
    /// At the beginning, the pool contains only the main thread, with
    /// its id. Then when is filled, the pool contains the thread
    /// identifier. This is an opaque number, which cannot serve the
    /// purpose of getting the thread progressive in the pool. This is
    /// why we define the next function
    std::vector<pthread_t> pool;
    
    /// Return the thread tag
    static pthread_t getThreadTag()
    {
      return pthread_self();
    }
    
    /// Number of threads
    int nThreads;
    
    /// Barrier used by the threads
    threads::Barrier barrier;
    
    /// Pair of parameters containing the threadpool and the thread id
    using ThreadPars=std::tuple<ThreadPool*,int>;
    
    /// Function called when starting a thread
    ///
    /// When called, get the thread pool and the thread id as
    /// arguments through the function parameter. This is expcted to
    /// be allocated outside through a \c new call, so it is deleted
    /// after taking reference to the pool, and checking the thread thread.
    ///
    /// All threads but the master one swim in this pool back and forth,
    /// waiting for job to be done.
    static void* threadPoolSwim(void* _ptr) ///< Initialization data
    {
      /// Cast the \c void pointer to the tuple
      ThreadPool::ThreadPars* ptr=static_cast<ThreadPool::ThreadPars*>(_ptr);
      
      /// Takes a reference to the parameters
      ThreadPool::ThreadPars& pars=*ptr;
      
      /// Takes a reference to the pool
      ThreadPool& pool=*std::get<0>(pars);
      
      /// Copy the thread id
      const int threadId=std::get<1>(pars);
      
      delete ptr;
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"entering the pool"<<endl;
#endif
      
      /// Work until asked to empty
      bool keepSwimming=pool.isFilled;
      
      pool.tellTheMasterThreadIsCreated(threadId);
      
      while(keepSwimming)
	{
	  pool.waitForWorkToBeAssigned(threadId);
	  
	  keepSwimming=pool.isFilled;
	  
#ifdef USE_THREADS_DEBUG
	  LOGGER<<" keep swimming: "<<keepSwimming<<endl;
#endif
	  
	  if(keepSwimming)
	    {
	      pool.work(threadId);
	      
	      pool.tellTheMasterWorkIsFinished(threadId);
	    }
	}
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"exiting the pool"<<endl;
#endif
      
      return nullptr;
    }
    
    /// Fill the pool with the number of thread assigned
    void fill(const pthread_attr_t* attr=nullptr); ///< Possible attributes of the threads
    
    /// Stop the pool
    void doNotworkAnymore()
    {
      // Mark that the pool is not waiting any more for work
      isWaitingForWork=false;
      
      // Temporary name to force the other threads go out of the pool
      barrier.sync();
    }
    
    /// Empty the thread pool
    void empty()
    {
      // Check that the pool is not empty
      if(not isFilled)
	CRASHER<<"Cannot empty an empty pool!"<<endl;
      
      // Mark that the pool is not filled
      isFilled=
	false;
      
      /// Stop the pool from working
      tellThePoolNotToWorkAnyLonger(masterThreadId);
      
      for(int threadId=1;threadId<nThreads;threadId++)
	{
	  if(pthread_join(pool[threadId],nullptr)!=0)
	    CRASHER<<"joining threads"<<endl;
	  
#ifdef USE_THREADS_DEBUG
	  LOGGER<<"Thread of id "<<(int)threadId<<" destroyed"<<endl;
#endif
	}
      
      // Resize down the pool
      pool.resize(1);
    }
    
    /// Assert that only the pool is accessing
    void assertPoolOnly(const int& threadId) ///< Calling thread
      const
    {
      if(threadId==masterThreadId)
	CRASHER<<"Only pool threads are allowed"<<endl;
    }
    
    /// Assert that only the master thread is accessing
    void assertMasterOnly(const int& threadId) ///< Calling thread
      const
    {
      if(threadId!=masterThreadId)
	CRASHER<<"Only master thread is allowed, but thread "<<threadId<<" is trying to act"<<endl;
    }
    
    /// Get the thread id of the current thread
    int getThreadId() const
    {
      /// Current pthread
      const pthread_t threadTag=getThreadTag();
      
      /// Position in the pool
      int threadId=0;
      while(pool[threadId]!=threadTag and threadId<nActiveThreads())
	threadId++;
      
      // Check that the thread is found
      if(threadId==nActiveThreads())
	{
	  fprintf(stdout,"%d %d\n",threadId,nActiveThreads());
	  for(auto & p : pool)
	    fprintf(stdout,"%d\n",(int)p);
	  CRASHER<<"Unable to find thread with tag "<<threadTag<<endl;
	}
      
      return
	threadId;
    }
    
    /// Global mutex
    Mutex mutex;
    
    /// Puts a scope mutex locker making the scope sequential
#define THREADS_SCOPE_SEQUENTIAL()				\
    ScopeMutexLocker sequentializer ## __LINE__ (mutex);
    
    /// Lock the internal mutex
    void mutexLock()
    {
      mutex.lock();
    }
    
    /// Unlock the mutex
    void mutexUnlock()
    {
      mutex.unlock();
    }
    
    /// Compares the thread tag with the master one
    bool isMasterThread()
      const
    {
      return
	getThreadTag()==masterThreadTag;
    }
    
    /// Gets the number of allocated threads
    int nActiveThreads()
      const
    {
      return
	pool.size();
    }
    
    /// Tag to mark that assignment has been finished
    static constexpr const char* workAssignmentTag()
    {
      return "WorkAssOrNoMoreWork";
    }
    
    /// Tag to mark that no more work has to do
    static constexpr auto& workNoMoreTag=
      workAssignmentTag;
    
    /// Start the work for the other threads
    void tellThePoolWorkIsAssigned(const int& threadId) ///< Thread id
    {
      assertMasterOnly(threadId);
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"Telling the pool that work has been assigned (tag: "<<workAssignmentTag()<<")"<<endl;
#endif
      
      // Mark down that the pool is not waiting for work
      isWaitingForWork=false;
      
      // The master signals to the pool to start work by synchronizing with it
      barrier.sync(workAssignmentTag(),threadId);
    }
    
    /// Tag to mark that the thread is ready to swim
    static constexpr const char* threadHasBeenCreated()
    {
      return "ThreadHasBeenCreated";
    }
    
    /// Tell the master that the thread is created and ready to swim
    void tellTheMasterThreadIsCreated(const int& threadId) ///< Thread id
    {
      assertPoolOnly(threadId);
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"Telling that thread "<<threadId<<" has been created and is ready to swim (tag: "<<threadHasBeenCreated()<<")"<<endl;
#endif
      
      // The thread signals to the master that has been created and ready to swim
      barrier.sync(threadHasBeenCreated(),threadId);
    }
    
    /// Waiting for threads are created and ready to swim
    void waitPoolToBeFilled(const int& threadId) ///< Thread id
    {
      assertMasterOnly(threadId);
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"waiting for threads in the pool to be ready to swim (tag: "<<threadHasBeenCreated()<<")"<<endl;
#endif
      
      // The master wait that the threads have been created by syncing with them
      barrier.sync(threadHasBeenCreated(),threadId);
    }
    
    /// Waiting for work to be done means to synchronize with the master
    void waitForWorkToBeAssigned(const int& threadId) ///< Thread id
    {
      assertPoolOnly(threadId);
      
      // This printing is messing up, because is occurring in the pool
      // where the thread is expected to be already waiting for work,
      // and is not locking the logger correctly
      //minimalLogger(runLog,"waiting in the pool for work to be assigned (tag %s)",workAssignmentTag);
      
      barrier.sync(workAssignmentTag(),threadId);
    }
    
    /// Stop the pool from working
    void tellThePoolNotToWorkAnyLonger(const int& threadId) ///< Thread id
    {
      assertMasterOnly(threadId);
      
      if(not isWaitingForWork)
	CRASHER<<"We cannot stop a working pool"<<endl;
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"Telling the pool not to work any longer (tag: "<<workNoMoreTag()<<")"<<endl;
#endif
      
      // Mark down that the pool is waiting for work
      isWaitingForWork=
	false;
      
      // The master signals to the pool that he is waiting for the
      // pool to finish the work
      barrier.sync(workNoMoreTag(),threadId);
    }
    
    /// Tag to mark that the work is finished
    static const char* workFinishedTag()
    {
      return "WorkFinished";
    }
    
    /// Waiting for work to be done means to synchronize with the master
    void tellTheMasterWorkIsFinished(const int& threadId) ///< Thread id
    {
      assertPoolOnly(threadId);
      
#ifdef USE_THREADS_DEBUG
      LOGGER<<"finished working (tag: "<<workFinishedTag()<<")"<<endl;
#endif
      
      barrier.sync(workFinishedTag(),threadId);
    }
    
    /// Wait that the work assigned to the pool is finished
    void waitForPoolToFinishAssignedWork(const int& threadId) ///< Thread id
    {
      assertMasterOnly(threadId);
      
      // if constexpr(DEBUG_THREADS)
      // 	{
      // 	  /// Makes the print sequential across threads
      // 	  THREADS_SCOPE_SEQUENTIAL();
	  
      // 	  minimalLogger(runLog,"waiting for pool to finish the work (tag: %s)",workFinishedTag);
      // 	  mutexUnlock();
      // 	}
      
      // The master signals to the pool that he is waiting for the
      // pool to finish the work
      barrier.sync(workFinishedTag(),threadId);
      
      // Mark down that the pool is waiting for work
      isWaitingForWork=true;
    }
    
    /// Return whether the pool is waiting for work
    const bool& getIfWaitingForWork() const
    {
      return isWaitingForWork;
    }
    
    /// Gives to all threads some work to be done
    ///
    /// The object \c f must be callable, returning void and getting
    /// an integer as a parameter, representing the thread id
    //void workOn(Work&& f) ///< Function embedding the work
    template <typename F>
    void workOn(F&& f) ///< Function embedding the work
    {
      // Check that the pool is waiting for work
      // if(not isWaitingForWork)
      // 	CRASHER<<"Trying to give work to not-waiting pool!"<<endl;
      
      // Store the work
      // asm("#pre");
      work=
	std::move(f);
      // asm("#pre");
      
      // Set off the other threads
      tellThePoolWorkIsAssigned(masterThreadId);
      
      work(0);
      
      // Wait that the pool finishes the work
      waitForPoolToFinishAssignedWork(masterThreadId);
    }
    
    /// Split a loop into \c nTrheads chunks, giving each chunk as a work for a corresponding thread
    template <typename Size,           // Type for the range of the loop
	      typename F>
    void loopSplit(const Size& beg,  ///< Beginning of the loop
		   const Size& end,  ///< End of the loop
		   const F& f)       ///< Function to be called, accepting two integers: the first is the thread id, the second the loop argument
    //void(*f)(const int&,const int&))       ///< Function to be called, accepting two integers: the first is the thread id, the second the loop argument
    {
      workOn([beg,end,nPieces=this->nActiveThreads(),&f](const int& threadId) INLINE_ATTRIBUTE
      // _beg=beg;
      // _end=end;
      // _nPieces=this->nActiveThreads();
      // _f=f;
      
      //workOn([](const int& threadId) //INLINE_ATTRIBUTE
	     {
	       /// Workload for each thread, taking into account the remainder
	       const Size threadLoad=
		 //(_end-_beg+_nPieces-1)/_nPieces;
		 (end-beg+nPieces-1)/nPieces;
	       
	       /// Beginning of the chunk
	       const Size threadBeg=
		 threadLoad*threadId;
	       
	       /// End of the chunk
	       const Size threadEnd=
		 //std::min(_end,threadBeg+threadLoad);
		 std::min(end,threadBeg+threadLoad);
	       
	       for(Size i=threadBeg;i<threadEnd;i++)
		 // _f
		 f(threadId,i);
	     });
    }
    
    /// Constructor starting the thread pool with a given number of threads
    ThreadPool(int nThreads) :
      pool(1,getThreadTag()),
      nThreads(nThreads),
      barrier(nThreads)
    {
      fill();
    }
    
    /// Destructor emptying the pool
    ~ThreadPool()
    {
      LOGGER<<"Destroying the pool"<<endl;
      empty();
    }
  };
  
  /// Global thread pool
  EXTERN_POOL ThreadPool* threadPool;
}

#undef EXTERN_POOL
#undef INIT_POOL_TO

#endif