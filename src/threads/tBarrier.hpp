#ifndef _TBARRIER_HPP
#define _TBARRIER_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <cstring>

#include <base/debug.hpp>
#include <base/logger.hpp>

namespace ciccios
{
  namespace threads
  {
    /// Wrapper for the pthread barrier functionality
    ///
    /// Low level barrier not meant to be called explictly
    struct Barrier
    {
      /// Raw barrier
      pthread_barrier_t barrier;
      
#ifdef THREAD_DEBUG_MODE
      
      /// Value used to check the barrier
      [[maybe_unused ]]
      const char* currBarrName;
      
#endif
      
      /// Raw synchronization, simply wait that all threads call this
      void rawSync()
      {
	/// Call the barrier and get the result
	const int rc=
	  pthread_barrier_wait(&barrier);
	
	// if(rc!=0 and rc!=PTHREAD_BARRIER_SERIAL_THREAD)
	//   CRASHER<<"while barrier was waiting"<<endl;
      }
      
      /// Build the barrier for \c nThreads threads
      Barrier(const int& nThreads) ///< Number of threads for which the barrier is defined
      {
	if(pthread_barrier_init(&barrier,nullptr,nThreads)!=0)
	  CRASHER<<"while barrier inited"<<endl;
      }
      
      /// Destroys the barrier
      ~Barrier()
      {
	if(pthread_barrier_destroy(&barrier)!=0)
	  CRASHER<<"while barrier was destroyed"<<endl;
      }
      
      /// Synchronize, without checking the name of the barrier
      void sync()
      {
	rawSync();
      }
      
      /// Synchronize checking the name of the barrier
      void sync(const char* barrName, ///< Name of the barrier
		const int& threadId)  ///< Id of the thread used coming to check
      {
	rawSync();
	
#ifdef THREAD_DEBUG_MODE
      
	if(threadId==masterThreadId)
	  currBarrName=barrName;
	
	sync();
	
	if(currBarrName!=barrName)
	  CRASHER<<"Thread id "<<threadId<<" was expecting "<<currBarrName<<" but encountered "<<barrName<<endl;
#endif
      }
    };
  }
}
#endif
