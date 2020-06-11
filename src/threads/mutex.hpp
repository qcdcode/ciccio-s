#ifndef _MUTEX_HPP
#define _MUTEX_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <cstring>
#include <pthread.h>

#include "base/debug.hpp"

namespace ciccios
{
  /// Class to lock a mutex for the object scope
  struct Mutex
  {
    /// Internal mutex
    pthread_mutex_t mutex PTHREAD_MUTEX_INITIALIZER;
    
  public:
    
    /// Lock the mutex
    void lock()
    {
      if(pthread_mutex_lock(&mutex))
	CRASHER<<"Error locking, "<<strerror(errno)<<endl;
    }
    
    /// Unlock the mutex
    void unlock()
    {
      if(pthread_mutex_unlock(&mutex))
	CRASHER<<"Error unlocking, "<<strerror(errno)<<endl;
    }
  };
  
  /// Keep a mutex locked for the duration of the object
  struct ScopeMutexLocker
  {
    /// Reference to the mutex
    Mutex& mutex;
    
  public:
    
    /// Create, store the reference and lock
    ScopeMutexLocker(Mutex& mutex) ///< Mutex to be kept locked
      : mutex(mutex)
    {
      mutex.lock();
    }
    
    /// Unlock and destroy
    ~ScopeMutexLocker()
    {
      mutex.unlock();
    }
  };
}

#endif
