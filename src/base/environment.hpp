#ifndef _ENVIRONMENT_HPP
#define _ENVIRONMENT_HPP

/// \file environment.hpp
///
/// \brief Functions needed to read environment variables

#ifndef EXTERN_ENVIRONMENT
 
 /// Make external if put in front of a variable
 ///
 /// Actual allocation is done in the cpp file
 #define EXTERN_ENVIRONMENT extern
 
 /// Defines the list of argument in the cpp, or make it external in the hpp
 #define FLAG_LIST(ARGS...) extern const decltype(ARGS) flagList
 
#else

 #define FLAG_LIST(ARGS...) const auto flagList=ARGS

#endif

#include <tuple>

#include <base/debug.hpp>
#include <threads/pool.hpp>

namespace ciccios
{
  /// Type to hold flag to be read from environment
  template <typename T>
  using Flag=std::tuple<T&,const T,const char*,const char*>;
  
  /// List of known flags
  FLAG_LIST(std::make_tuple(std::make_tuple(&waitToAttachDebuggerFlag,false,"WAIT_TO_ATTACH_DEBUGGER","to be used to wait for gdb to attach")
#ifdef USE_THREADS
			    ,std::make_tuple(&useDetachedPool,false,"USE_DETACHED_POOL","to be used to create a pool at the begin")
#endif
			    ));
  
  /// Read all flags from environment
  void readAllFlags();
}

#undef FLAG_LIST
#undef EXTERN_ENVIRONMENT

#endif
