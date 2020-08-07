#ifndef _DEBUG_HPP
#define _DEBUG_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file debug.hpp
///
/// \brief Backtrace, assembly bookkeeping, crasher, etc

#ifndef EXTERN_DEBUG
 
 /// Make external if put in front of a variable
 ///
 /// Actual allocation is done in the cpp file
# define EXTERN_DEBUG extern
 
#endif

#ifdef USE_CUDA
# include <cuda_runtime.h>
#endif

#include <chrono>

#ifdef HAVE_CXXABI_H
# include <cxxabi.h>
#endif

#include <base/preprocessor.hpp>
#include <base/logger.hpp>
#include <base/metaProgramming.hpp>
#include <base/unroll.hpp>

namespace ciccios
{
  /// Wait to attach gdb
  EXTERN_DEBUG bool waitToAttachDebuggerFlag;
  
  /// Write the list of called routines
  void printBacktraceList(std::ofstream&);
  
  /// A Structure allowing to pass the crashing message and crash at the end
  struct Crasher
  {
    /// Failing line
    const int lineNo;
    
    /// Failing file name
    const char* fileName;
    
    /// Failing function name
    const char* funcName;
    
    /// Constructor taking the line number, the file name and the function name as an input
    Crasher(const int lineNo,const char* fileName,const char* funcName) : lineNo(lineNo),fileName(fileName),funcName(funcName)
    {
      errLogger()<<"\x1b[31m" "ERROR on line "<<lineNo<<" of file \""<<fileName<<"\", function \""<<funcName<<"\", message error: \"";
    }
    
    /// Overload calling with any operand
    template <typename T>
    Crasher& operator<<(T&& t)
    {
      errLogger()<<std::forward<T>(t);
      
      return
	*this;
    }
    
    /// Overload calling with a function, to capure endl
    Crasher& operator<<(std::ostream&(*p)(std::ostream&))
    {
      errLogger()<<p;
      
      return
	*this;
    }
    
    /// Destroy exiting
    ~Crasher()
    {
      errLogger()<<"\".\n\x1b[0m";
      printBacktraceList(errLogger());
      ranksAbort(0);
    }
  };
  
/// Invoke the crasher, passing line, file and function
#define CRASHER Crasher(__LINE__,__FILE__,__FUNCTION__)

#ifdef COMPILING_FOR_DEVICE
 
  /// Symbol to be used to begin an assembler comment, different in nvcc
# define _ASM_BOOKMARK_SYMBOL "//"
 
#else
 
# define _ASM_BOOKMARK_SYMBOL "#"
 
#endif
  
/// Include a comment in the assembler, recognizable in the compilation
#define ASM_BOOKMARK(COMMENT)					\
  asm(_ASM_BOOKMARK_SYMBOL "Bookmark file: \"" __FILE__ "\", line: " LINE_AS_STRING  ", " COMMENT)
  
  /// Put a BEGIN for asm bookmark section
#define ASM_BOOKMARK_BEGIN(COMMENT)					\
  ASM_BOOKMARK("BEGIN " COMMENT)
  
  /// Put an END for asm bookmark section
#define ASM_BOOKMARK_END(COMMENT)					\
  ASM_BOOKMARK("END " COMMENT)
  
/////////////////////////////////////////////////////////////////
  
/// Defines an inlined function BOOKMARK_BEGIN/END_NAME(Args...)
///
/// Internal implementation
#define PROVIDE_ASM_DEBUG_HANDLE_BEGIN_OR_END(BE,NAME,ARGS...)		\
  /*! Put in the assembly a bookmark named composing name and the arguments */ \
  INLINE_FUNCTION CUDA_HOST_DEVICE						\
  void BOOKMARK_ ## BE ## _ ## NAME (ARGS)				\
  {									\
    ASM_BOOKMARK_ ## BE(#NAME #ARGS);					\
  }
  
  /// Defines an inlined function BOOKMARK_BEGIN/END_NAME(Args...)
  ///
  /// The arguments can be used to distinguish different template
  /// instances (e.g. double from float)
#define PROVIDE_ASM_DEBUG_HANDLE(NAME,ARGS...)			\
  PROVIDE_ASM_DEBUG_HANDLE_BEGIN_OR_END(BEGIN,NAME,ARGS)	\
  PROVIDE_ASM_DEBUG_HANDLE_BEGIN_OR_END(END,NAME,ARGS)
  
  /// Implements the trap to debug
  void possiblyWaitToAttachDebugger();
  
  /// Print version, configuration and compilation time
  void printVersionAndCompileFlags(std::ofstream& out);
  
#ifdef USE_CUDA
  
  /// Put line, file and function in the actual call
#  define DECRYPT_CUDA_ERROR(...) \
  internalDecryptCudaError(__LINE__,__FILE__,__FUNCTION__,__VA_ARGS__)
  
  /// Crash with a cuda error
  void internalDecryptCudaError(const int lineNo,        ///< Line of error
				const char *fileName,    ///< Filename
				const char* function,    ///< Function where the error occurred
				const cudaError_t rc,    ///< Error code
				const char *templ,...);  ///< Message template

#endif
  
  /////////////////////////////////////////////////////////////////
  
  /// Measure time
  using Instant=
    std::chrono::time_point<std::chrono::steady_clock>;
  
  /// Returns the current time
  inline Instant takeTime()
  {
    return
      std::chrono::steady_clock::now();
  }
  
  /// Difference between two instants
  inline double timeDiffInSec(const Instant& end,   ///< Starting moment
			      const Instant& start) ///< Ending moment
  {
    return
      std::chrono::duration<double>(end-start).count();
  }
  
  /////////////////////////////////////////////////////////////////
  
  PROVIDE_HAS_MEMBER(nameOfType);
  
  /// Generic call to related method for a class type
  template <typename T,
	    ENABLE_TEMPLATE_IF(not hasMember_nameOfType<T>)>
  std::string nameOfType(T*)
  {
    /// Mangled name
    std::string name=
      typeid(T).name();
    
#ifdef HAVE_CXXABI_H
    
    /// Status returned
    int status=0;
    
    /// Demangled name
    char* demangled=
      abi::__cxa_demangle(name.c_str(),nullptr,nullptr,&status);
    
    if(status==0)
      name=demangled;
    
    if(status!=-1)
      free(demangled);
    
#endif
    
    return
      name;
  }
  
  /// Generic call to related method for a class type
  template <typename T,
	    ENABLE_TEMPLATE_IF(hasMember_nameOfType<T>)>
  std::string nameOfType(T*)
  {
    return
      std::decay_t<T>::nameOfType();
  }
  
  /// Returns the name of a type
#define NAME_OF_TYPE(A) \
  ciccios::nameOfType((A*){nullptr})
}

#undef EXTERN_DEBUG

#endif
