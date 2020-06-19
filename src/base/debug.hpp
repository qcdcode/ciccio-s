#ifndef _DEBUG_HPP
#define _DEBUG_HPP

#ifndef EXTERN_DEBUG
 #define EXTERN_DEBUG extern
#endif

#ifdef USE_CUDA
 #include <cuda_runtime.h>
#endif

#include <chrono>

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
      
      return *this;
    }
    
    /// Overload calling with a function, to capure endl
    Crasher& operator<<(std::ostream&(*p)(std::ostream&))
    {
      errLogger()<<p;
      
      return *this;
    }
    
    /// Destroy exiting
    ~Crasher()
    {
      errLogger()<<"\".\n\x1b[0m";
      printBacktraceList(errLogger());
      ranksAbort(0);
    }
  };
  
#define CRASHER Crasher(__LINE__,__FILE__,__FUNCTION__)

#ifdef COMPILING_FOR_DEVICE
 #define _ASM_BOOKMARK_SYMBOL "//"
#else
 #define _ASM_BOOKMARK_SYMBOL "#"
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
  INLINE_FUNCTION HOST DEVICE						\
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
 #define DECRYPT_CUDA_ERROR(...)  internalDecryptCudaError(__LINE__,__FILE__,__FUNCTION__,__VA_ARGS__)
  void internalDecryptCudaError(const int lineNo,const char *fileName,const char* function,const cudaError_t rc,const char *templ,...);
#endif
  
  /////////////////////////////////////////////////////////////////
  
  /// Measure time
  using Instant=std::chrono::time_point<std::chrono::steady_clock>;
  
  /// Returns the current time
  inline Instant takeTime()
  {
    return std::chrono::steady_clock::now();
  }
  
  /// Difference between two instants
  inline double timeDiffInSec(const Instant& end,const Instant& start)
  {
    return std::chrono::duration<double>(end-start).count();
  }
  
  /////////////////////////////////////////////////////////////////
  
  /// Generic call to related method for a class type
  template <typename T,
	    SFINAE_ON_TEMPLATE_ARG(std::is_class<T>::value)>
  std::string nameOfType(T*)
  {
    return T::nameOfType();
  }
  
  /// Return "double"
  ///
  /// \todo All type traits into a struct
  INLINE_FUNCTION const char* nameOfType(double*)
  {
    return "double";
  }
  
  /// Return "float"
  INLINE_FUNCTION const char* nameOfType(float*)
  {
    return "float";
  }
  
  /// Returns the name of a type
#define NAME_OF_TYPE(A) \
  ciccios::nameOfType((A*){nullptr})
}

#undef EXTERN_DEBUG

#endif
