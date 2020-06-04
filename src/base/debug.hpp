#ifndef _DEBUG_HPP
#define _DEBUG_HPP

#ifdef USE_CUDA
 #include <cuda_runtime.h>
#endif

#include <chrono>

#include "preprocessor.hpp"
#include "logger.hpp"

namespace ciccios
{
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
  
/// Include a comment in the assembler, recognizable in the compilation
#define ASM_BOOKMARK(COMMENT)					\
  asm("#Bookmark file: \"" __FILE__ "\", line: " LINE_AS_STRING  ", " COMMENT)
  
  /// Put a BEGIN for asm bookmark section
#define ASM_BOOKMARK_BEGIN(COMMENT)					\
  ASM_BOOKMARK("BEGIN " COMMENT)
  
  /// Put an END for asm bookmark section
#define ASM_BOOKMARK_END(COMMENT)					\
  ASM_BOOKMARK("END " COMMENT)
  
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
}

#endif
