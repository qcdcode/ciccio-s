#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <execinfo.h>
#include <unistd.h>

#ifdef USE_CUDA
 #include <cuda_runtime.h>
#endif

#include "debug.hpp"
#include "git_info.hpp"
#include "logger.hpp"

namespace ciccios
{
  void printBacktraceList(std::ofstream& out)
  {
    int nMaxStack=128;
    void *callStack[nMaxStack];
    int frames=backtrace(callStack,nMaxStack);
    char **strs=backtrace_symbols(callStack,frames);
    
    out<<"Backtracing..."<<endl;
    for(int i=0;i<frames;i++) out<<strs[i]<<endl;
    
    free(strs);
  }
  
  void waitToAttachDebugger()
  {
    /// Flag used to trap
    volatile int flag=0;
    
    printf("Entering debug loop on rank %d, flag has address %p please type:\n"
	   "$ gdb -p %d\n"
	   "$ set flag=1\n"
	   "$ continue\n",
	   rank(),
	   &flag,
	   getpid());
    
    if(rank()==0)
      while(flag==0);
    
    ranksBarrier();
  }
  
  void possiblyWaitToAttachDebugger()
  {
    /// String used to detect debugging directive
    const char DEBUG_LOOP_STRING[]="WAIT_TO_ATTACH_DEBUGGER";
    
    if(getenv(DEBUG_LOOP_STRING)!=NULL)
      waitToAttachDebugger();
    else
      LOGGER<<"To wait attaching the debugger please export: "<<DEBUG_LOOP_STRING<<"=1"<<endl;
  }
  
  namespace resources
  {
    /// Compile time, detected when including the ciccio-s.hpp file
    extern char PROG_COMPILE_TIME[];
    
    /// Compile date, detected when including the ciccio-s.hpp file
    extern char PROG_COMPILE_DATE[];
  }
  
#if USE_CUDA
  void internalDecryptCudaError(const int lineNo,const char* fileName,const char* funcName,const cudaError_t rc,const char *templ,...)
  {
    if(rc!=cudaSuccess)
      {
	char mess[1024];
	va_list ap;
	va_start(ap,templ);
	va_end(ap);
	
	vsprintf(mess,templ,ap);
	Crasher(lineNo,fileName,funcName)<<mess<<", cuda raisded error: "<<cudaGetErrorString(rc)<<endl;
      }
  }
#endif
  void printVersionAndCompileFlags(std::ofstream& out)
  {
    out<<endl;
    out<<"Git hash: "<<GIT_HASH<<", last commit at "<<GIT_TIME<<" with message: \""<<GIT_LOG<<"\""<<endl;
    out<<"Configured at "<<CONFIGURE_TIME<<" with flags: "<<CONFIGURE_FLAGS<<endl;
    out<<"Library compiled at "<<__TIME__<<" of "<<__DATE__<<
      ", executable compiled at "<<resources::PROG_COMPILE_TIME<<" of "<<resources::PROG_COMPILE_DATE<<endl;
    out<<endl;
  }
}
