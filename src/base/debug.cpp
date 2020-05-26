#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <execinfo.h>

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
  

}
