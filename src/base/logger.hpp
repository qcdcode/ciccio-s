#ifndef _LOGGER_HPP
#define _LOGGER_HPP

#ifndef EXTERN_LOGGER
 #define EXTERN_LOGGER extern
#define INIT_LOGGER_TO(...)
#else
 #define INIT_LOGGER_TO(...) (__VA_ARGS__)
#endif

#include <fstream>

#include "base/ranks.hpp"

namespace ciccios
{
  using std::endl;
  
  namespace resources
  {
    /// Actual logger writing to the console
    EXTERN_LOGGER std::ofstream actualLogger INIT_LOGGER_TO("/dev/stderr");
    
    /// Wired out logger
    EXTERN_LOGGER std::ofstream dummyLogger INIT_LOGGER_TO("/dev/null");
  }
  
  /// Returns the true logger or the dymmy one depending if on master rank
  inline std::ofstream& logger()
  {
    using namespace resources;
    
    if(isMasterRank())
      return actualLogger;
    else
      return dummyLogger;
  }
  
  //print the banner
  void printBanner();
}

#undef EXTERN_LOGGER
#undef INIT_LOGGER_TO

#endif
