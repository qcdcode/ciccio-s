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
    /// Actual logger writing to the console, standard output
    EXTERN_LOGGER std::ofstream logger INIT_LOGGER_TO("/dev/stdout");
    
    /// Actual logger writing to the console, standard error
    EXTERN_LOGGER std::ofstream errLogger INIT_LOGGER_TO("/dev/stdout");
    
    /// Wired out logger
    EXTERN_LOGGER std::ofstream dummyLogger INIT_LOGGER_TO("/dev/null");
    
    /// Level of verbosity to be used for logging
    EXTERN_LOGGER int verbosityLv INIT_LOGGER_TO(1);
  }
  
  /// Returns the true logger or the dummy one depending if on master rank
  inline std::ofstream& logger()
  {
    if(isMasterRank())
      return resources::logger;
    else
      return resources::dummyLogger;
  }
  
  /// Returns the error logger or the dummy one depending if on master rank
  inline std::ofstream& errLogger()
  {
    if(isMasterRank())
      return resources::errLogger;
    else
      return resources::dummyLogger;
  }
  
  /// Verbose logger or not
  inline std::ofstream& verbLogger(const int& lv)
  {
    if(lv<=resources::verbosityLv)
      return logger();
    else
      return resources::dummyLogger;
  }
  
  /// Shortcut to avoid having to put ()
#define LOGGER logger()
  
  /// Verbose logger or not, capital worded for homogeneity
#define VERB_LOGGER(LV) verbLogger(LV)
  
  /// Prints the banner
  void printBanner();
}

#undef EXTERN_LOGGER
#undef INIT_LOGGER_TO

#endif
