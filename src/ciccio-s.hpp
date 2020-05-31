#ifndef _CICCIO_S_HPP
#define _CICCIO_S_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "base.hpp"
#include "dataTypes.hpp"

namespace ciccios
{
  namespace resources
  {
    /// Holds the time at which the program has been compiled
    char PROG_COMPILE_TIME[]=__TIME__;
    /// Holds the date at which the program has been compiled
    char PROG_COMPILE_DATE[]=__DATE__;
  }
}

#endif
