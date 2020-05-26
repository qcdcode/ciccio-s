#ifndef _RANKS_HPP
#define _RANKS_HPP

#include <mpi.h>

#ifndef EXTERN_RANK
 #define EXTERN_RANK extern
#define INIT_RANK_TO(...)
#else
 #define INIT_RANK_TO(...) (__VA_ARGS__)
#endif

namespace ciccios
{
  namespace resources
  {
    /// Current rank
    EXTERN_RANK int rank INIT_RANK_TO(0);
    
    /// Total number of ranks
    EXTERN_RANK int nRanks INIT_RANK_TO(1);
  }
  
  /// Returns current rank
  inline const int& rank()
  {
    return resources::rank;
  }
  
  /// Returns the total number of ranks
  inline const int& nRranks()
  {
    return resources::nRanks;
  }
  
  /// Returns true on master rank
  inline bool isMasterRank()
  {
    return rank()==0;
  }
  
  /// Initialize MPI
  void initRanks(int& narg,char **&arg);
  
  /// Finalize MPI
  void finalizeRanks();
  
  /// Abort all ranks
  void ranksAbort(const int& errNo);
}

#undef EXTERN_RANK
#undef INIT_RANK_TO

#endif
