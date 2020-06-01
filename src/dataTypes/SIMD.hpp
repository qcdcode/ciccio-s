#ifndef _SIMD_HPP
#define _SIMD_HPP

#include <immintrin.h>

// \todo Rename Simd into something more appropriate

namespace ciccios
{
#define SIMD_TYPE M256D
  
#if SIMD_TYPE == M256D
  
  using Simd=__m256d;
  
#elif SIMD_TYPE == M256
  
  using Simd=__m256;
  
#else
  
  using Simd=std::array<double,1>;
  
  /// Summassign two simple SIMD
  inline Simd operator+=(Simd& a,const Simd& b)
  {
    a[0]+=b[0];
    
    return a;
  }
  
  /// Multiply two simple SIMD
  inline Simd operator*(const Simd&a,const Simd& b)
  {
    /// Result
    Simd c;
    
    c[0]=a[0]*b[0];
    
    return c;
  }
  
#endif
  
  /// Length of a SIMD vector
  constexpr int simdLength=sizeof(Simd)/sizeof(double);
}

#endif
