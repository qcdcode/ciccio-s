#ifndef _SIMD_HPP
#define _SIMD_HPP

#include <immintrin.h>

// \todo Rename Simd into something more appropriate

namespace ciccios
{
#if 1
  
  namespace resources
  {
    template <typename Fund>
    struct Simd;
    
    template <>
    struct Simd<double>
    {
      using Type=__m256d;
    };
    
    template <>
    struct Simd<float>
    {
      using Type=__m256;
    };
  }
  
  template <typename Fund>
  using Simd=typename resources::Simd<Fund>::Type;
  
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
  template <typename Fund>
  constexpr int simdLength=sizeof(Simd<Fund>)/sizeof(Fund);
}

#endif
