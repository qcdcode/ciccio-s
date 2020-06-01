#ifndef _COMPLEX_HPP
#define _COMPLEX_HPP

#include <array>

#include "dataTypes/SIMD.hpp"

namespace ciccios
{
  /// Complex number
  template <typename T>
  struct Complex : public std::array<T,2>
  {
    /// Product with another complex
    Complex operator*(const Complex& oth) const
    {
      /// Alias for this
      const Complex& t=*this;
      
      /// Result
      Complex out;
      
      out[0]=t[0]*oth[0]-t[1]*oth[1];
      out[1]=t[0]*oth[1]+t[1]*oth[0];
      
      return out;
    }
    
    /// Summassign
    Complex& operator+=(const Complex& oth)
    {
      /// Alias for this
      Complex& t=*this;
      
      t[0]+=oth[0];
      t[1]+=oth[1];
      
      return t;
    }
  };
  
  /// Simd version of a complex
  using SimdComplex=Complex<Simd>;
  
}

#endif
