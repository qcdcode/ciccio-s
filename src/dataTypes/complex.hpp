#ifndef _COMPLEX_HPP
#define _COMPLEX_HPP

#include <array>

#include "base/metaProgramming.hpp"
#include "base/unroll.hpp"
#include "dataTypes/SIMD.hpp"

namespace ciccios
{
  enum{RE,IM};
  
  /// Complex number
  template <typename T>
  struct Complex
  {
    T real;
    T imag;
    
    /// Product with another complex
    Complex operator*(const Complex& oth) const
    {
      /// Alias for this
      const Complex& t=*this;
      
      /// Result
      Complex out;
      
      out.real=t.real*oth.real-t.imag*oth.imag;
      out.imag=t.real*oth.imag+t.imag*oth.real;
      
      return out;
    }
    
    /// Summassign
    Complex& operator+=(const Complex& oth)
    {
      /// Alias for this
      Complex& t=*this;
      
      t.real+=oth.real;
      t.imag+=oth.imag;
      
      return t;
    }
    
    /// Product with another complex
    ALWAYS_INLINE Complex& sumProd(const Complex oth1,const Complex oth2) //Don't take it by reference or aliasing might arise
    {
      /// Alias for this
      Complex& t=*this;
      
      // Dumb compiler would not fuse this if we put together
      t.real+=oth1.real*oth2.real;
      t.real-=oth1.imag*oth2.imag;
      t.imag+=oth1.real*oth2.imag;
      t.imag+=oth1.imag*oth2.real;
      
      return t;
    }
  };
  
  /// Simd version of a complex
  template <typename Fund>
  using SimdComplex=Complex<Simd<Fund>>;
}

#endif
