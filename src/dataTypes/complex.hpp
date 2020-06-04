#ifndef _COMPLEX_HPP
#define _COMPLEX_HPP

#include <array>

#include "base/metaProgramming.hpp"
#include "dataTypes/SIMD.hpp"

namespace ciccios
{
  enum{RE,IM};
  
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
      
      out[RE]=t[RE]*oth[RE]-t[IM]*oth[IM];
      out[IM]=t[RE]*oth[IM]+t[IM]*oth[RE];
      
      return out;
    }
    
    /// Summassign
    Complex& operator+=(const Complex& oth)
    {
      /// Alias for this
      Complex& t=*this;
      
      t[RE]+=oth[RE];
      t[IM]+=oth[IM];
      
      return t;
    }
    
    /// Product with another complex
    ALWAYS_INLINE Complex& sumProd(const Complex oth1,const Complex oth2) //Don't take it by reference or aliasing might arise
    {
      /// Alias for this
      Complex& t=*this;
      
      if(0)
	{
	  t[RE]+=oth1[RE]*oth2[RE]-oth1[IM]*oth2[IM];
	  t[IM]+=oth1[RE]*oth2[IM]+oth1[IM]*oth2[RE];
	}
      else
	{
	  t[RE]+=oth1[RE]*oth2[RE];
	  t[RE]-=oth1[IM]*oth2[IM];
	  t[IM]+=oth1[RE]*oth2[IM];
	  t[IM]+=oth1[IM]*oth2[RE];
	}
      
      return t;
    }
  };
  
  /// Simd version of a complex
  template <typename Fund>
  using SimdComplex=Complex<Simd<Fund>>;
}

#endif
