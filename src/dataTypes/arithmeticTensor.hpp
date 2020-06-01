#ifndef _ARITHMETIC_TENSOR_HPP
#define _ARITHMETIC_TENSOR_HPP

#include <array>

#include "base/debug.hpp"
#include "base/metaProgramming.hpp"
#include "base/preprocessor.hpp"

/// Simple implementation of array and matrix
///
/// To be replaced with the metaprogrammed type-aware tensor

namespace ciccios
{
  /// An array
  template <typename T,
	    int N>
  struct ArithmeticArray : public std::array<T,N>
  {
    /// Multiply another array
    template <typename U,
	      typename R=decltype(T()*U())>
    auto operator*(const ArithmeticArray<U,N>& oth) const
    {
      /// Result
      ArithmeticArray<R,N> out={};
      
      for(int i=0;i<N;i++)
	out[i]+=(*this)[i]*oth[i];
      
      return out;
    }
    
    /// Summassign another array
    template <typename U>
    auto operator+=(const ArithmeticArray<U,N>& oth)
    {
      for(int i=0;i<N;i++)
	(*this)[i]+=oth[i];
      
      return *this;
    }
  };
  
  /// A matrix
  template <typename T,
	    int N>
  struct ArithmeticMatrix : public std::array<std::array<T,N>,N>
  {
    /// Multiply another matrix
    template <typename U,
	      typename R=decltype(T()*U())>
    auto operator*(const ArithmeticMatrix<U,N>& oth) const
    {
      /// Result
      ArithmeticMatrix<R,N> out={};
      
      ASM_BOOKMARK("Matrix multiplication begin");
      unrollLoop<N>([&](int ir){
		      unrollLoop<N>([&](int ic){
				      unrollLoop<N>([&](int i)
						    {
						      out[ir][ic]+=(*this)[ir][i]*oth[i][ic];
						    });}
			);});
      ASM_BOOKMARK("Matrix multiplication end");
      
      return out;
    }
    
    /// Summassign another matrix
    template <typename U>
    auto operator+=(const ArithmeticMatrix<U,N>& oth)
    {
      unrollLoop<N>([&](int ir){
		      unrollLoop<N>([&](int ic){
				      (*this)[ir][ic]+=oth[ir][ic];
				    });
		    });
      
      return *this;
    }
  };
}

#endif

