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
      ASM_BOOKMARK("Matrix sum begin");
      unrollLoop<N>([&](int ir){
		      unrollLoop<N>([&](int ic){
				      (*this)[ir][ic]+=oth[ir][ic];
				    });
		    });
      ASM_BOOKMARK("Matrix sum end");
      
      return *this;
    }
    
    /// Sum the product between two another matrices
    template <typename U1,
	      typename U2,
	      typename R=decltype(T()+=U1()*U2())>
    ALWAYS_INLINE auto& sumProd(const ArithmeticMatrix<U1,N>& __restrict oth1,const ArithmeticMatrix<U2,N>& __restrict oth2)
    {
      ASM_BOOKMARK_BEGIN("Metaprog unrolled");
      
      unrollLoopAlt<N>([&](int ir){
			 unrollLoopAlt<N>([&](int ic){
					    unrollLoopAlt<N>([&](int i){
							       auto& o=(*this)[ir][ic];
							       const auto& f=oth1[ir][i];
							       const auto& s=oth2[i][ic];
							       
							       if(0)
								 o+=f*s;
							       else
								 o.sumProd(f,s);
							     });
					  });
			   });
      
      ASM_BOOKMARK_END("Metaprog unrolled");
      
      return *this;
    }
  };
}

#endif

