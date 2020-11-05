#ifndef _ARITHMETIC_TENSOR_HPP
#define _ARITHMETIC_TENSOR_HPP

/// \file arithmeticTensor.hpp
///
/// \brief Simple implementation of array and matrix
///
/// \todo To be replaced with the metaprogrammed type-aware tensor

#include <base/debug.hpp>
#include <base/metaProgramming.hpp>
#include <base/preprocessor.hpp>

namespace ciccios
{
  /// An array
  template <typename T,
	    int N>
  struct ArithmeticArray
  {
    /// Internal data
    T data[N];
    
    /// Access to internal data
    INLINE_FUNCTION CUDA_HOST_DEVICE
    const T& operator[](const int i)
      const
    {
      return
	data[i];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(operator[]);
    
    /// Multiply another array
    template <typename U,
	      typename R=decltype(T()*U())>
    CUDA_HOST_DEVICE
    auto operator*(const ArithmeticArray<U,N>& oth) const
    {
      /// Result
      ArithmeticArray<R,N> out={};
      
      for(int i=0;i<N;i++)
	out[i]+=(*this)[i]*oth[i];
      
      return
	out;
    }
    
    /// Summassign another array
    template <typename U>
    CUDA_HOST_DEVICE
    auto operator+=(const ArithmeticArray<U,N>& oth)
    {
      for(int i=0;i<N;i++)
	(*this)[i]+=oth[i];
      
      return
	*this;
    }
    
    /// Subtassign another array
    template <typename U>
    CUDA_HOST_DEVICE
    auto operator-=(const ArithmeticArray<U,N>& oth)
    {
      for(int i=0;i<N;i++)
	(*this)[i]-=oth[i];
      
      return *this;
    }
  };
  
  /// A matrix
  template <typename T,
	    int N>
  struct ArithmeticMatrix
  {
    /// Internal data
    T data[N][N];
    
    /// Access to internal data
    INLINE_FUNCTION const T& get(const int i,const int j) const
    {
      return data[i][j];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(get);
    
    /// Multiply another matrix
    template <typename U,
	      typename R=decltype(T()*U())>
    auto operator*(const ArithmeticMatrix<U,N>& oth) const
    {
      /// Result
      ArithmeticMatrix<R,N> out={};
      
      UNROLLED_FOR(ir,N)
	UNROLLED_FOR(i,N)
	  UNROLLED_FOR(ic,N)
	    out[ir][ic]+=(*this)[ir][i]*oth[i][ic];
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      return out;
    }
    
    /// Summassign another matrix
    template <typename U>
    auto operator+=(const ArithmeticMatrix<U,N>& oth)
    {
      ASM_BOOKMARK("Matrix sum begin");
      
      UNROLLED_FOR(ir,N)
	UNROLLED_FOR(ic,N)
	  (*this)[ir][ic]+=oth[ir][ic];
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      ASM_BOOKMARK("Matrix sum end");
      
      return *this;
    }
    
    PROVIDE_ASM_DEBUG_HANDLE(sumProdMethod,double)
    PROVIDE_ASM_DEBUG_HANDLE(sumProdMethod,float)
    
    /// Sum the product between two another matrices
    template <typename U1,
	      typename U2,
	      typename R=decltype(T()+=U1()*U2())>
    INLINE_FUNCTION auto& sumProd(const ArithmeticMatrix<U1,N>& oth1,const ArithmeticMatrix<U2,N>& oth2)
    {
      /// To get assembly report for float and double separately
      using Fund=std::remove_reference_t<decltype(this->get(0,0).real[0])>;
      
      BOOKMARK_BEGIN_sumProdMethod(Fund{});
      
      UNROLLED_FOR(ir,N)
	UNROLLED_FOR(i,N)
	  UNROLLED_FOR(ic,N)
	    this->get(ir,ic).sumProd(oth1.get(ir,i),oth2.get(i,ic));
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      BOOKMARK_END_sumProdMethod(Fund{});
      
      return *this;
    }
  };
}

#endif
