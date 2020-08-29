#ifndef _EXPR_EXPR_HPP
#define _EXPR_EXPR_HPP

/// \file expr/expr.hpp
///
/// \brief Implements base expression

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <dataTypes/SIMD.hpp>
#include <expr/assign.hpp>
#include <tensors/tensDecl.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  /// Base expression
  template <typename T>
  struct Expr
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    // /// Assign to an expression
    // template <typename U>
    // INLINE_FUNCTION CUDA_HOST_DEVICE
    // void operator=(const Expr<U>& u)
    // {
    //   assign(*this,u,(typename U::Comps*)nullptr);
    // }
    
    /// Assign to an expression
    template <typename U>
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    const U& operator=(const Expr<U>& u)
    {
      assign(*this,u.deFeat(),(typename T::Comps*)nullptr);
      
      return
	u.deFeat();
    }
    
    // Feature used to signal to close to tens
    DEFINE_FEATURE(TO_TENS);
    
    // Feature used to signal to close to fundamental type
    DEFINE_FEATURE(TO_FUND);
    
    /// Closes the expression
    ///
    /// Returns a tensor with the components of the expression
    /// \todo Dynamic case to be implemented
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto _close(TO_TENS)
      const
    {
      /// Result to be returned
      Tens<typename T::Comps,typename T::Fund> a;
      
      a=
	this->deFeat();
	
      return
	a;
    }
    
    /// Closes the expression
    ///
    /// Returns the fundamental value arising from closing the expression
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto _close(TO_FUND)
      const
    {
      return
	static_cast<typename T::Fund>(this->deFeat());
    }
    
    /// Closes the expression
    ///
    /// Returns a tensor with the components of the expression
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto close()
      const
    {
      /// Check how to close
      constexpr bool close=
       nOfComps<T> >0;
      
      return
	this->_close(std::conditional_t<close,TO_TENS,TO_FUND>());
    }
  };
  
  /// Provide a function which casts to the fundamental type
  ///
  /// Forward declaration
  template <bool B,
	    typename T,
	    typename ExtFund>
  struct ToFundCastProvider;
  
  /// Provide a function which casts to the fundamental type
  ///
  /// Does not provide the access
  template <typename T,
	    typename ExtFund>
  struct ToFundCastProvider<false,T,ExtFund>
  {
  };
  
  /// Provide a function which casts to the fundamental type
  ///
  /// Provides the actual access
  template <typename T,
	    typename ExtFund>
  struct ToFundCastProvider<true,T,ExtFund>
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    /// Provide the cast to fundamental
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    operator ExtFund()
      const
    {
      return
    	this->deFeat().eval();
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// Provide a function which assign from fund
  ///
  /// Forward declaration
  template <bool B,
	    typename T,
	    typename ExtFund>
  struct AssignFromFundProvider;
  
  /// Provide a function which assign from fund
  ///
  /// Does not provide the assignment operator
  template <typename T,
	    typename ExtFund>
  struct AssignFromFundProvider<false,T,ExtFund>
  {
  };
  
  /// Provide a function which assign from fund
  ///
  /// Provides the actual assignment operator
  template <typename T,
	    typename ExtFund>
  struct AssignFromFundProvider<true,T,ExtFund>
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    /// Provides the assignment from fund
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    const ExtFund& operator=(const ExtFund& oth)
    {
      return
    	static_cast<ExtFund&>(this->deFeat())=
	oth;
    }
  };
}

#endif
