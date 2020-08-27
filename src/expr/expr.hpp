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
    INLINE_FUNCTION CUDA_HOST_DEVICE
    void operator=(const U& t)
    {
      assign(*this,t,(typename T::Comps*)nullptr);
    }
    
    /// Closes the expression
    ///
    /// Returns a tensor with the components of the expression
    /// \todo Dynamic case to be implemented
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto close()
      const
    {
      /// Result to be returned
      Tens<typename T::Comps,typename T::Fund> a;
      
      a=
	this->deFeat();
	
      return
	a;
    }
    
  };
}

#endif
