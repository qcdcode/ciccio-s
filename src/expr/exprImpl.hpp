#ifndef _EXPR_EXPR_IMPL_HPP
#define _EXPR_EXPR_IMPL_HPP

/// \file expr/exprImpl.hpp
///
/// \brief Implements base expression

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <dataTypes/SIMD.hpp>
#include <expr/assign.hpp>
#include <tensors/tens.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  /// Base expression
  template <typename T>
  struct Expr
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    /// Assign to an expression
    template <typename U// ,
	      // ENABLE_THIS_TEMPLATE_IF(std::is_same<typename T::Comps,typename U::Comps>::value)
	      >
    void operator=(const Expr<U>& u)
    {
      assign(*this,u,(typename U::Comps*)nullptr);
    }
  };
}

#endif
