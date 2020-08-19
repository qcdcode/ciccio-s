#ifndef _PRODUCT_HPP
#define _PRODUCT_HPP

/// \file expr/product.hpp
///
/// \brief Implements product of expressions

#include <expr/expr.hpp>

namespace ciccios
{
  /// Product of two expressions
  template <typename F1,
	    typename F2>
  struct Product;
  
  /// Capture the product operator for two generic expressions
  template <typename U1,
	    typename U2>
  auto operator*(const Expr<U1>& u1, ///< Left of the product
		 const Expr<U2>& u2) ///> Right of the product
  {
    return
      Product<U1,U2>(u1.deFeat(),u2.deFeat());
  }
  
  template <typename F1,
	    typename F2>
  struct Product : Expr<Product<F1,F2>>
  {
    const F1& f1;
    const F2& f2;
    
    Product(const F1& f1,
	    const F2& f2)
      : f1(f1),f2(f2)
    {
    }
  };
}

#endif
