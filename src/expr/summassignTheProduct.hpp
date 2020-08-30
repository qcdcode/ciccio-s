#ifndef _SUMMASSIGN_THE_PRODUCT_HPP
#define _SUMMASSIGN_THE_PRODUCT_HPP

/// \file expr/summassignTheProduct.hpp
///
/// \brief Implements sum of product of expressions

#include <dataTypes/SIMD.hpp>
#include <expr/product.hpp>

namespace ciccios
{
  /// Captures the sum-assign of a product
  template <typename S,
	    typename U1,
	    typename U2,
	    ENABLE_THIS_TEMPLATE_IF(nOfComps<U1> >0 or
				    nOfComps<U2> >0)>
  constexpr INLINE_FUNCTION
  auto operator+=(const Expr<S>& a,
		  const Product<U1,U2>& bc)
  {
    using F=
      std::tuple_element_t<0,typename U1::Comps>;
    
    const auto& _f1=bc.f1[F{0}];
    
    const auto& f1=
      *reinterpret_cast<const Simd<std::decay_t<decltype(_f1)>>*>(&_f1);
    
    const auto& _f2=bc.f2[F{0}];
    
    const auto& f2=
      *reinterpret_cast<const Simd<std::decay_t<decltype(_f2)>>*>(&_f2);
    
    auto& _t=a.deFeat()[F{0}];
    
    auto& t=
      *reinterpret_cast<Simd<std::decay_t<decltype(_t)>>*>(&_t);
    
    return
      t+=f1*f2;
  }
  
  /// Captures the sum-assign of a product
  template <typename S,
	    typename U1,
	    typename U2,
	    ENABLE_THIS_TEMPLATE_IF(nOfComps<U1> ==0 and
				    nOfComps<U2> ==0)>
  constexpr INLINE_FUNCTION
  auto operator+=(const Expr<S>& a,
		  const Product<U1,U2>& bc)
  {
    return
      a.deFeat().eval()+=bc.eval();
  }
  
}

#endif
