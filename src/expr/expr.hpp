#ifndef _EXPR_EXPR_HPP
#define _EXPR_EXPR_HPP

/// \file expr/expr.hpp
///
/// \brief Implements base expression

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <dataTypes/SIMD.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  template <typename T>
  struct Expr
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    template <typename Head,
	      typename...Tail,
	      typename A,
	      typename B>
    static void assign(Expr<A>& a,
		       const Expr<B>& b,
		       TensComps<Head,Tail...>*)
    {
      for(Head i{0};i<a.deFeat().template compSize<Head>();i++)
      	a.deFeat()[i]=
      	  b.deFeat()[i];
    }
    
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
