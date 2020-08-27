#ifndef _ASSIGN_HPP
#define _ASSIGN_HPP

/// \file assign.hpp
///
/// \brief Implements assignment

#include <expr/exprDecl.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  /// Assign an expression, parsing one component
  ///
  /// /\todo reshape, add check on size
  template <typename Head,
	    typename...Tail,
	    typename A,
	    typename B>
  INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
  void assign(Expr<A>& a,
	      const Expr<B>& b,
	      TensComps<Head,Tail...>*)
  {
    for(Head i{0};i<a.deFeat().template compSize<Head>();i++)
      a.deFeat()[i]=
	b.deFeat()[i];
  }
  
  /// Assign an expression, when the component is a complex
  ///
  /// /\todo reshape, add check on size
  template <typename...Tail,
	    typename A,
	    typename B>
  INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
  void assign(Expr<A>& a,
	      const Expr<B>& b,
	      TensComps<Compl,Tail...>*)
  {
    a.deFeat().real()=
      b.deFeat().real();
    
    a.deFeat().imag()=
      b.deFeat().imag();
  }
}

#endif