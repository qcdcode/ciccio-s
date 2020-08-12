#ifndef _EXPR_HPP
#define _EXPR_HPP

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  template <typename F1,
	    typename F2>
  struct Product;
  
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
    
    template <typename U>
    auto operator*(const Expr<U>& u) const
    {
      return
	Product<T,U>(this->deFeat(),u.deFeat());
    }
    
    template <typename U1,
	      typename U2>
    auto operator+=(const Product<U1,U2>& u) const
    {
      return
	this->deFeat();
    }
    
    template <typename U1,
	      typename U2>
    auto operator-=(const Product<U1,U2>& u) const
    {
      return
	*this;
    }
  };
  
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
