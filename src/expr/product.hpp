#ifndef _PRODUCT_HPP
#define _PRODUCT_HPP

/// \file expr/product.hpp
///
/// \brief Implements product of expressions

#include <expr/exprImpl.hpp>
#include <expr/exprArg.hpp>
#include <tensors/component.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/tens.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  namespace impl
  {
    /// Computes the resulting components
    template <typename Comps1,
	      typename Comps2>
    struct ProductComps
    {
      /// Gets col components from 1
      using Cln1=
	TensCompsFilterCln<Comps1>;
      
      /// Gets row components from 2
      using Row2=
	TensCompsFilterRow<Comps2>;
      
      /// List contracted components
      using ContractedComps=
	 TupleCommonTypes<TensCompsTransp<Cln1>,Row2>;
      
      /// Filter out the contracted components from Comps1
      using F1FilteredContractedComps=
      	TupleFilterOut<TensCompsTransp<ContractedComps>,Comps1>;
      
      /// Filter out the contracted components from Comps2
      using F2FilteredContractedComps=
      	TupleFilterOut<ContractedComps,Comps2>;
      
      /// Get unique types of F2Filtered not contained in F1Filtered
      using F2UniqueFilteredComps=
      	TupleFilterOut<F1FilteredContractedComps,F2FilteredContractedComps>;
      
      /// Merge components catting them
      using MergedComps=
      	TupleCat<F1FilteredContractedComps,F2UniqueFilteredComps>;
      
      /// Number of merged components
      static constexpr int nMergedComps=
	std::tuple_size<MergedComps>::value;
      
      /// Check if we can rearrange components as Comps1
      static constexpr bool CanBeRearrangedAs1=
	std::tuple_size<TupleCommonTypes<Comps1,MergedComps>>::value==nMergedComps and
	std::tuple_size<Comps1>::value==nMergedComps;
      
      /// Check if we can rearrange components as Comps2
      static constexpr bool CanBeRearrangedAs2=
	std::tuple_size<TupleCommonTypes<Comps2,MergedComps>>::value==nMergedComps and
	std::tuple_size<Comps1>::value==nMergedComps;
      
      /// Resulting components is Comps1, Comps2, or MergedComps if
      /// none of the firsts is a reordered version of the latter
      ///
      /// \todo Improve
      using Comps=
	std::conditional_t<CanBeRearrangedAs1,Comps1,
			   std::conditional_t<CanBeRearrangedAs2,Comps2,MergedComps>>;
    };
  }
  
  template <bool B,
	    typename T,
	    typename ExtFund>
  struct ProductFundCastProvider;
  
  template <typename T,
	    typename ExtFund>
  struct ProductFundCastProvider<false,T,ExtFund>
  {
  };
  
  template <typename T,
	    typename ExtFund>
  struct ProductFundCastProvider<true,T,ExtFund>
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    INLINE_FUNCTION
    operator ExtFund()
      const
    {
      return
    	this->deFeat().eval();
    }
  };
  
  /// Product of two expressions
  template <typename F1,
	    typename F2,
	    typename ExtComps=typename impl::ProductComps<typename F1::Comps,typename F2::Comps>::Comps,
	    typename ExtFund=std::common_type_t<typename F1::Fund,typename F2::Fund>,
	    bool CanBeCastToFund=std::tuple_size<ExtComps>::value==0>
  struct Product;
  
  /// Capture the product operator for two generic expressions
  template <typename U1,
	    typename U2,
	    ENABLE_THIS_TEMPLATE_IF(nOfComps<U1> >0 or
				    nOfComps<U2> >0)>
  auto operator*(const Expr<U1>& u1, ///< Left of the product
		 const Expr<U2>& u2) ///> Right of the product
  {
    return
      Product<U1,U2>(u1.deFeat(),u2.deFeat());
  }
  
  namespace impl
  {
    template <typename...>
    struct _ProductContracter;
    
    template <>
    struct _ProductContracter<TensComps<>>
    {
      template <typename T>
      static constexpr INLINE_FUNCTION
      void eval(T& out,
    		const T& f1,
    		const T& f2)
      {
    	out+=
    	  f1*f2;
      }
    };
    
    template <typename Head,
	      typename...Tail>
    struct _ProductContracter<TensComps<Head,Tail...>>
    {
      template <typename T,
		typename F1,
		typename F2>
      static constexpr INLINE_FUNCTION
      void eval(T& out,
		const F1& f1,
		const F2& f2)
      {
	for(Head i{0};i<f2.template compSize<Head>();i++)
	  _ProductContracter<TensComps<Tail...>>::eval(out,f1[i.transp()],f2[i]);
      }
    };
  }
  
#define THIS					\
  Product<F1,F2,ExtComps,ExtFund,CanBeCastToFund>
  
  /// Product of two expressions
  template <typename F1,
	    typename F2,
	    typename ExtComps,
	    typename ExtFund,
	    bool CanBeCastToFund>
  struct Product :
    Expr<THIS>,
    ProductFundCastProvider<CanBeCastToFund,THIS,ExtFund>
  {
    /// Product is simple to create
    static constexpr bool takeAsArgByRef=
      false;
    
    /// First expression
    ExprArg<F1 const> f1;
    
    /// Second expression
    ExprArg<F2 const> f2;
    
    /// Resulting fundamental type
    using Fund=
      ExtFund;
    
    /// Resulting components
    using Comps=
      ExtComps;
    
    template <typename F=Fund>
    INLINE_FUNCTION
    F eval()
      const
    {
      F out=
	0;
      
      using ContractedComps=
	typename impl::ProductComps<typename F1::Comps,typename F2::Comps>::ContractedComps;
    
      impl::_ProductContracter<ContractedComps>::eval(out,f1,f2);
      
      return
	out;
    }
    
    /// Construct taking two expressions
    Product(const F1& f1,
	    const F2& f2)
      : f1(f1),f2(f2)
    {
    }
    
    template <typename Tc>
    static constexpr bool firstOperandHasFreeComp=
      TupleHasType<Tc,
		   typename impl::ProductComps<typename F1::Comps,typename F2::Comps>::F1FilteredContractedComps>;
    
    template <typename Tc>
    static constexpr bool secondOperandHasFreeComp=
      TupleHasType<Tc,
		   typename impl::ProductComps<typename F1::Comps,typename F2::Comps>::F2FilteredContractedComps>;
      
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF(firstOperandHasFreeComp<Tc> and
				      secondOperandHasFreeComp<Tc>)>
    auto operator[](const Tc& tc)
    {
      return
	f1[tc]*f2[tc];
    }
    
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF((not firstOperandHasFreeComp<Tc>) and
				      secondOperandHasFreeComp<Tc>)>
    auto operator[](const Tc& tc)
    {
      return
	f1*f2[tc];
    }
    
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF(firstOperandHasFreeComp<Tc> and
				      not secondOperandHasFreeComp<Tc>)>
    auto operator[](const Tc& tc)
    {
      return
	f1[tc]*f2;
    }
    
    //va aggiunto l'operatore sottoscrizione, che prende la componente
    //dal primo operando se e' riga, dal secondo se e' colonna, o da
    //tutt e due se e' any. Quando non ci sono piu componenti
    //disponibili, la contrazione degli indici colonna/riga avviene.
    //Quando si assegna, l'operatore di assegnazione procedera' a
    //fissare i vari indici dell'output. Poi un giorno si puo'
    //scrivere anche l'assegnazione di un prodotto, che ciclera'
    //diversamente
    
    /// Occorre far si che il costruttore di copie di un tensore
    /// effettui effettivamente una copia, e che anche gli stack e
    /// dynamic storage lo facciano, e facciamo si che importare in un
    /// kernel sia esplicito
    
    /// Close the product
    ///
    /// \todo move to expr
    auto close()
      const
    {
    	Tens<Comps,Fund> a;
	
	static_cast<Expr<Tens<Comps,Fund>>>(a)=
	  static_cast<const Expr<THIS>&>(*this);
	
    	return
	  a;
    }
  };
  
#undef THIS
}

#endif
