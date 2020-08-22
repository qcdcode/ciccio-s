#ifndef _PRODUCT_HPP
#define _PRODUCT_HPP

/// \file expr/product.hpp
///
/// \brief Implements product of expressions

#include <expr/expr.hpp>
#include <tensors/component.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/tensor.hpp>
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
  
  /// Product of two expressions
  template <typename F1,
	    typename F2,
	    typename ExtComps=typename impl::ProductComps<typename F1::Comps,typename F2::Comps>::Comps>
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
  
  template <bool B,
	    typename T,
	    typename F1,
	    typename F2>
  struct ProductFundCastProvider;
  
  template <typename T,
	    typename F1,
	    typename F2>
  struct ProductFundCastProvider<false,T,F1,F2>
  {
  };
  
  template <typename T,
	    typename F1,
	    typename F2>
  struct ProductFundCastProvider<true,T,F1,F2>
  {
    /// Resulting fundamental type
    using Fund=
      std::common_type_t<typename F1::Fund,
			 typename F2::Fund>;
    
    INLINE_FUNCTION explicit
    operator Fund()
      const
    {
      
    }
  };
  
  /// Product of two expressions
  template <typename F1,
	    typename F2,
	    typename ExtComps>
  struct Product :
    Expr<Product<F1,F2>>,
    ProductFundCastProvider<std::tuple_size<ExtComps>::value==0,Product<F1,F2,ExtComps>,F1,F2>
  {
    /// First expression
    const F1& f1;
    
    /// Second expression
    const F2& f2;
    
    /// Resulting fundamental type
    using Fund=
      std::common_type_t<typename F1::Fund,
			 typename F2::Fund>;
    
    /// Resulting components
    using Comps=
      ExtComps;
    
    /// Construct taking two expressions
    Product(const F1& f1,
	    const F2& f2)
      : f1(f1),f2(f2)
    {
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
    
    // auto close()
    //   const
    // {
    // 	Tens<Comps,Fund> a;
	
    // 	return a;
    // }
  };
}

#endif
