#ifndef _PRODUCT_HPP
#define _PRODUCT_HPP

/// \file expr/product.hpp
///
/// \brief Implements product of expressions

#include <expr/expr.hpp>
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
    template <typename F1,
	      typename F2>
    struct ProductComps
    {
      using Comps1=
	typename F1::Comps;
      
      using Comps2=
	typename F2::Comps;
      
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
	    typename ExtComps=typename impl::ProductComps<F1,F2>::Comps,
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
    /// Contract inner indices of a product
    ///
    /// Forward declaration
    template <typename...>
    struct _ProductContracter;
    
    /// Contract inner indices of a product
    ///
    /// Case in which no more components are left
    template <>
    struct _ProductContracter<TensComps<>>
    {
      /// Contract the actual data
      template <typename T,
		typename F1,
		typename F2>
      static constexpr INLINE_FUNCTION CUDA_HOST_DEVICE
      void eval(T& out,
    		const F1& f1,
    		const F2& f2)
      {
    	out+=
    	  f1*f2;
      }
    };
    
    /// Contract inner indices of a product
    ///
    /// Contract component Head
    template <typename Head,
	      typename...Tail>
    struct _ProductContracter<TensComps<Head,Tail...>>
    {
      /// Evaluate the contruction
      template <typename T,
		typename F1,
		typename F2>
      static constexpr INLINE_FUNCTION CUDA_HOST_DEVICE
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
    ToFundCastProvider<CanBeCastToFund,THIS,ExtFund>
  {
    /// Product is simple to create
    static constexpr bool takeAsArgByRef=
      false;
    
    /// Product cannot be assigned
    static constexpr bool canBeAssigned=
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
    
    /// Return evalutation of the product, valid only if no free component is present
    template <typename F=Fund>
    constexpr INLINE_FUNCTION CUDA_HOST_DEVICE
    F eval()
      const
    {
      /// Result
      F out=
	0;
      
      /// Components to contract
      using ContractedComps=
	typename impl::ProductComps<F1,F2>::ContractedComps;
      
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
    
    /// Check if the free components of factor1 contain Tc
    template <typename Tc>
    static constexpr bool firstOperandHasFreeComp=
      TupleHasType<Tc,
		   typename impl::ProductComps<F1,F2>::F1FilteredContractedComps>;
    
    /// Check if the free components of factor2 contain Tc
    template <typename Tc>
    static constexpr bool secondOperandHasFreeComp=
      TupleHasType<Tc,
		   typename impl::ProductComps<F1,F2>::F2FilteredContractedComps>;
    
    /// Returns the real part
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    decltype(auto) real()
      const
    {
      return
	f1[RE]*f2[RE]-f1[IM]*f2[IM];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(real);
    
    /// Returns the imaginary part
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    decltype(auto) imag()
      const
    {
      return
	f1[RE]*f2[IM]+f1[IM]*f2[RE];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(imag);
    
    /// Subscribe a component present in both factors
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF(firstOperandHasFreeComp<Tc>),
	      ENABLE_THIS_TEMPLATE_IF(secondOperandHasFreeComp<Tc>)>
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto operator[](const Tc& tc)
      const
    {
      return
	f1[tc]*f2[tc];
    }
    
    /// Subscribe a component present in second factors
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF(not firstOperandHasFreeComp<Tc>),
	      ENABLE_THIS_TEMPLATE_IF(secondOperandHasFreeComp<Tc>)>
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto operator[](const Tc& tc)
      const
    {
      return
	f1*f2[tc];
    }
    
    /// Subscribe a component present in first factors
    template <typename Tc,
	      ENABLE_THIS_TEMPLATE_IF(firstOperandHasFreeComp<Tc>),
	      ENABLE_THIS_TEMPLATE_IF(not secondOperandHasFreeComp<Tc>)>
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    auto operator[](const Tc& tc)
      const
    {
      return
	f1[tc]*f2;
    }
  };
  
#undef THIS
}

#endif
