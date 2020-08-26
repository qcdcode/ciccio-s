#ifndef _TENS_SLICE_HPP
#define _TENS_SLICE_HPP

/// \file tensSlice.hpp
///
/// \brief Implements a sliced view of a tensor

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <expr/expr.hpp>
#include <tensors/complSubscribe.hpp>
#include <tensors/component.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/tensDecl.hpp>
#include <tensors/tensFeat.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  DEFINE_FEATURE(IsTensSlice);
  
  DEFINE_FEATURE_GROUP(TensSliceFeat);
  
  /// Sliced view of a tensor
  ///
  /// Forward implementation
  template <bool Const,    // Const or not
	    typename T,    // Tensor
	    typename S>    // Subscribed componenents
  struct TensSlice;
  
  /// Shortcut for TensSlice
#define THIS					\
  TensSlice<Const,T,TensComps<Sc...>>
  
  /// Sliced view of a tensor
  template <bool Const,    // Const or not
	    typename T,    // Tensor
	    typename...Sc> // Subscribed components
  struct THIS : public
    Expr<THIS>,
    ComplexSubscribe<THIS>,
    TensSliceFeat<IsTensSlice,THIS>
  {
    /// A slice can be copied easily
    static constexpr bool takeAsArgByRef=
      false;
    
    /// Holds info on whether the slice is constant
    static constexpr bool IsConst=
      Const;
    
    /// A slice can be assigned provided is not const
    static constexpr bool canBeAssigned=
      not IsConst;
    
    /// Import assignement from Expr class
    using Expr<THIS>::operator=;
    
    /// Fundamental type
    using Fund=
      typename T::Fund;
    
    /// Original tensor to which we refer
    using OrigTens=
      T;
    
    /// Original components
    using OrigComps=
      typename T::Comps;
    
    /// Subscribed components
    using SubsComps=
      TensComps<Sc...>;
    
    /// Visible components
    using Comps=
      TupleFilterOut<SubsComps,OrigComps>;
    
    /// Reference to original tensor
    const OrigTens& t;
    
    /// Subscribed components
    const SubsComps subsComps;
    
    /// Get components size from the tensor
    template <typename C>
    INLINE_FUNCTION constexpr
    decltype(auto) compSize() const
    {
      return
	this->t.template compSize<C>();
    }
    
    /// Provide subscribe operator when returning a reference
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR,CONST_AS_BOOL)		\
    /*! Operator to take a const slice to a given component */		\
    template <typename C,						\
	      typename Cp=Comps,					\
	      ENABLE_THIS_TEMPLATE_IF((std::tuple_size<Cp>::value>1) and \
				      TupleHasType<C,Cp>)>		\
    CUDA_HOST_DEVICE INLINE_FUNCTION					\
    auto operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR	\
    {									\
									\
      /*! Nested reference subscribed components */			\
      auto nestedSubsComps=						\
	std::tuple_cat(subsComps,TensComps<C>{cFeat.deFeat()});		\
									\
      /*! Type used to hold all components */				\
      using NestedSubsComps=						\
	decltype(nestedSubsComps);					\
    									\
      /*! Reference type */						\
      using R=								\
	TensSlice<CONST_AS_BOOL or IsConst,T,NestedSubsComps>;		\
      									\
      return								\
	R(this->t,nestedSubsComps);					\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */, false);
    PROVIDE_SUBSCRIBE_OPERATOR(const, true);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Provide subscribe operator when returning direct access
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR)				\
    /*! Operator to return direct access to data */			\
    template <typename C,						\
	      typename Cp=Comps,					\
	      ENABLE_THIS_TEMPLATE_IF(std::tuple_size<Cp>::value==1 and	\
				 TupleHasType<C,Cp>)>			\
    CUDA_HOST_DEVICE INLINE_FUNCTION					\
    ConstIf<IsConst,Fund>& operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR \
    {									\
      return								\
	(ConstIf<IsConst,Fund>&)t.trivialAccess(t.index(std::tuple_cat(subsComps,std::make_tuple(cFeat.deFeat())))); \
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Create from slice and list of subscribed components
    CUDA_HOST_DEVICE
    TensSlice(const TensFeat<IsTens,T>& t,
	      const SubsComps& subsComps) :
      t(t.deFeat()),subsComps(subsComps)
    {
    }
    
    /// Return a tensor pointing to the offsetted data, with the resulting component
    CUDA_HOST_DEVICE INLINE_FUNCTION constexpr
    auto carryOver() const
    {
      /// Number of dynamic components must be zero
      constexpr int nDynComps=
	std::tuple_size<TupleFilter<SizeIsKnownAtCompileTime<false>::t,Comps>>::value;
      
      static_assert(nDynComps==0,"Not supported if residual dynamic components are present");
      
      /// Offset to data
      const auto offset=
	t.index(fillTuple<OrigComps>(subsComps));
      
      /// Data with offset
      auto carriedData=
	t.getDataPtr()+
	offset;
      
      return
	Tens<Comps,typename T::Fund,T::storLoc,Stackable::CANNOT_GO_ON_STACK>(carriedData,t.data.getSize());
    }
  };
  
#undef THIS
}

#endif
