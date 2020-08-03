#ifndef _REFERENCE_HPP
#define _REFERENCE_HPP

/// \file reference.hpp
///
/// \brief Implements a reference to a tensor

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <tensors/component.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/tensFeat.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  DEFINE_FEATURE(IsTensRef);
  
  DEFINE_FEATURE_GROUP(TensRefFeat);
  
  /// Reference to a Tensor
  ///
  /// Forward implementation
  template <bool Const,    // Const or not
	    typename T,    // Tensor
	    typename S>    // Subscribed componenents
  struct TensRef;
  
  /// Shortcut for TensRef
#define THIS					\
  TensRef<Const,T,TensComps<Sc...>>
  
  /// Reference to a Tensor
  template <bool Const,    // Const or not
	    typename T,    // Tensor
	    typename...Sc> // Subscribed components
  struct THIS : public
    TensRefFeat<IsTensRef,THIS>
  {
    /// Holds info on whether the reference is constant
    static constexpr bool IsConst=
      Const;
    
    /// Original tensor to which we refer
    using OrigTens=T;
    
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
    
    /// Returns the reference
    OrigTens& deRef() const
    {
      return t;
    }
    
    /// Provide subscribe operator when returning a reference
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR,CONST_AS_BOOL)		\
    /*! Operator to take a const reference to a given component */	\
    template <typename C,						\
	      typename Cp=Comps,					\
	      SFINAE_ON_TEMPLATE_ARG((std::tuple_size<Cp>::value>1) and \
				     TupleHasType<C,Cp>)>		\
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
      TensRef<CONST_AS_BOOL or IsConst,T,NestedSubsComps>;		\
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
	      SFINAE_ON_TEMPLATE_ARG(std::tuple_size<Cp>::value==1 and \
				     TupleHasType<C,Cp>)>		\
    CONST_ATTR auto& operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR \
    {									\
      return								\
	t.getDataPtr()[t.index(std::tuple_cat(subsComps,std::make_tuple(cFeat.deFeat())))]; \
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Create from reference and list of subscribed components
    TensRef(const TensFeat<IsTens,T>& t,
	    const SubsComps& subsComps) : t(t.deFeat()),subsComps(subsComps)
    {
    }
  };
  
  #undef THIS
  
  /////////////////////////////////////////////////////////////////
  
  /// Provides a constant or not dereferencing
#define PROVIDE_DEREF(ATTR)					\
  /*! Returns the referred tensor */				\
  template <typename T>						\
  decltype(auto) deRef(ATTR TensRefFeat<IsTensRef,T>& rFeat)	\
  {								\
    return rFeat.deFeat().deRef();				\
  }
  
  PROVIDE_DEREF();
  PROVIDE_DEREF(const);
  
#undef PROVIDE_DEREF
  
}

#endif
