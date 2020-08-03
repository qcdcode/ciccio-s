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
	    typename O,    // Offset
	    typename C>    // Visible componenents
  struct TensRef;
  
  /// Shortcut for TensRef
#define THIS					\
  TensRef<Const,T,O,TensComps<Vc...>>
  
  /// Reference to a Tensor
  template <bool Const,    // Const or not
	    typename T,    // Tensor
	    typename O,    // Offset
	    typename...Vc> // Visible components
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
    
    /// Visible components
    using Comps=
      TensComps<Vc...>;
    
    /// Reference to original tensor
    const OrigTens& t;
    
    /// Returns the reference
    OrigTens& deRef() const
    {
      return t;
    }
    
    /// Offset of the data
    const O offs;
    
    /// Returns the pointer to data, which incorporates possible offsets w.r.t t.data
    // decltype(auto) getDataPtr() const
    // {
    //   return data;
    // }
    
    // PROVIDE_ALSO_NON_CONST_METHOD(getDataPtr);
    
    /// Provide subscribe operator when returning a reference
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR,CONST_AS_BOOL)		\
    /*! Operator to take a const reference to a given component */	\
    template <typename C,						\
	      SFINAE_ON_TEMPLATE_ARG(sizeof...(Vc)>1  and TupleHasType<C,Comps>)> \
    auto operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR	\
    {									\
      /* Residual components */						\
      using RefComps=							\
	TupleFilterOut<TensComps<C>,Comps>;				\
									\
      /*! Nested offset */						\
      const auto no=							\
	offs+t.computeShiftOfComp(cFeat);				\
      									\
      /*! Offset */							\
      using NO=								\
	decltype(no);							\
      									\
      return								\
	TensRef<CONST_AS_BOOL or IsConst,T,NO,RefComps>(this->t,no);	\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */, false);
    PROVIDE_SUBSCRIBE_OPERATOR(const, true);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Provide subscribe operator when returning direct access
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR)				\
    /*! Operator to return direct access to data */			\
    template <typename C,						\
	      SFINAE_ON_TEMPLATE_ARG(sizeof...(Vc)==1 and TupleHasType<C,Comps>)> \
    CONST_ATTR auto& operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR \
    {									\
      return								\
	t.getDataPtr()[offs+t.computeShiftOfComp(cFeat)];		\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    // /// Offset to access to data
    // template <typename C>
    // auto computeShiftOfComp(const TensCompFeat<IsTensComp,C>& c) const
    // {
    //   return
    // 	t.computeShiftOfComp(c.deFeat());
    // }
    
    /// Constructor taking the original object as a reference
    ///
    /// Note that data may contains offsets
    TensRef(const TensFeat<IsTens,T>& t,
	    const O offs) : t(t.deFeat()),offs(offs)
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
