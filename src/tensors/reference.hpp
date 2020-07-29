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
  template <typename T, // Tensor
	    typename C> // Visible componenents
  struct TensRef;
  
  /// Shortcut for TensRef
#define THIS					\
  TensRef<T,TensComps<Vc...>>
  
  /// Reference to a Tensor
  template <typename T,    // Tensor
	    typename...Vc> // Visible components
  struct THIS : public
  TensRefFeat<IsTensRef,THIS>,
    TensRefFeat<std::conditional_t<(sizeof...(Vc)>1),ReturnsRefWhenSliced,ReturnsDataWhenSliced>,THIS>
  {
    /// Original tensor to which we refer
    using OrigTens=T;
    
    /// Fundamental data type
    using Fund=
      typename T::Fund;
    
    /// Original components
    using OrigComps=
      typename T::Comps;
    
    /// Visible components
    using Comps=
      TensComps<Vc...>;
    
    /// Reference to original quantity
    const T& t;
    
    /// Returns the reference
    const T& deRef() const
    {
      return t;
    }
    
    /// Storage for the data
    const Fund* const data;
    
    /// Returns the pointer to data, which incorporates possible offsets w.r.t t.data
    decltype(auto) getDataPtr() const
    {
      return data;
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(getDataPtr);
    
    // /// Offset to access to data
    // template <typename C>
    // auto computeShiftOfComp(const TensCompFeat<IsTensComp,C>& c) const
    // {
    //   return
    // 	t.computeShiftOfComp(c.defeat());
    // }
    
    /// Constructor taking the original object as a reference
    ///
    /// Note that data may contains offsets
    TensRef(const TensFeat<IsTens,T>& t,
	    const Fund* data) : t(t.defeat()),data(data)
    {
    }
  };
  
  #undef THIS
  
  /////////////////////////////////////////////////////////////////
  
  /// Returns the same than the passed argument
  template <typename T>
  decltype(auto) deRef(const TensRefFeat<IsTensRef,T>& rFeat)
  {
    return rFeat.defeat().deRef();
  }
  
  // template <typename T>
  // auto ref(TensFeat<IsTens,T>& t)
  // {
  //   return TensRef<T,TensComps<>>(t,t.data);
  // }
  
  template <typename T,
	    typename C,
	    template <typename,typename> class FG>
  auto ref(const FG<ReturnsRefWhenSliced,T>& tFeat,
	   const TensCompFeat<IsTensComp,C>& cFeat)
  {
    auto& t=
      tFeat.defeat();
    
    auto& base=
      deRef(t);
    
    using Bt=
      std::decay_t<decltype(base)>;
    
    using Vc=
      TupleFilterOut<TensComps<C>,typename T::Comps>;
    
    auto shift=
      base.computeShiftOfComp(cFeat.defeat());
    
    return TensRef<Bt,Vc>(base,t.getDataPtr()+shift);
  }
  
  template <typename T,
	    typename C,
	    template <typename,typename> class FG>
  decltype(auto) ref(const FG<ReturnsDataWhenSliced,T>& tFeat,
		     const TensCompFeat<IsTensComp,C>& cFeat)
  {
    auto& t=
      tFeat.defeat();
    
    auto& base=
      deRef(t);
    
    auto shift=
      base.computeShiftOfComp(cFeat.defeat());
    
    return t.getDataPtr()[shift];
  }
  
}

#endif
