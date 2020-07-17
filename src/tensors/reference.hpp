#ifndef _REFERENCE_HPP
#define _REFERENCE_HPP

/// \file reference.hpp
///
/// \brief Implements a reference to a tensor

#include <tensors/tensor.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  /// Reference to a Tensor
  ///
  /// Forward implementation
  template <typename T, // Tensor
	    typename S> // Subscribed componets
  struct TensRef;
  
  /// Reference to a Tensor
  template <typename T, // Tensor
	    typename...Sp> // Subscribed componets
  struct TensRef<T,TensComps<Sp...>>
  {
    /// Original tensor to which we refer
    using OrigTens=T;
    
    /// Fundamental data type
    using Fund=
      typename T::Fund;
    
    /// Original components
    using OrigComps=
      typename T::Comps;
    
    /// Subscribed components
    using SubscribedComps=
      TensComps<Sp...>;
    
    /// Visible components
    using Comps=
      TupleFilterOut<OrigComps, SubscribedComps>;
    
    /// Storage for the data
    Fund* const data;
    
    TensRef(T& t,
	    Fund* data) :data(data)
    {
      
    }
  };
  
  template <typename T>
  auto ref(BaseTens<T>& t)
  {
    return TensRef<T,TensComps<>>(t,t.data);
  }
  
  template <typename T,
	    typename C>
  auto ref(BaseTens<T>& _t,const TensCompSignature<C>& _c)
  {
    T& t=
      _t.crtp();
    
    const C& c=
      _c.crtp();
    
    const auto offset=
      t.template offset<C>();
    
    const auto shift=
      offset*c;
    
    return TensRef<T,TensComps<C>>(t,t.data.data.data+shift);
  }
}

#endif
