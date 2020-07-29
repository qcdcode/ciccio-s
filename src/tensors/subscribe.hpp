#ifndef _SUBSCRIBE_HPP
#define _SUBSCRIBE_HPP

/// \file subscribe.hpp
///
/// \brief Implements the [] operator

#include <tensors/component.hpp>

namespace ciccios
{
  /// Implements the [] operator
  template <typename T>
  struct Subscribable
  {
    /// Operator to take a reference to a given component
    template <typename C>
    decltype(auto) operator[](const TensCompFeat<IsTensComp,C>& cFeat)
    {
      return
	ref(this->defeat(),cFeat);
    }
    
    PROVIDE_DEFEAT_METHOD(T);
  };
}

#endif
