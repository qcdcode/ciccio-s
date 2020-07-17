#ifndef _BASE_TENS_HPP
#define _BASE_TENS_HPP

/// \file baseTens.hpp
///
/// \brief Implements the basic signature for a tensor

#include <base/metaProgramming.hpp>

namespace ciccios
{
  /// Base type to detect a tensor
  template <typename T>
  struct BaseTens : public Crtp<T>
  {
  };
}

#endif
