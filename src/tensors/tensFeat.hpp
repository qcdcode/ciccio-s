#ifndef _TENSFEAT_HPP
#define _TENSFEAT_HPP

/// \file tensFeat.hpp
///
/// \brief Implements the feature to detect a tensor

#include <base/feature.hpp>
#include <base/metaProgramming.hpp>

namespace ciccios
{
  DEFINE_FEATURE(IsTens);
  
  DEFINE_FEATURE_GROUP(TensFeat);
}

#endif
