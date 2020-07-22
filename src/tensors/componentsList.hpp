#ifndef _COMPONENTS_LIST_HPP
#define _COMPONENTS_LIST_HPP

#include <tuple>

/// \file componentsList.hpp
///
/// \brief Implements a list of components
///
/// \todo Extend or incorporate into another file

namespace ciccios
{
  /// Collection of components
  template <typename...Tc>
  using TensComps=std::tuple<Tc...>;
  
  /// Returns the number of components of a tensComp
  template <typename T>
  constexpr int nOfComps=
    std::tuple_size<typename T::Comps>::value;
}

#endif
