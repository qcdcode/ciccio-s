#ifndef _COMPONENTS_LIST_HPP
#define _COMPONENTS_LIST_HPP

#include <tuple>

namespace ciccios
{
  /// Collection of components
  template <typename...Tc>
  using TensComps=std::tuple<Tc...>;
  
  /// \todo Maybe we promote it to a class and we mak
  
}

#endif
