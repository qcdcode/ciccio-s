#ifndef _TENS_REF_HPP
#define _TENS_REF_HPP

/// \file tensRef.hpp
///
/// \brief A reference is a slice with no subscribed components

#include <tensors/tensSlice.hpp>

namespace ciccios
{
  /// A simple slice with no subscribed types
  template <bool B,
	    typename T>
  using TensRef=
    TensSlice<B,T,TensComps<>>;
  
#  define PROVIDE_TENS_REF_CREATOR(CONST_ATTR)				\
  /*! Returns a CONST_ATTR simple reference with no subscribed types */ \
  template <typename T>							\
  auto ref(CONST_ATTR TensFeat<IsTens,T>& t)				\
  {									\
    return								\
      TensRef<std::is_const<CONST_ATTR int>::value,T>(t);		\
  }
  
  PROVIDE_TENS_REF_CREATOR(const);
  PROVIDE_TENS_REF_CREATOR(/* not const*/);
  
#  undef PROVIDE_TENS_REF_CREATOR
}

#endif
