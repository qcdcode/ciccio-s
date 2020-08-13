#ifndef _COMPONENT_SIZE_HPP
#define _COMPONENT_SIZE_HPP

/// \file componentSize.hpp
///
/// \brief Implements the size of a tensor component

namespace ciccios
{
  /// Dynamic size
  constexpr int DYNAMIC=-1;
  
  /// Specify the size at compile time
  template <typename Size,
	    Size SIZE=DYNAMIC>
  struct TensCompSize
  {
    /// Value beyond end
    static constexpr Size sizeAtCompileTime=
      SIZE;
  };
  
  /// Predicate returning whether the size is known ow not at compile time
  template <bool Comp=true> // Value to be compared
  struct SizeIsKnownAtCompileTime
  {
    /// Internal implementation
    template <typename T>
    struct t
    {
      /// Predicate result
      static constexpr bool value=
	(T::SizeIsKnownAtCompileTime==Comp);
    };
  };
}

#endif
