#ifndef _COMPONENT_SIGNATURE_HPP
#define _COMPONENT_SIGNATURE_HPP

/// \file componentSignature.hpp
///
/// \brief Implements the signature of a tensor component
///
/// The tensor component signature specifies whether the component has
/// a fixed or dynamic size, and which integer type is used to
/// represent the value. It is also used to tell apart the different
/// TensComp.

#include <base/metaProgramming.hpp>

namespace ciccios
{
  /// Row or column
  enum RwCl{ROW,CLN,ANY};
  
  /// Define the signature for a componenent convertible to TYPE of given NAME and SIZE
  ///
  /// The specified name is suffixed with "Signature", to allow the
  /// definition of the actual component with the expected name
#define DECLARE_COMPONENT_SIGNATURE(NAME,TYPE,SIZE)		\
  /*! Signature for the NAME component */			\
  struct NAME ## Signature :					\
    public TensCompSize<TYPE,SIZE>				\
  {								\
    /*! Type used for the index */				\
    using Size=TYPE;						\
  }
}

#endif
