#ifndef _FIELD_IMPL_HPP
#define _FIELD_IMPL_HPP

/// \file fieldImpl.hpp
///
/// \brief Implements fields on top of tensors

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <fields/field.hpp>
#include <fields/fieldComponents.hpp>

namespace ciccios
{
  /// Short name for the field
#  define THIS					\
  Field<TensComps<TC...>,SPComp,Fund,SL,FL>
  
  /// Field
  template <typename SPComp,
	    typename...TC,
	    typename Fund,
	    StorLoc SL,
	    FieldLayout FL>
  struct THIS : public
    FieldFeat<IsField,THIS>
  {
  };
  
#undef THIS
}

#endif
