#ifndef _FIELD_HPP
#define _FIELD_HPP

/// \file field.hpp
///
/// \brief Implements fields on top of tensors

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <tensors/tensor.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  DEFINE_FEATURE(IsField);
  
  DEFINE_FEATURE_GROUP(FieldFeat);
  
  /// Field: a tensor with spacetime tipe
  template <typename SPComp,
	    typename Comps,
	    typename Fund=double,
	    StorLoc SL=DefaultStorage>
  struct Field;
  
  /// Short name for the field
#  define THIS					\
  Field<TensComps<TC...>,SPComp,Fund,SL>
  
  /// Field
  template <typename SPComp,
	    typename...TC,
	    typename Fund,
	    StorLoc SL>
  struct THIS : public
    FieldFeat<IsField,THIS>
  {
  };
  
#undef THIS
}

#endif
