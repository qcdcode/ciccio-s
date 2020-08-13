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
  /// List the various kind of layouts for a field
  enum class FieldLayout{CPU_LAYOUT,GPU_LAYOUT,SIMD_LAYOUT};
  
  /// Default layout to be used for all fields
  static constexpr
  FieldLayout DefaultFieldLayout=
    FieldLayout::
#ifdef USE_CUDA
    GPU_LAYOUT
#else
    SIMD_LAYOUT
#endif
    ;
  
  DEFINE_FEATURE(IsField);
  
  DEFINE_FEATURE_GROUP(FieldFeat);
  
  /// Field: a tensor with spacetime type
  template <typename SPComp,
	    typename TC,
	    typename F=double,
	    StorLoc SL=DefaultStorage,
	    FieldLayout FL=DefaultFieldLayout>
  struct Field;
}

#endif
