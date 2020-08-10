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
#define THIS					\
  Field<SPComp,TC,F,SL,FL>
  
  /// Field
  template <typename SPComp,
	    typename TC,
	    typename F,
	    StorLoc SL,
	    FieldLayout FL>
  struct Field : public
    FieldFeat<IsField,THIS>
  {
    /// Fundamental type
    using Fund=
      typename FieldComponents<FL,SPComp,TC,F>::Fund;
    
    /// Components
    using Comps=
      typename FieldComponents<FL,SPComp,TC,F>::Comps;
    
    /// Tensor type implementing the field
    using T=
      Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
    
    /// Tensor
    T t;
  };
  
#undef THIS
}

#endif
