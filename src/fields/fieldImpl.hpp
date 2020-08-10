#ifndef _FIELD_IMPL_HPP
#define _FIELD_IMPL_HPP

/// \file fieldImpl.hpp
///
/// \brief Implements fields on top of tensors

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <fields/field.hpp>
#include <fields/fieldTensProvider.hpp>

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
  FieldFeat<IsField,THIS>,
    FieldTensProvider<SPComp,TC,F,SL,FL>
  {
    /// Construct from sizes
    template <typename...TD,
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)==
				      std::tuple_size<typename FieldTensProvider<SPComp,TC,F,SL,FL>::T::DynamicComps>::value)>
    Field(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
	  const TensCompFeat<IsTensComp,TD>&...dynCompSize) :
      FieldTensProvider<SPComp,TC,F,SL,FL>(spaceTime.deFeat(),dynCompSize.deFeat()...)
    {
    }
    
    // /// Tensor type implementing the field
    // using T=
    //   Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
    
    /// Tensor
    //T t;
  };
  
#undef THIS
}

#endif
