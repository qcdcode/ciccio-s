#ifndef _FIELD_IMPL_HPP
#define _FIELD_IMPL_HPP

/// \file fieldImpl.hpp
///
/// \brief Implements fields on top of tensors

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <expr/expr.hpp>
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
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)+1==
				      std::tuple_size<typename FieldTensProvider<SPComp,TC,F,SL,FL>::T::DynamicComps>::value)>
    Field(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
	  const TensCompFeat<IsTensComp,TD>&...dynCompSize) :
      FieldTensProvider<SPComp,TC,F,SL,FL>(spaceTime.deFeat(),dynCompSize.deFeat()...)
    {
    }
    
    /// Get components size from the tensor
    template <typename C>
    INLINE_FUNCTION constexpr
    decltype(auto) compSize() const
    {
      return
	this->t.template compSize<C>();
    }
    
    /// Copy from a non-simd layout to a simd layout
    template <typename OF,
	      FieldLayout TFL=FL,
	      ENABLE_THIS_TEMPLATE_IF(TFL==FieldLayout::SIMD_LAYOUT)>
    Field& operator=(const Field<SPComp,TC,OF,SL,FieldLayout::CPU_LAYOUT>& oth)
    {
      /// Get volume
      const SPComp& fieldVol=
	oth.template compSize<SPComp>();
      
      /// Traits of the field
      using FT=
	typename FieldTensProvider<SPComp,TC,F,SL,FL>::FT;
      
      for(SPComp spComp{0};spComp<fieldVol;spComp++)
	{
	  typename FT::UnFusedSPComp unFusedSPComp
	    {spComp/simdLength<F>};
	  
	  typename FT::FusedSPComp fusedSPComp
	    {spComp%simdLength<F>};
	  
	  this->t[unFusedSPComp][fusedSPComp]=
	    oth[spComp];
	}
      
      return
	*this;
    }
    
    /// Create SIMD from non simd
    template <typename OF,
	      FieldLayout TFL=FL,
	      ENABLE_THIS_TEMPLATE_IF(TFL==FieldLayout::SIMD_LAYOUT)>
    explicit Field(const Field<SPComp,TC,OF,SL,FieldLayout::CPU_LAYOUT>& oth) :
      Field(oth.template compSize<SPComp>())
    {
      (*this)=
	oth;
    }
  };
  
#undef THIS
}

#endif
