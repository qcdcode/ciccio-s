#ifndef _FIELD_COMPONENTS_HPP
#define _FIELD_COMPONENTS_HPP

/// \file fieldComponents.hpp
///
/// \brief Implements the remapping of field components into tensor

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <dataTypes/SIMD.hpp>
#include <fields/field.hpp>
#include <tensors/component.hpp>

namespace ciccios
{
  /// Components fort a given layout
  template <FieldLayout FL,
	    typename SPComp,
	    typename TC,
	    typename F>
  struct FieldComponents;
  
  /// Components for the CPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F>
  struct FieldComponents<FieldLayout::CPU_LAYOUT,SPComp,TensComps<TC...>,F>
  {
    /// Components
    using Comps=
      TensComps<TC...,SPComp>;
    
    /// Fundamenal type is unchanged
    using Fund=
      F;
  };
  
  /// Components for the GPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F>
  struct FieldComponents<FieldLayout::GPU_LAYOUT,SPComp,TensComps<TC...>,F>
  {
    /// Components
    using Comps=
      TensComps<SPComp,TC...>;
    
    /// Fundamenal type is unchanged
    using Fund=
      F;
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// Split the components for SIMD field layout
  template <typename SPComp,
	    typename F>
  struct SIMDSPComp
  {
    /// Signature of the non-fused site component
    struct UnFusedCompSignature :
      public TensCompSize<typename SPComp::Index,DYNAMIC>
    {
      /// Type used for the index
      using Index=
	typename SPComp::Index;
    };
    
    /// Unfused component
    using UnFusedComp=
      TensComp<UnFusedCompSignature,SPComp::rC,SPComp::which>;
  };
  
  /// Provides the components for the field
  template <typename SPComp,
	    typename...TC,
	    typename F>
  struct FieldComponents<FieldLayout::SIMD_LAYOUT,SPComp,TensComps<TC...>,F>
  {
    /// Components of the field
    using Comps=
      TensComps<typename SIMDSPComp<SPComp,F>::UnFusedComp,
		TC...>;
    
    /// Fundamental type is the SIMD version of F
    using Fund=
      Simd<F>;
  };
}

#endif
