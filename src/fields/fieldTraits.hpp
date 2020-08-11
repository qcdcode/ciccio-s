#ifndef _FIELD_COMPONENTS_HPP
#define _FIELD_COMPONENTS_HPP

/// \file fieldComponents.hpp
///
/// \brief Determines fundamental type and order for components of a
/// field, given layout and list of components

#include <dataTypes/SIMD.hpp>
#include <fields/field.hpp>
#include <tensors/component.hpp>

namespace ciccios
{
  /// Componets and fundamental type of a field
  template <typename SPComp,
	    typename TC,
	    typename F,
	    FieldLayout FL>
  struct FieldTraits;
  
  /////////////////////////////////////////////////////////////////
  
  /// CPU field layout
  template <typename SPComp,
	    typename...Tc,
	    typename F>
  struct FieldTraits<SPComp,
		     TensComps<Tc...>,
		     F,
		     FieldLayout::CPU_LAYOUT>
  {
    /// Spacetime runs slower than everything else
    using Comps=
      TensComps<SPComp,Tc...>;
    
    /// Fundamental type is unchanged
    using Fund=
      F;
    
    /// Return the size, unchanged
    INLINE_FUNCTION static constexpr
    SPComp adaptSpaceTime(const SPComp& in)
    {
      return
	in;
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// GPU field layout
  template <typename SPComp,
	    typename...Tc,
	    typename F>
  struct FieldTraits<SPComp,
		     TensComps<Tc...>,
		     F,
		     FieldLayout::GPU_LAYOUT>
  {
    /// Spacetime runs faster than all the rest
    using Comps=
      TensComps<Tc...,SPComp>;
    
    /// Fundamental type is unchanged
    using Fund=
      F;
    
    /// Return the size, unchanged
    INLINE_FUNCTION static constexpr
    SPComp adaptSpaceTime(const SPComp& in)
    {
      return
	in;
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD field layout
  template <typename SPComp,
	    typename...Tc,
	    typename F>
  struct FieldTraits<SPComp,
		     TensComps<Tc...>,
		     F,
		     FieldLayout::SIMD_LAYOUT>
  {
    /// Signature of the non-fused site component
    struct UnFusedSPCompSignature :
      public TensCompSize<typename SPComp::Index,DYNAMIC>
    {
      /// Type used for the index
      using Index=
	typename SPComp::Index;
    };
    
    /// Unfused component
    using UnFusedSPComp=
      TensComp<UnFusedSPCompSignature,SPComp::rC,SPComp::which>;
    
    /// Components: Spacetime is replaced with the unfused part
    using Comps=
      TensComps<UnFusedSPComp,Tc...>;
    
    /// Fundamental type: SIMD vector
    using Fund=
      Simd<F>;
    
    /// Return the size, dividing by simd size
    INLINE_FUNCTION static constexpr
    UnFusedSPComp adaptSpaceTime(const SPComp& in)
    {
      CRASHER<<"SpaceTime "<<in<<" is not a multiple of simdLength for type "<<nameOfType((F*)nullptr)<<" "<<simdLength<F><<endl;
      
      return
	in/simdLength<F>;
    }
    
  };
}

#endif
