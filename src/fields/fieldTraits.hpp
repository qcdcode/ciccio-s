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
  /// Components and fundamental type of a field
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
    
    /// Return the size, unchanged
    INLINE_FUNCTION static constexpr
    SPComp adaptSpaceTime(const SPComp& in)
    {
      return
	in;
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// Split a component
  ///
  /// \todo rename, move
  template <typename T,
	    typename F>
  struct SIMDSplitter;
  
  template <typename Signature,
	    RwCl RC,
	    int Which,
	    typename F>
  struct SIMDSplitter<TensComp<Signature,RC,Which>,F>
  {
    /// Signature of the non-fused site component
    struct UnFusedSPCompSignature :
      public TensCompSize<typename Signature::Index,DYNAMIC>
    {
      /// Type used for the index
      using Index=
	typename Signature::Index;
    };
    
    /// Signature of the fused site component
    struct FusedSPCompSignature :
      public TensCompSize<int,simdLength<F>>
    {
      /// Type used for the index
      using Index=
	int;
    };
    
    /// Unfused component
    using UnFusedSPComp=
      TensComp<UnFusedSPCompSignature,RC,Which>;
    
    /// Fused component
    using FusedSPComp=
      TensComp<FusedSPCompSignature,RC,Which>;
    
    
  };
  
  /// SIMD field layout
  template <typename SPComp,
	    typename...Tc,
	    typename F>
  struct FieldTraits<SPComp,
		     TensComps<Tc...>,
		     F,
		     FieldLayout::SIMD_LAYOUT>
  {
    using UnFusedSPComp=
      typename SIMDSplitter<SPComp,F>::UnFusedSPComp;
    
    using FusedSPComp=
      typename SIMDSplitter<SPComp,F>::FusedSPComp;
    
    /// Components: Spacetime is replaced with the unfused part
    using Comps=
      TensComps<UnFusedSPComp,
		Tc...,
		FusedSPComp>;
    
    /// Return the size, dividing by simd size
    INLINE_FUNCTION static constexpr
    UnFusedSPComp adaptSpaceTime(const SPComp& in)
    {
      if(in%simdLength<F>!=0)
	CRASHER<<"SpaceTime "<<in<<" is not a multiple of simdLength for type "<<nameOfType((F*)nullptr)<<" "<<simdLength<F><<endl;
      
      return
	UnFusedSPComp{in/simdLength<F>};
    }
    
  };
}

#endif
