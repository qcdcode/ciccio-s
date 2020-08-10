#ifndef _FIELDTENS_PROVIDER_HPP
#define _FIELDTENS_PROVIDER_HPP

/// \file fieldTensProvider.hpp
///
/// \brief Implements the remapping of field components into tensor, constructor etc

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <dataTypes/SIMD.hpp>
#include <fields/field.hpp>
#include <tensors/component.hpp>

namespace ciccios
{
    
  /// Components for a given layout
  template <typename SPComp,
	    typename TC,
	    typename F,
	    StorLoc SL,
	    FieldLayout FL>
  struct FieldTensProvider;
  
  /// Components for the CPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::CPU_LAYOUT>
  {
    /// Components
    using Comps=
      TensComps<TC...,SPComp>;
    
    /// Fundamenal type is unchanged
    using Fund=
      F;
    
    /// Tensor type
    using T=
      Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
    
    /// Create from the components sizes
    template <typename...TD,
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)==
				      std::tuple_size<typename T::DynamicComps>::value)>
    FieldTensProvider(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
		      const TensCompFeat<IsTensComp,TD>&...dynCompSize) : T(spaceTime.deFeat(),dynCompSize.dFeat()...)
    {
    }
  };
  
  /// Components for the GPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::GPU_LAYOUT>
  {
    /// Components
    using Comps=
      TensComps<SPComp,TC...>;
    
    /// Fundamenal type is unchanged
    using Fund=
      F;
    
    /// Tensor type
    using T=
      Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
  };
  
  /// Provides the tensor for a SIMD layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::SIMD_LAYOUT>
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
    
    /// Components of the field
    using Comps=
      TensComps<UnFusedComp,TC...>;
    
    /// Fundamental type is the SIMD version of F
    using Fund=
      Simd<F>;
    
    /// Tensor type
    using T=
      Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
  };
}

#endif
