#ifndef _FIELDTENS_PROVIDER_HPP
#define _FIELDTENS_PROVIDER_HPP

/// \file fieldTensProvider.hpp
///
/// \brief Implements the remapping of field components into tensor,
/// constructor etc

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <base/debug.hpp>
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
  
  /// Include Subscribe operator
  template <typename T>
  struct BaseFieldTensProvider
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    /// Provide subscribe method
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR)				\
    /*! Subscribe a component via CRTP */				\
    template <typename TC>						\
    decltype(auto) operator[](const TensCompFeat<IsTensComp,TC>& tc) CONST_ATTR	\
    {									\
      return								\
	this->deFeat().t[tc.deFeat()];					\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* non const*/);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
  };
  
  /// CPU layout
#define THIS								\
  FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::CPU_LAYOUT>
  
  /// Components for the CPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct THIS : BaseFieldTensProvider<THIS>
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
    
    /// Tensor
    T t;
    
    /// Create from the components sizes
    template <typename...TD,
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)+1==
				      std::tuple_size<typename T::DynamicComps>::value)>
    FieldTensProvider(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
		      const TensCompFeat<IsTensComp,TD>&...dynCompSize) :
      t(spaceTime.deFeat(),dynCompSize.dFeat()...)
    {
    }
  };
  
#undef THIS
  
  /// GPU layout
#define THIS								\
  FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::GPU_LAYOUT>
  
  /// Components for the GPU layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct THIS : public BaseFieldTensProvider<THIS>
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
  
#undef THIS
  
  /// SIMD layout
#define THIS								\
  FieldTensProvider<SPComp,TensComps<TC...>,F,SL,FieldLayout::SIMD_LAYOUT>
  
  /// Provides the tensor for a SIMD layout
  template <typename SPComp,
	    typename...TC,
	    typename F,
	    StorLoc SL>
  struct THIS : public BaseFieldTensProvider<THIS>
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
    
    /// Tensor
    T t;
    
    /// Create from the components sizes
    template <typename...TD,
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)+1==
				      std::tuple_size<typename T::DynamicComps>::value)>
    FieldTensProvider(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
		      const TensCompFeat<IsTensComp,TD>&...dynCompSize) :
      t(UnFusedComp(spaceTime.deFeat()/simdLength<F>),dynCompSize.dFeat()...)
    {
      if(spaceTime.deFeat()%simdLength<F>)
	CRASHER<<"SpaceTime "<<spaceTime.deFeat()<<" is not a multiple of simdLength for type "<<nameOfType((F*)nullptr)<<" "<<simdLength<F><<endl;
    }
  };
  
#undef THIS
}

#endif
