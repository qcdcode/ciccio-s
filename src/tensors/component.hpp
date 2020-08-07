#ifndef _COMPONENT_HPP
#define _COMPONENT_HPP

/// \file component.hpp
///
/// \brief Implements the basic feature of a tensor component
///
/// A tensor component is a class with a given signature, representing
/// the tensor index of a certain kind S. Any tensor is represented by
/// a list of component. Each component can be of Row or Column type,
/// and can be listed more than once. Each occurrency is distinguished
/// by the parameter Which.
///
/// To declare a new tensor component, create a class inheriting from
/// TensCompIdx with the appropriated signature.

#include <base/inliner.hpp>
#include <base/feature.hpp>
#include <base/metaProgramming.hpp>
#include <tensors/componentSize.hpp>
#include <tensors/componentSignature.hpp>

#include <array>

namespace ciccios
{
  DEFINE_FEATURE(IsTensComp);
  
  DEFINE_FEATURE_GROUP(TensCompFeat);
  
  /// Short name for the TensComp
#define THIS					\
  TensComp<S,RC,Which>
  
  /// Tensor component defined by base type S
  template <typename S,
	    RwCl RC=ROW,
	    int Which=0>
  struct TensComp : public TensCompFeat<IsTensComp,THIS>
  {
    /// Transposed type of component
    static constexpr RwCl TranspRC=
      (RC==ANY)?ANY:((RC==CLN)?ROW:CLN);
    
    /// Transposed component
    using Transp=
      TensComp<S,TranspRC,Which>;
    
    /// Base type
    using Base=S;
    
    /// Value type
    using Index=
      typename S::Index;
    
    /// Value
    Index i;
    
    /// Check if the size is known at compile time
    static constexpr
    bool SizeIsKnownAtCompileTime=
      Base::sizeAtCompileTime!=DYNAMIC;
    
    /// Init from value
    template <typename T>
    CUDA_HOST_DEVICE
    explicit constexpr TensComp(T&& i) : i(i)
    {
    }
    
    /// Default constructor
    CUDA_HOST_DEVICE
    TensComp()
    {
    }
    
    /// Convert to actual value
    CUDA_HOST_DEVICE
    operator Index&()
    {
      return i;
    }
    
    /// Convert to actual value with const attribute
    CUDA_HOST_DEVICE
    operator const Index&() const
    {
      return i;
    }
    
    /// Transposed index
    CUDA_HOST_DEVICE
    auto transp() const
    {
      return Transp{i};
    }
    
    /// Assignment operator
    CUDA_HOST_DEVICE
    TensComp& operator=(const Index& oth)
    {
      i=oth;
      
      return *this;
    }
  };
  
 # undef THIS
  
  /// Promotes the argument i to a COMPONENT, through a function with given NAME
#define DECLARE_COMPONENT_FACTORY(NAME,COMPONENT...)		\
  template <typename T>						\
  CUDA_HOST_DEVICE						\
  INLINE_FUNCTION COMPONENT NAME(T&& i)				\
  {								\
    return							\
      COMPONENT(i);						\
  }
  
  /// Declare a component with no special feature
  ///
  /// The component has no row/column tag or index, so it can be
  /// included only once in a tensor
#define DECLARE_COMPONENT(NAME,TYPE,SIZE,FACTORY)		\
  DECLARE_COMPONENT_SIGNATURE(NAME,TYPE,SIZE);			\
  								\
  /*! NAME component */						\
  using NAME=							\
    TensComp<NAME ## Signature,ANY,0>;				\
								\
  DECLARE_COMPONENT_FACTORY(FACTORY,NAME)
  
  /// Declare a component which can be included more than once
  ///
  /// The component has a row/column tag, and an additional index, so
  /// it can be included twice in a tensor
#define DECLARE_ROW_OR_CLN_COMPONENT(NAME,TYPE,SIZE,FACTORY)	\
  DECLARE_COMPONENT_SIGNATURE(NAME,TYPE,SIZE);			\
  								\
  /*! NAME component */						\
  template <RwCl RC=ROW,					\
	    int Which=0>					\
  using NAME ## RC=TensComp<NAME ## Signature,RC,Which>;	\
								\
  /*! Row kind of NAME component */				\
  using NAME ## Row=NAME ## RC<ROW,0>;				\
								\
  /*! Column kind of NAME component */				\
  using NAME ## Cln=NAME ## RC<CLN,0>;				\
  								\
  /*! Default NAME component is Row */				\
  using NAME=NAME ## Row;					\
  								\
  DECLARE_COMPONENT_FACTORY(FACTORY ## Row,NAME ## Row);	\
								\
  DECLARE_COMPONENT_FACTORY(FACTORY ## Cln,NAME ## Cln);	\
								\
  DECLARE_COMPONENT_FACTORY(FACTORY,NAME)
  
  /////////////////////////////////////////////////////////////////
  
  /// \todo move to a physics file
  
  DECLARE_COMPONENT(Compl,int,2,complComp);
  
  /// Number of component for a spin vector
  constexpr int NSpinComp=4;
  
  DECLARE_ROW_OR_CLN_COMPONENT(Spin,int,NSpinComp,sp);
  
  /// Number of component for a color vector
  constexpr int NColComp=3;
  
  DECLARE_ROW_OR_CLN_COMPONENT(Col,int,NColComp,cl);
  
  // Spacetime
  //DECLARE_COMPONENT(SpaceTime,int64_t,DYNAMIC,spaceTime);
  DECLARE_COMPONENT(SpaceTime,int,DYNAMIC,spaceTime);
}

#endif
