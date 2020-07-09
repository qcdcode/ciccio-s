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
#include <base/metaProgramming.hpp>
#include <tensors/componentSize.hpp>
#include <tensors/componentSignature.hpp>

#include <array>

namespace ciccios
{
  int p=std::array<int,2>({0,8})[1];
  
  /// Tensor component defined by base type S
  template <typename S,
	    RwCl RC=ROW,
	    int Which=0>
  struct TensCompIdx : public TensCompSignature<TensCompIdx<S,RC,Which>>
  {
    /// Transposed type of component
    static constexpr RwCl TranspRC=(RC==ANY)?ANY:((RC==CLN)?ROW:CLN);
    
    /// Transposed component
    using Transp=TensCompIdx<S,TranspRC,Which>;
    
    /// Base type
    using Base=S;
    
    /// Value type
    using Size=typename S::Size;
    
    /// Value
    Size i;
    
    /// Check if the size is known at compile time
    static constexpr bool SizeIsKnownAtCompileTime=Base::sizeAtCompileTime()!=DYNAMIC;
    
    /// Init from value
    explicit constexpr TensCompIdx(Size i) : i(i)
    {
    }
    
    /// Default constructor
    TensCompIdx()
    {
    }
    
    /// Convert to actual value
    operator Size&()
    {
      return i;
    }
    
    /// Convert to actual value with const attribute
    operator const Size&() const
    {
      return i;
    }
    
    /// Transposed index
    auto transp() const
    {
      return Transp{i};
    }
    
    /// Assignment operator
    TensCompIdx& operator=(const Size& oth)
    {
      i=oth;
      
      return *this;
    }
  };
  
  /// Promotes the argument i to a COMPONENT, through a function with given NAME
#define DECLARE_COMPONENT_FACTORY(NAME,COMPONENT...)		\
  template <typename T>						\
  INLINE_ATTRIBUTE COMPONENT NAME(T&& i)			\
  {								\
    return i;							\
  }
  
  /// Declare a component with no special feature
  ///
  /// The component has no row/column tag or index, so it can be
  /// included only once in a tensor
#define DECLARE_COMPONENT(NAME,TYPE,SIZE,FACTORY)		\
  DECLARE_COMPONENT_SIGNATURE(NAME,TYPE,SIZE);			\
  								\
  /*! NAME component */						\
  using NAME=TensCompIdx<NAME ## Signature,ANY,0>;		\
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
  using NAME ## RC=TensCompIdx<NAME ## Signature,RC,Which>;	\
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
  
  DECLARE_COMPONENT(Compl,int,2,complComp);
  
  /// Number of component for a spin vector
  constexpr int NSpinComp=4;
  
  DECLARE_ROW_OR_CLN_COMPONENT(Spin,int,NSpinComp,sp);
  
  // constexpr int NDirac=4;
  
  // /// Qualify a component as such
  // template <typename T>
  // struct Comp : public Crtp<T>
  // {
  // };
  
  // enum RC{ROW,CLN};
  
  // template <typename Q,
  // 	    RC Rc,
  // 	    typename I,
  // 	    I _Max>
  // struct CompImpl : public Comp<CompImpl<Q,Rc,I,_Max>>
  // {
  //   using Int=I;
    
  //   static constexpr I Max=_Max;
    
  //   I i;
    
  //   operator const I&() const
  //   {
  //     return i;
  //   }
    
  //   operator I&()
  //   {
  //     return i;
  //   }
    
  //   CompImpl& operator=(const I& oth)
  //   {
  //     i=oth;
      
  //     return *this;
  //   }
  // };
  
  // struct _Spin; qua eredita da basecomp, che contiene min ed eventualmente disambigua

  // template <RC Rc>
  // using Spin=CompImpl<_Spin,Rc,int,NDirac>;
}

#endif
