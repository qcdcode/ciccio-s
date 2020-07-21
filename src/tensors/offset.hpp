#ifndef _OFFSET_HPP
#define _OFFSET_HPP

/// \file offset.hpp
///
/// \brief Calculation of the offset for a given component in a tensor

#include <tensors/component.hpp>
#include <tensors/componentsList.hpp>

namespace ciccios
{
  template <typename T,
	    typename C>
  struct TensOffset;
  
  template <typename T,
	    typename HeadComp,
	    typename...TailComps>
  struct TensOffset<T,TensComps<HeadComp,TailComps...>> : public TensOffset<T,TensComps<TailComps...>>
  {
    using Nested=TensOffset<T,TensComps<TailComps...>>;
    
    using NestedOffset=typename Nested::Offset;
    
    using ThisCompSize=typename HeadComp::Size;
    
    using Offset=decltype(NestedOffset{}*ThisCompSize{});
    
    Offset offset;
    
    using Nested::getOffset;
    
    INLINE_FUNCTION const Offset& getOffset(HeadComp*) const
    {
      return offset;
    }
    
    INLINE_FUNCTION constexpr
    auto compSize() const
    {
      return
	static_cast<const T&>(*this).template compSize<HeadComp>();
    }
    
    void setOffset()
    {
      auto& nested=
	static_cast<Nested&>(*this);
      
      nested.setOffset();
      
      offset=(nested.offset*nested.compSize());
    }
  };
  
  template <typename T>
  struct TensOffset<T,TensComps<>>
  {
    using Offset=int;
    
    const Offset offset{1};
    
    INLINE_FUNCTION constexpr
    Offset compSize() const
    {
      return 1;
    }
    
    INLINE_FUNCTION const Offset& getOffset(void*) const
    {
      return offset;
    }
    
    void setOffset() const
    {
    }
  };
  
  // namespace impl
  // {
    
    
  //   /// Stauts of the tensor offset parsing
  //   enum class TensorOffsetParsing{BEFORE_COMP,AFTER_COMP};
    
  //   /// Compute the offset
  //   ///
  //   /// Forward declaration
  //   template <TensorOffsetParsing State,
  // 	      typename C,
  // 	      typename T>
  //   struct TensOffset;
    
  //   /// Compute the offset
  //   ///
  //   /// No more args to parse
  //   template <TensorOffsetParsing State,
  // 	      typename C>
  //   struct TensOffset<State,TensCompFeat<C>,TensComps<>>
  //   {
  //     /// Returns 1, to form the basis for the calculation
  //     template <typename T>
  //     INLINE_FUNCTION constexpr static
  //     auto offset(const TensFeat<T>& t)
  //     {
  // 	// LOGGER<<"We run out of comps"<<endl;
	
  // 	return 1;
  //     }
  //   };
    
  //   /// Compute the offset
  //   ///
  //   /// Case in which we are parsing the actual component
  //   template <typename C,
  // 	      typename...Tail>
  //   struct TensOffset<TensorOffsetParsing::BEFORE_COMP,TensCompFeat<C>,TensComps<C,Tail...>> :
  //     public TensOffset<TensorOffsetParsing::AFTER_COMP,TensCompFeat<C>,TensComps<C,Tail...>>
  //   {
  //   };
    
  //   /// Compute the offset
  //   ///
  //   /// Case in which we are parsing before the actual component
  //   template <typename C,
  // 		typename Head,
  // 		typename...Tail>
  //   struct TensOffset<TensorOffsetParsing::BEFORE_COMP,TensCompFeat<C>,TensComps<Head,Tail...>>
  //   {
  //     /// Returns the nested offset
  //     template <typename T>
  //     INLINE_FUNCTION constexpr static
  //     auto offset(const TensFeat<T>& t)
  //     {
  // 	// LOGGER<<"We are not deeper"<<endl;
	
  // 	/// Nested offset
  // 	const auto nested=
  // 	  TensOffset<TensorOffsetParsing::BEFORE_COMP,TensCompFeat<C>,TensComps<Tail...>>::offset(t);
	
  // 	return nested;
  //     }
  //   };
    
  //   /// Compute the offset
  //   ///
  //   /// Case in which we are parsing after the actual component
  //   template <typename C,
  // 	      typename Head,
  // 	      typename...Tail>
  //   struct TensOffset<TensorOffsetParsing::AFTER_COMP,TensCompFeat<C>,TensComps<Head,Tail...>>
  //   {
  //     /// Returns the nested offset multiplied by current size
  //     template <typename T>
  //     INLINE_FUNCTION constexpr static
  //     auto offset(const TensFeat<T>& t)
  //     {
  // 	// LOGGER<<"We are deeper"<<endl;
	
  // 	/// Nested offset
  // 	const auto nestedOffset=
  // 	  TensOffset<TensorOffsetParsing::AFTER_COMP,TensCompFeat<C>,TensComps<Tail...>>::offset(t);
	
  // 	/// Current size
  // 	const auto thisSize=
  // 	  t().template compSize<Head>();
	
  // 	return nestedOffset*thisSize;
  //     }
  //   };
  // }
  
  // /// Computes the offset of a given component
  // ///
  // /// Gives visibility to the internal implementation
  // template <typename C,
  // 	    typename T>
  // INLINE_FUNCTION constexpr
  // auto tensCompOffset(const TensFeat<T>& t)
  //   {
  //     using namespace impl;
      
  //     return TensOffset<TensorOffsetParsing::BEFORE_COMP,C,typename T::Comps>::offset(t);
  //   }
    

}

#endif
