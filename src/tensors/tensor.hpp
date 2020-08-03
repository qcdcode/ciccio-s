#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <tensors/componentSignature.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/reference.hpp>
#include <tensors/storage.hpp>
#include <tensors/tensFeat.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  /// Default storage for a tensor
  constexpr StorLoc DefaultStorage=StorLoc::
#ifdef USE_CUDA
    ON_GPU
#else
    ON_CPU
#endif
    ;
  
  /// Tensor with Comps components, of Fund fundamental type
  ///
  /// Forward definition to capture actual components
  template <typename Comps,
	    typename Fund=double,
	    StorLoc SL=DefaultStorage>
  struct Tens;
  
  /// Short name for the tensor
  #define THIS \
    Tens<TensComps<TC...>,F,SL>
  
  /// Tensor
  template <typename F,
	    StorLoc SL,
	    typename...TC>
  struct THIS : public
    TensFeat<IsTens,THIS>
  {
    /// Fundamental type
    using Fund=F;
    
    /// Components
    using Comps=
      TensComps<TC...>;
    
    /// Type to be used for the index
    using Index=
      std::common_type_t<TC...>;
    
    /// List of all statically allocated components
    using StaticComps=
      TupleFilter<SizeIsKnownAtCompileTime<true>::t,TensComps<TC...>>;
    
    /// List of all dynamically allocated components
    using DynamicComps=
      TupleFilter<SizeIsKnownAtCompileTime<false>::t,TensComps<TC...>>;
    
    /// Sizes of the dynamic components
    const DynamicComps dynamicSizes;
    
    /// Static size
    static constexpr Index staticSize=
      productAll<Size>((TC::SizeIsKnownAtCompileTime?
			TC::Base::sizeAtCompileTime:
			1)...);
    
    /// Size of the Tv component
    ///
    /// Case in which the component size is knwon at compile time
    template <typename Tv,
	      std::enable_if_t<Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
    constexpr auto compSize() const
    {
      return
	Tv::Base::sizeAtCompileTime;
    }
    
    /// Size of the Tv component
    ///
    /// Case in which the component size is not knwon at compile time
    template <typename Tv,
	      std::enable_if_t<not Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
    const auto& compSize() const
    {
      return
	std::get<Tv>(dynamicSizes);
    }
    
    /// Calculate the index - no more components to parse
    Index _index(Index outer) const ///< Value of all the outer components
    {
      return
	outer;
    }
    
    /// Calculate index iteratively
    ///
    /// Given the components (i,j,k) we must compute ((0*ni+i)*nj+j)*nk+k
    ///
    /// The parsing of the variadic components is done left to right, so
    /// to compute the nested bracket list we must proceed inward. Let's
    /// say we are at component j. We define outer=(0*ni+i) the result
    /// of inner step. We need to compute thisVal=outer*nj+j and pass it
    /// to the inner step, which incorporate iteratively all the inner
    /// components. The first step requires outer=0.
    template <typename T,
    	      typename...Tp>
    Index _index(Index outer,        ///< Value of all the outer components
    	       T&& thisComp,       ///< Currently parsed component
    	       Tp&&...innerComps)  ///< Inner components
      const
    {
      /// Remove reference to access to types
      using Tv=
	std::remove_reference_t<T>;
      
      /// Size of this component
      const auto thisSize=
	compSize<Tv>();
      
      //cout<<"thisSize: "<<thisSize<<endl;
      /// Value of the index when including this component
      const auto thisVal=
	outer*thisSize+thisComp;
      
      return
	_index(thisVal,innerComps...);
    }
    
    /// Intermediate layer to reorder the passed components
    template <typename...T>
    Index index(const TensComps<T...>& comps) const
    {
      /// Build the index reordering the components
      return
	_index(0,std::get<TC>(comps)...);
    }
    
    /// Determine whether the components are all static, or not
    static constexpr bool allCompsAreStatic=
      std::is_same<DynamicComps,std::tuple<>>::value;
    
    /// Computes the storage size at compile time, if knwon
    static constexpr Index storageSizeAtCompileTime=
      allCompsAreStatic?staticSize:DYNAMIC;
    
    /// Storage type
    using StorageType=
      TensStorage<Fund,storageSizeAtCompileTime,SL>;
    
    /// Actual storage
    StorageType data;
    
    decltype(auto) getDataPtr() const
    {
      return data.getDataPtr();
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(getDataPtr);
    
    /// Initialize the dynamical component \t Out using the inputs
    template <typename Ds,   // Type of the dynamically allocated components
	      typename Out>  // Type to set
    Index initializeDynSize(const Ds& inputs, ///< Input sizes
			   Out& out)         ///< Output size to set
    {
      out=std::get<Out>(inputs);
      
      return out;
    }
    
    /// Compute the size needed to initialize the tensor and set it
    template <typename...Td,
	      typename...T>
    TensComps<Td...> initializeDynSizes(TensComps<Td...>*,
					T&&...in)
    {
      static_assert(sizeof...(T)==sizeof...(Td),"Number of passed dynamic sizes not matching the needed one");
      
      return {std::get<Td>(std::make_tuple(in...))...};
    }
    
    /// Initialize the tensor with the knowledge of the dynamic size
    template <typename...TD>
    Tens(const TensComp<TD>&...td) :
      dynamicSizes{initializeDynSizes((DynamicComps*)nullptr,td()...)},
      data(staticSize*productAll<Size>(td()...))
    {
    }
    
    /// Move constructor
    Tens(Tens<TensComps<TC...>,Fund,SL>&& oth) : dynamicSizes(oth.dynamicSizes),data(oth.data.data.data)
    {
    }
    
    /// Provide trivial access to the fundamental data
    const Fund& trivialAccess(const Size& i) const
    {
      return data[i];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(trivialAccess);
    
    /// Gets access to the inner data
    const Fund* getRawAccess() const
    {
      return &trivialAccess(0);
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(getRawAccess);
    
    /// Provide subscribe operator when returning a reference
    ///
    /// \todo move to tag dispatch, so we can avoid the horrible sfinae subtleties
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR,CONST_AS_BOOL)		\
    /*! Operator to take a const reference to a given component */	\
    template <typename C,						\
	      typename Cp=Comps,					\
	      SFINAE_ON_TEMPLATE_ARG(std::tuple_size<Cp>::value>1 and	\
				     TupleHasType<C,Comps>)>		\
    INLINE_FUNCTION auto operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR	\
    {									\
      /*! Subscribed components */					\
      using SubsComps=							\
	TensComps<C>;							\
									\
      return								\
	TensRef<CONST_AS_BOOL,THIS,SubsComps>(*this,SubsComps(cFeat.deFeat())); \
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */, false);
    PROVIDE_SUBSCRIBE_OPERATOR(const, true);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Provide subscribe operator when returning direct access
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR)				\
    /*! Operator to return direct access to data */			\
    template <typename C,						\
	      typename Cp=Comps,					\
	      SFINAE_ON_TEMPLATE_ARG(std::tuple_size<Cp>::value==1 and	\
				     TupleHasType<C,Comps>)>		\
    INLINE_FUNCTION CONST_ATTR						\
    Fund& operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR \
    {									\
      return								\
	data[cFeat.deFeat()];						\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* not const */);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
  };
  
#undef THIS
}

#endif
