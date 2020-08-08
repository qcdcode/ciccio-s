#ifndef _TENSOR_IMPL_HPP
#define _TENSOR_IMPL_HPP

/// \file tensorImpl.hpp
///
/// \brief Implements all functionalities of tensors

#include <tensors/tensor.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/reference.hpp>
#include <tensors/tensFeat.hpp>
#include <utilities/tuple.hpp>

namespace ciccios
{
  /// Short name for the tensor
#  define THIS \
    Tens<TensComps<TC...>,F,SL,IsStackable>
  
  /// Tensor
  template <typename F,
	    StorLoc SL,
	    typename...TC,
	    Stackable IsStackable>
  struct THIS : public
    TensFeat<IsTens,THIS>
  {
    /// Fundamental type
    using Fund=F;
    
    /// Components
    using Comps=
      TensComps<TC...>;
    
    /// Storage Location
    static constexpr
    StorLoc storLoc=
      SL;
    
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
	      ENABLE_TEMPLATE_IF(Tv::SizeIsKnownAtCompileTime)>
    CUDA_HOST_DEVICE INLINE_FUNCTION
    constexpr auto compSize() const
    {
      return
	Tv::Base::sizeAtCompileTime;
    }
    
    /// Size of the Tv component
    ///
    /// Case in which the component size is not knwon at compile time
    template <typename Tv,
	      ENABLE_TEMPLATE_IF(not Tv::SizeIsKnownAtCompileTime)>
    constexpr CUDA_HOST_DEVICE INLINE_FUNCTION
    const auto& compSize() const
    {
      return
	std::get<Tv>(dynamicSizes);
    }
    
    /// Calculate the index - no more components to parse
    constexpr CUDA_HOST_DEVICE INLINE_FUNCTION
    Index orderedCompsIndex(Index outer) const ///< Value of all the outer components
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
    constexpr CUDA_HOST_DEVICE INLINE_FUNCTION
    Index orderedCompsIndex(Index outer,        ///< Value of all the outer components
			    T&& thisComp,       ///< Currently parsed component
			    Tp&&...innerComps)  ///< Inner components
      const
    {
      /// Remove reference and all attributes to access to types
      using Tv=
	std::decay_t<T>;
      
      /// Size of this component
      const auto thisSize=
	compSize<Tv>();
      
      //cout<<"thisSize: "<<thisSize<<endl;
      /// Value of the index when including this component
      const auto thisVal=
	outer*thisSize+thisComp;
      
      return
	orderedCompsIndex(thisVal,innerComps...);
    }
    
    /// Intermediate layer to reorder the passed components
    template <typename...T>
    constexpr CUDA_HOST_DEVICE INLINE_FUNCTION
    Index index(const TensComps<T...>& comps) const
    {
      /// Build the index reordering the components
      return
	orderedCompsIndex(0,std::get<TC>(comps)...);
    }
    
    /// Determine whether the components are all static, or not
    static constexpr bool allCompsAreStatic=
      std::is_same<DynamicComps,std::tuple<>>::value;
    
    /// Computes the storage size at compile time, if knwon
    static constexpr Index storageSizeAtCompileTime=
      allCompsAreStatic?staticSize:DYNAMIC;
    
    /// Storage type
    using StorageType=
      TensStorage<Fund,storageSizeAtCompileTime,SL,IsStackable>;
    
    /// Actual storage
    StorageType data;
    
    /// Returns the pointer to the data
    CUDA_HOST_DEVICE
    decltype(auto) getDataPtr() const
    {
      return
	data.getDataPtr();
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(getDataPtr);
    
    /// Returns the name of the type
    static std::string nameOfType()
    {
      return
	std::string("Tensor<")+NAME_OF_TYPE(Fund)+","+storLocTag<SL>()+">";
    }
    
    /// Initialize the dynamical component \t Out using the inputs
    template <typename Ds,   // Type of the dynamically allocated components
	      typename Out>  // Type to set
    CUDA_HOST_DEVICE
    Index initializeDynSize(const Ds& inputs, ///< Input sizes
			    Out& out)         ///< Output size to set
    {
      out=std::get<Out>(inputs);
      
      return out;
    }
    
    /// Compute the size needed to initialize the tensor and set it
    template <typename...Td,
	      typename...T>
    CUDA_HOST_DEVICE
    TensComps<Td...> initializeDynSizes(TensComps<Td...>*,
					T&&...in)
    {
      static_assert(sizeof...(T)==sizeof...(Td),"Number of passed dynamic sizes not matching the needed one");
      
      return {std::get<Td>(std::make_tuple(in...))...};
    }
    
    /// Initialize the tensor with the knowledge of the dynamic size
    template <typename...TD,
	      ENABLE_TEMPLATE_IF(sizeof...(TD)>=1)>
    Tens(const TensCompFeat<IsTensComp,TD>&...tdFeat) :
      dynamicSizes{initializeDynSizes((DynamicComps*)nullptr,tdFeat.deFeat()...)},
      data(staticSize*productAll<Size>(tdFeat.deFeat()...))
    {
    }
    
    /// Initialize the tensor when no dynamic component is present
    template <typename...TD,
	      ENABLE_TEMPLATE_IF(sizeof...(TD)==0 and sizeof...(TC)==0)>
    CUDA_HOST_DEVICE
    Tens() :
      dynamicSizes{}
    {
    }
    
    /// Move constructor
    CUDA_HOST_DEVICE
    Tens(Tens<TensComps<TC...>,Fund,SL>&& oth) : dynamicSizes(oth.dynamicSizes),data(std::move(oth.data))
    {
    }
    
    /// Copy constructor
    CUDA_HOST_DEVICE
    Tens(const Tens& oth) : dynamicSizes(oth.dynamicSizes),data(oth.data)
    {
    }
    
    /// HACK
    CUDA_HOST_DEVICE
    Tens(Fund* oth) : dynamicSizes(),data(oth)
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
	      ENABLE_TEMPLATE_IF(std::tuple_size<Cp>::value>1 and	\
				 TupleHasType<C,Comps>)>		\
    CUDA_HOST_DEVICE INLINE_FUNCTION					\
    auto operator[](const TensCompFeat<IsTensComp,C>& cFeat) CONST_ATTR	\
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
	      ENABLE_TEMPLATE_IF(std::tuple_size<Cp>::value==1 and	\
				 TupleHasType<C,Comps>)>		\
    CUDA_HOST_DEVICE INLINE_FUNCTION CONST_ATTR				\
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
