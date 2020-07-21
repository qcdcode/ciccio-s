#ifndef _TENSOR_HPP
#define _TENSOR_HPP

#include <tensors/componentSignature.hpp>
#include <tensors/componentsList.hpp>
#include <tensors/offset.hpp>
#include <tensors/storage.hpp>
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
  
  /// Base type to detect a tensor
  template <typename T>
  struct TensFeat : public Feature<T>
  {
  };
  
  /// Tensor with Comps components, of Fund funamental type
  ///
  /// Forward definition to capture actual components
  template <typename Comps,
	    typename Fund=double,
	    StorLoc SL=DefaultStorage>
  struct Tens;
  
  /// Tensor
  template <typename F,
	    StorLoc SL,
	    typename...TC>
  struct Tens<TensComps<TC...>,F,SL> :
    public TensFeat<Tens<TensComps<TC...>,F,SL>>,
    public TensOffset<Tens<TensComps<TC...>,F,SL>,
		      TensComps<TC...>>
  {
    /// Fundamental type
    using Fund=F;
    
    /// Components
    using Comps=
      TensComps<TC...>;
    
    /// List of all statically allocated components
    using StaticComps=
      TupleFilter<SizeIsKnownAtCompileTime<true>::t,TensComps<TC...>>;
    
    /// List of all dynamically allocated components
    using DynamicComps=
      TupleFilter<SizeIsKnownAtCompileTime<false>::t,TensComps<TC...>>;
    
    /// Sizes of the dynamic components
    const DynamicComps dynamicSizes;
    
    /// Static size
    static constexpr Size staticSize=
      productAll<Size>((TC::SizeIsKnownAtCompileTime?
			TC::Base::sizeAtCompileTime:
			1)...);
    
    template <typename C>
    constexpr INLINE_FUNCTION
    auto offset() const
    {
      return this->getOffset((C*)nullptr);
    }
    
    // /// Calculate the index - no more components to parse
    // Size index(Size outer) const ///< Value of all the outer components
    // {
    //   return outer;
    // }
    
    /// Size of the Tv component
    ///
    /// Case in which the component size is knwon at compile time
    template <typename Tv,
	      std::enable_if_t<Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
    constexpr Size compSize() const
    {
      return Tv::Base::sizeAtCompileTime;
    }
    
    /// Size of the Tv component
    ///
    /// Case in which the component size is not knwon at compile time
    template <typename Tv,
	      std::enable_if_t<not Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
    const Size& compSize() const
    {
      return std::get<Tv>(dynamicSizes);
    }
    
    // // Calculate index iteratively
    
    // // Given the components (i,j,k) we must compute ((0*ni+i)*nj+j)*nk+k
    
    // // The parsing of the variadic components is done left to right, so
    // // to compute the nested bracket list we must proceed inward. Let's
    // // say we are at component j. We define outer=(0*ni+i) the result
    // // of inner step. We need to compute thisVal=outer*nj+j and pass it
    // // to the inner step, which incorporate iteratively all the inner
    // // components. The first step requires outer=0.
    // template <typename T,
    // 	      typename...Tp>
    // Size index(Size outer,      ///< Value of all the outer components
    // 	       T&& thisComp,       ///< Currently parsed component
    // 	       Tp&&...innerComps)  ///< Inner components
    //   const
    // {
    //   /// Remove reference to access to types
    //   using Tv=std::remove_reference_t<T>;
      
    //   /// Size of this component
    //   const Size thisSize=compSize<Tv>();
      
    //   //cout<<"thisSize: "<<thisSize<<endl;
    //   /// Value of the index when including this component
    //   const Size thisVal=outer*thisSize+thisComp;
      
    //   return index(thisVal,innerComps...);
    // }
    
    // /// Intermediate layer to reorder the passed components
    // template <typename...Cp>
    // Size reorderedIndex(Cp&&...comps) const
    // {
    //   /// Put the arguments in atuple
    //   auto argsInATuple=std::make_tuple(std::forward<Cp>(comps)...);
      
    //   /// Build the index reordering the components
    //   return index(0,std::get<TC>(argsInATuple)...);
    // }
    
    // /// Compute the data size
    // Size size;
    
    /// Determine whether the components are all static, or not
    static constexpr bool allCompsAreStatic=
      std::is_same<DynamicComps,std::tuple<>>::value;
    
    /// Computes the storage size at compile time, if knwon
    static constexpr Size storageSizeAtCompileTime=
      allCompsAreStatic?staticSize:DYNAMIC;
    
    /// Storage type
    using StorageType=
      TensStorage<Fund,storageSizeAtCompileTime,SL>;
    
    /// Actual storage
    StorageType data;
    
    /// Initialize the dynamical component \t Out using the inputs
    template <typename Ds,   // Type of the dynamically allocated components
	      typename Out>  // Type to set
    Size initializeDynSize(const Ds& inputs, ///< Input sizes
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
    Tens(const TensCompFeat<TD>&...td) :
      dynamicSizes{initializeDynSizes((DynamicComps*)nullptr,td()...)},
      data(staticSize*productAll<Size>(td()...))
    {
      this->setOffset();
      /// Dynamic size
      //const Size dynamicSize=product<Size>(std::forward<TD>(td)...);
      
      //size=dynamicSize*staticSize;
      //cout<<"Total size: "<<size<<endl;
      
      //data=std::unique_ptr<Fund[]>(new Fund[size]);
    }
    
    /// Move constructor
    Tens(Tens<TensComps<TC...>,Fund,SL>&& oth) : dynamicSizes(oth.dynamicSizes),data(oth.data.data.data)
    {
      LOGGER<<"Check offsets"<<endl;
      exit(0);
    }
    
    // Tens<TensComps<TC...>,Fund,SL,true> getRef()
    // {
    //   return Tens<TensComps<TC...>,Fund,SL,true>(dynamicSizes,data);
    // }
    
    // /// Access to inner data with any order
    // template <typename...Cp>
    // const Fund& eval(const TensCompFeat<Cp>&...comps) const ///< Components
    // {
    //   /// Compute the index
    //   const Size i=reorderedIndex(comps()...);
      
    //   //cout<<"Index: "<<i<<endl;
    //   return data[i];
    // }
    
    // PROVIDE_ALSO_NON_CONST_METHOD(eval);
    
    // /// Single component access via subscribe operator
    // template <typename T>                   // Subscribed component type
    // decltype(auto) operator[](T&& t) const  ///< Subscribed component
    // {
    //   return (*this)(std::forward<T>(t));
    // }
    
    //PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
    
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
  };
  
  // /// Traits of the Tensor
  // template <typename Fund,
  // 	    typename...TC,
  // 	    TensStorageLocation SL,
  // 	    bool IsRef>
  // struct CompsTraits<Tens<TensComps<TC...>,Fund,SL,IsRef>>
  // {
  //   using Type=Tens<TensComps<TC...>,Fund,SL,IsRef>;
    
  //   using Comps=TensComps<TC...>;
  // };
}

#endif
