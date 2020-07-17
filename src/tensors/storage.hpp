#ifndef _STORAGE_HPP
#define _STORAGE_HPP

#include <base/memoryManager.hpp>
#include <base/metaProgramming.hpp>
#include <tensors/componentSize.hpp>

namespace ciccios
{
  /// Basic storage, to use to detect storage
  template <typename T>
  struct BaseTensStorage : public Feature<T>
  {
  };
  
  /// Class to store the data
  template <typename Fund,           // Fundamental type
	    Size StaticSize,         // Size konwn at compile time
	    StorLoc SL>              // Location where to store data
  struct TensStorage
  {
    /// Structure to hold dynamically allocated data
    struct DynamicStorage
    {
      
      /// Storage
      Fund* data;
      
      /// Construct allocating data
      DynamicStorage(const Size& dynSize)
      {
	data=memoryManager<SL>()->template provide<Fund>(dynSize);
      }
      
      /// Destructor deallocating the memory
      ~DynamicStorage()
      {
	memoryManager<SL>()->release(// (void*&)
				     data);
      }
    };
    
    /// Structure to hold statically allocated data
    struct StackStorage
    {
      /// Storage
      Fund data[StaticSize];
      
      /// Constructor: since the data is statically allocated, we need to do nothing
      StackStorage(const Size&)
      {
      }
    };
    
    /// Threshold beyond which allocate dynamically in any case
    static constexpr Size MAX_STACK_SIZE=2304;
    
    /// Decide whether to allocate on the stack or dynamically
    static constexpr bool stackAllocated=
      (StaticSize!=DYNAMIC) and
      (StaticSize*sizeof(Fund)<=MAX_STACK_SIZE) and
      ((CompilingForDevice==true  and SL==StorLoc::ON_GPU) or
       (CompilingForDevice==false and SL==StorLoc::ON_CPU));
    
    /// Actual storage class
    using ActualStorage=std::conditional_t<stackAllocated,StackStorage,DynamicStorage>;
    
    /// Storage of data
    ActualStorage data;
    
    /// Forbids copy
    TensStorage(const TensStorage&) =delete;
    
    /// Construct taking the size to allocate
    TensStorage(const Size& size) ///< Size to allocate
      : data(size)
    {
    }
    
    /// Access to a sepcific value via subscribe operator
    template <typename T>                       // Subscribed component type
    const auto& operator[](const T& t) const  ///< Subscribed component
    {
      return data.data[t];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
  };
}

#endif
