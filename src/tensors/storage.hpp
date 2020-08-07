#ifndef _STORAGE_HPP
#define _STORAGE_HPP

#include <base/memoryManager.hpp>
#include <base/metaProgramming.hpp>
#include <tensors/componentSize.hpp>

namespace ciccios
{
  /// Stackability
  enum class Stackable{CANNOT_GO_ON_STACK,MIGHT_GO_ON_STACK};
  
  /// Basic storage, to use to detect storage
  template <typename T>
  struct BaseTensStorage
  {
    PROVIDE_DEFEAT_METHOD(T);
  };
  
  /// Class to store the data
  template <typename Fund,            // Fundamental type
	    Size StaticSize,          // Size konwn at compile time
	    StorLoc SL,               // Location where to store data
	    Stackable IsStackable=    // Select if can go or not on stack
	    Stackable::MIGHT_GO_ON_STACK>
  struct TensStorage
  {
    /// Structure to hold dynamically allocated data
    struct DynamicStorage
    {
      /// Hold info if it is a reference
      bool isRef;
      
      /// Storage
      Fund* data;
      
      /// Returns the pointer to data
      CUDA_HOST_DEVICE
      decltype(auto) getDataPtr() const
      {
	return data;
      }
      
      PROVIDE_ALSO_NON_CONST_METHOD_GPU(getDataPtr);
      
      /// Construct allocating data
      DynamicStorage(const Size& dynSize) :
	isRef(false),
	data(memoryManager<SL>()->template provide<Fund>(dynSize))
      {
      }
      
      /// "Copy" constructor, actually taking a reference
      CUDA_HOST_DEVICE
      DynamicStorage(const DynamicStorage& oth) :
	isRef(true),
	data(oth.data)
      {
      }
      
      /// Create a reference starting from a pointer
      CUDA_HOST_DEVICE
      DynamicStorage(Fund* oth) :
	isRef(true),
	data(oth)
      {
      }
      
      /// Move constructor
      CUDA_HOST_DEVICE
      DynamicStorage(DynamicStorage&& oth) :
	isRef(oth.isRef),
	data(oth.data)
      {
	oth.isRef=true;
	oth.data=nullptr;
      }
      
#ifndef COMPILING_FOR_DEVICE
      /// Destructor deallocating the memory
      ~DynamicStorage()
      {
	if(not isRef)
	  memoryManager<SL>()->release(data);
      }
#endif
    };
    
    /// Structure to hold statically allocated data
    struct StackStorage
    {
      /// Storage
      Fund data[StaticSize];
      
      /// Return the pointer to inner data
      CUDA_HOST_DEVICE
      const Fund* getDataPtr() const
      {
	return data;
      }
      
      PROVIDE_ALSO_NON_CONST_METHOD_GPU(getDataPtr);
      
      /// Constructor: since the data is statically allocated, we need to do nothing
      CUDA_HOST_DEVICE
      StackStorage(const Size&)
      {
      }
      
      /// Copy constructor is deleted
      CUDA_HOST_DEVICE
      StackStorage(const StackStorage&) =delete;
      
      /// Move constructor is deleted
      CUDA_HOST_DEVICE
      StackStorage(StackStorage&&) =delete;
    };
    
    /// Threshold beyond which allocate dynamically in any case
    static constexpr
    Size MAX_STACK_SIZE=2304;
    
    /// Decide whether to allocate on the stack or dynamically
    static constexpr
    bool stackAllocated=
      (StaticSize!=DYNAMIC) and
      (IsStackable==Stackable::MIGHT_GO_ON_STACK) and
      (StaticSize*sizeof(Fund)<=MAX_STACK_SIZE) and
      ((CompilingForDevice==true  and SL==StorLoc::ON_GPU) or
       (CompilingForDevice==false and SL==StorLoc::ON_CPU));
    
    /// Actual storage class
    using ActualStorage=
      std::conditional_t<stackAllocated,StackStorage,DynamicStorage>;
    
    /// Storage of data
    ActualStorage data;
    
    /// Returns the pointer to data
    CUDA_HOST_DEVICE
    decltype(auto) getDataPtr() const
    {
      return data.getDataPtr();
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(getDataPtr);
    
    /// Construct taking the size to allocate
    TensStorage(const Size& size) ///< Size to allocate
      : data(size)
    {
    }
    
    /// Creates starting from a reference
    CUDA_HOST_DEVICE
    TensStorage(Fund* oth) :
      data(oth)
    {
      static_assert(stackAllocated==false,"Only dynamic allocation is possible when creating a reference");
    }
    
    /// Construct taking the size to allocate
    TensStorage() ///< Size to allocate
    {
      static_assert(stackAllocated,"If not stack allocated must pass the size");
    }
    
    /// Copy constructor
    CUDA_HOST_DEVICE
    TensStorage(const TensStorage& oth) : data(oth.data)
    {
    }
    
    /// Move constructor
    CUDA_HOST_DEVICE
    TensStorage(TensStorage&& oth) : data(std::move(oth.data))
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
