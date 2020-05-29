#ifndef _MEMORY_MANAGER_HPP
#define _MEMORY_MANAGER_HPP

#include <map>
#include <vector>

#ifdef USE_CUDA
 #include <cuda_runtime.h>
#endif

#include "base/debug.hpp"
#include "base/logger.hpp"
#include "base/metaProgramming.hpp"
#include "utilities/valueWithExtreme.hpp"

namespace ciccios
{
#ifndef EXTERN_MEMORY_MANAGER
 #define EXTERN_MEMORY_MANAGER extern
#define INIT_MEMORY_MANAGER_TO(...)
#else
 #define INIT_MEMORY_MANAGER_TO(...) (__VA_ARGS__)
#endif
  
  enum class MemoryType{CPU ///< Memory allocated on CPU side
#ifdef USE_CUDA
			,GPU ///< Memory allocated on GPU side
#endif
  };
  
  /// Type used for size
  using Size=long int;
  
  /// Minimal alignment
#define DEFAULT_ALIGNMENT 32
  
  /// Memory manager, base type
  template <typename C>
  class BaseMemoryManager : public Crtp<C>
  {
  protected:
    
    /// Number of allocation performed
    Size nAlloc{0};
    
  private:
    
    /// List of dynamical allocated memory
    std::map<void*,Size> used;
    
    /// List of cached allocated memory
    std::map<Size,std::vector<void*>> cached;
    
    /// Size of used memory
    ValWithMax<Size> usedSize;
    
    /// Size of cached memory
    ValWithMax<Size> cachedSize;
    
    /// Use or not cache
    bool useCache{true};
    
    /// Number of cached memory reused
    Size nCachedReused{0};
    
    /// Add to the list of used memory
    void pushToUsed(void* ptr,
		    const Size size)
    {
      used[ptr]=size;
      
      usedSize+=size;
      
      VERB_LOGGER(3)<<"Pushing to used "<<ptr<<" "<<size<<", used: "<<usedSize<<endl;
    }
    
    /// Removes a pointer from the used list, without actually freeing associated memory
    ///
    /// Returns the size of the memory pointed
    Size popFromUsed(void* ptr) ///< Pointer to the memory to move to cache
    {
      VERB_LOGGER(3)<<"Popping from used "<<ptr<<endl;
      
      /// Iterator to search result
      auto el=used.find(ptr);
      
      if(el==used.end())
	CRASHER<<"Unable to find dinamically allocated memory "<<ptr<<endl;
      
      /// Size of memory
      const Size size=el->second;
      
      usedSize-=size;
      
      used.erase(el);
      
      return size;
    }
    
    /// Adds a memory to cache
    void pushToCache(void* ptr,          ///< Memory to cache
		     const Size size)    ///< Memory size
    {
      cached[size].push_back(ptr);
      
      cachedSize+=size;
      
      VERB_LOGGER(3)<<"Pushing to cache "<<size<<" "<<ptr<<", cache size: "<<cached.size()<<endl;
    }
    
    /// Check if a pointer is suitably aligned
    static bool isAligned(const void* ptr,
			  const Size alignment)
    {
      return reinterpret_cast<uintptr_t>(ptr)%alignment==0;
    }
    
    /// Pop from the cache, returning to use
    void* popFromCache(const Size& size,
		       const Size& alignment)
    {
      VERB_LOGGER(3)<<"Try to popping from cache "<<size<<endl;
      
      /// List of memory with searched size
      auto cachedIt=cached.find(size);
      
      if(cachedIt==cached.end())
	return nullptr;
      else
	{
	  /// Vector of pointers
	  auto& list=cachedIt->second;
	  
	  /// Get latest cached memory with appropriate alignment
	  auto it=list.end()-1;
	  
	  while(it!=list.begin()-1 and not isAligned(*it,alignment))
	    it--;
	  
	  if(it==list.begin()-1)
	    return nullptr;
	  else
	    {
	      /// Returned pointer, copied here before erasing
	      void* ptr=*it;
	      
	      list.erase(it);
	      
	      cachedSize-=size;
	      
	      if(list.size()==0)
		cached.erase(cachedIt);
	      
	      return ptr;
	    }
	}
    }
    
    /// Move the allocated memory to cache
    void moveToCache(void* ptr) ///< Pointer to the memory to move to cache
    {
      VERB_LOGGER(3)<<"Moving to cache "<<ptr<<endl;
      
      /// Size of pointed memory
      const Size size=popFromUsed(ptr);
      
      pushToCache(ptr,size);
    }
    
  public:
    
    /// Enable cache usage
    void enableCache()
    {
      useCache=true;
    }
    
    /// Disable cache usage
    void disableCache()
    {
      useCache=false;
      
      clearCache();
    }
    
    /// Allocate or get from cache after computing the proper size
    template <class T>
    T* provide(const Size nel,
	       const Size alignment=DEFAULT_ALIGNMENT)
    {
      /// Total size to allocate
      const Size size=sizeof(T)*nel;
      
      /// Allocated memory
      void* ptr;
      
      // Search in the cache
      ptr=popFromCache(size,alignment);
      
      // If not found in the cache, allocate new memory
      if(ptr==nullptr)
	ptr=this->crtp().allocateRaw(size,alignment);
      else
	nCachedReused++;
      
      pushToUsed(ptr,size);
      
      return static_cast<T*>(ptr);
    }
    
    /// Declare unused the memory and possibly free it
    template <typename T>
    void release(T* &ptr) ///< Pointer getting freed
    {
      if(useCache)
	moveToCache(static_cast<void*>(ptr));
      else
	{
	  popFromUsed(ptr);
	  this->crtp().deAllocateRaw(static_cast<void*>(ptr));
	}
      
      ptr=nullptr;
    }
    
    /// Release all used memory
    void releaseAllUsedMemory()
    {
      /// Iterator on elements to release
      auto el=
	used.begin();
      
      while(el!=used.end())
	{
	  VERB_LOGGER(3)<<"Releasing "<<el->first<<" size "<<el->second<<endl;
	  
	  /// Pointer to memory to release
	  void* ptr=el->first;
	  
	  // Increment iterator before releasing
	  el++;
	  
	  this->crtp().release(ptr);
	}
    }
    
    /// Release all memory from cache
    void clearCache()
    {
      VERB_LOGGER(3)<<"Clearing cache\n"<<endl;
      
      /// Iterator to elements of the cached memory list
      auto el=cached.begin();
      
      while(el!=cached.end())
	{
	  /// Number of elements to free
	  const Size n=el->second.size();
	  
	  /// Size to be removed
	  const Size size=el->first;
	  
	  // Increment before erasing
	  el++;
	  
	  for(Size i=0;i<n;i++)
	    {
	      VERB_LOGGER(3)<<"Removing from cache size  "<<el->first<<endl;
	      
	      /// Memory to free
	      void* ptr=popFromCache(size,DEFAULT_ALIGNMENT);
	      
	      VERB_LOGGER(3)<<"ptr: "<<ptr<<endl;
	      this->crtp().deAllocateRaw(ptr);
	    }
	}
    }
    
    /// Print to a stream
    void printStatistics()
    {
      LOGGER<<"Maximal memory cached: "<<cachedSize.extreme()<<" bytes, currently used: "<<(Size)cachedSize<<" bytes, number of reused: "<<nCachedReused<<endl;
    }
    
    /// Create the memory manager
    BaseMemoryManager() :
      usedSize(0),
      cachedSize(0)
    {
      LOGGER<<"Starting the memory manager"<<endl;
    }
    
    /// Destruct the memory manager
    ~BaseMemoryManager()
    {
      LOGGER<<"Stopping the memory manager"<<endl;
      
      printStatistics();
      
      releaseAllUsedMemory();
      
      clearCache();
    }
  };
  
  /// Manager of CPU memory
  struct CPUMemoryManager : public BaseMemoryManager<CPUMemoryManager>
  {
    /// Get memory
    ///
    /// Call the system routine which allocate memory
    void* allocateRaw(const Size size,        ///< Amount of memory to allocate
		      const Size alignment)   ///< Required alignment
    {
      /// Result
      void* ptr=nullptr;
      
      /// Returned condition
      VERB_LOGGER(3)<<"Allocating size "<<size<<" on CPU"<<endl;
      int rc=posix_memalign(&ptr,alignment,size);
      if(rc) CRASHER<<"Failed to allocate "<<size<<" CPU memory with alignement "<<alignment<<endl;
      VERB_LOGGER(3)<<"ptr: "<<ptr<<endl;
      
      nAlloc++;
      
      return ptr;
    }
    
    /// Properly free
    void deAllocateRaw(void* ptr)
    {
      VERB_LOGGER(3)<<"Freeing from CPU memory "<<ptr<<endl;
      free(ptr);
    }
  };
  
  EXTERN_MEMORY_MANAGER CPUMemoryManager* cpuMemoryManager;
  
#ifdef USE_CUDA
  
  /// Manager of GPU memory
  struct GPUMemoryManager : public BaseMemoryManager<GPUMemoryManager>
  {
    /// Get memory on GPU
    ///
    /// Call the system routine which allocate memory
    void* allocateRaw(const Size size,        ///< Amount of memory to allocate
		      const Size alignment)   ///< Required alignment
    {
      /// Result
      void* ptr=nullptr;
      
      VERB_LOGGER(3)<<"Allocating size "<<size<<" on GPU, "<<size<<endl;
      DECRYPT_CUDA_ERROR(cudaMalloc(&ptr,size),"Allocating on Gpu");
      VERB_LOGGER(3)<<"ptr: "<<ptr<<endl;
      
      nAlloc++;
      
      return ptr;
    }
    
    /// Properly free
    void deAllocateRaw(void* ptr)
    {
      VERB_LOGGER(3)<<"Freeing from GPU memory "<<ptr<<endl;
      DECRYPT_CUDA_ERROR(cudaFree(ptr),"Freeing from GPU");
    }
  };
  
  EXTERN_MEMORY_MANAGER GPUMemoryManager *gpuMemoryManager;

#endif
}

#undef EXTERN_MEMORY_MANAGER

#endif
