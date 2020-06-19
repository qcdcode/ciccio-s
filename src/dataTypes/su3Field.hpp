#ifndef _SU3FIELD_HPP
#define _SU3FIELD_HPP

#include "base/memoryManager.hpp"
#include "dataTypes/su3.hpp"

namespace ciccios
{
  /// Base type for SU3Field, needed to obtain static polymorphism
  template <typename T>
  struct SU3Field : public Crtp<T>
  {
    /// Copy from oth, using the correct deep copier
    template <typename O>
    T& deepCopy(const O& oth);
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD su3 field
  ///
  /// Forward definition
  template <typename Fund,
	    StorLoc SL>
  struct SimdSU3Field;
  
  /// Trivial su3 field
  ///
  /// Forward definition
  template <typename Fund,
	    StorLoc SL>
  struct CpuSU3Field;
  
  /// GPU version of su3 field
  ///
  /// Forward definition
  template <typename Fund,
	    StorLoc SL>
  struct GpuSU3Field;
  
  /////////////////////////////////////////////////////////////////
  
  /// Trivial su3 field
  template <typename Fund,
	    StorLoc SL>
  struct CpuSU3Field : public SU3Field<CpuSU3Field<Fund,SL>>
  {
    /// Returns the name of the type
    static std::string nameOfType()
    {
      return std::string("CpuSU3Field<")+NAME_OF_TYPE(Fund)+","+storLocTag<SL>()+">";
    }
    
    /// Base type
    using BaseType=
      Fund;
    
    /// Volume
    const int vol;
    
    /// Store wether this is a reference
    const bool isRef;
    
    /// Internal data
    Fund* data;
    
    /// Index function
    HOST DEVICE
    int index(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iSite));
    }
    
    /// Access to data
    HOST DEVICE
    const Fund& operator()(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(operator());
    
    /// Access to the site
    const SU3<Complex<Fund>>& site(const int& iSite) const
    {
      const Fund& ref=(*this)(iSite,0,0,0);
      
      return *reinterpret_cast<const SU3<Complex<Fund>>*>(&ref);
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(site);
    
    /// Create knowning volume
    CpuSU3Field(const int vol) : vol(vol),isRef(false)
    {
      /// Compute size
      const int size=index(vol,0,0,0);
      
      data=(Fund*)memoryManager<SL>()->template provide<Fund>(size);
    }
    
    /// Copy constructor
    CpuSU3Field(const CpuSU3Field& oth) : vol(oth.vol),isRef(true),data(oth.data)
    {
    }
    
    /// Destroy
    HOST DEVICE
    ~CpuSU3Field()
    {
#ifndef COMPILING_FOR_DEVICE
      if(not isRef)
	memoryManager<SL>()->release(data);
#endif
    }
    
    /// Sum the product of the two passed fields
    INLINE_FUNCTION CpuSU3Field& sumProd(const CpuSU3Field& oth1,const CpuSU3Field& oth2)
    {
      ASM_BOOKMARK_BEGIN("UnrolledCPUmethod");
      for(int iSite=0;iSite<this->vol;iSite++)
      	{
      	  auto& a=this->CPUSite(iSite);
      	  const auto& b=oth1.CPUSite(iSite);
      	  const auto& c=oth2.CPUSite(iSite);
	  
	  a.sumProd(b,c);
	}
      ASM_BOOKMARK_END("UnrolledCPUmethod");
      
      return *this;
    }
    
    /// Loop over all sites
    template <typename F>
    void sitesLoop(F&& f) const
    {
      ThreadPool::loopSplit(0,vol,std::forward<F>(f));
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD version of the field
  template <typename Fund,
	    StorLoc SL>
  struct SimdSU3Field : public SU3Field<SimdSU3Field<Fund,SL>>
  {
    /// Returns the name of the type
    static std::string nameOfType()
    {
      return std::string("SimdSU3Field<")+NAME_OF_TYPE(Fund)+","+storLocTag<SL>()+">";
    }
    
    /// Base type
    using BaseType=
      Fund;
    
    /// Volume
    const int fusedVol;
    
    /// Store wether this is a reference
    const bool isRef;
    
    /// Internal data
    Simd<Fund>* data;
    
    /// Index to internal data
    HOST DEVICE
    int index(const int& iFusedSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iFusedSite));
    }
    
    /// Access to data
    HOST DEVICE
    const Simd<Fund>& operator()(const int& iFusedSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iFusedSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(operator());
    
    /// Access to data as a complex quantity
    const Complex<Simd<Fund>>& operator()(const int& iFusedSite,const int& icol1,const int& icol2) const
    {
      return *reinterpret_cast<const Complex<Simd<Fund>>*>(&data[index(iFusedSite,icol1,icol2,0)]);
    }
    
    /// Creates starting from the physical volume
    SimdSU3Field(const int& vol) : fusedVol(vol/simdLength<Fund>),isRef(false)
    {
      /// Compute the size
      const int size=index(fusedVol,0,0,0);
      
      data=cpuMemoryManager->template provide<Simd<Fund>>(size);
    }
    
    /// Copy constructor
    HOST DEVICE
    SimdSU3Field(const SimdSU3Field& oth) : fusedVol(oth.fusedVol),isRef(true),data(oth.data)
    {
    }
    
    /// Destroy
    HOST DEVICE
    ~SimdSU3Field()
    {
#ifndef COMPILING_FOR_DEVICE
      if(not isRef)
	cpuMemoryManager->release(data);
#endif
    }
    
    /// Sum the product of the two passed fields
    INLINE_FUNCTION SimdSU3Field& sumProd(const SimdSU3Field& oth1,const SimdSU3Field& oth2)
    {
      ASM_BOOKMARK_BEGIN("UnrolledSIMDmethod");
      
      for(int iFusedSite=0;iFusedSite<this->fusedVol;iFusedSite++)
	this->simdSite(iFusedSite).sumProd(oth1.simdSite(iFusedSite),oth2.simdSite(iFusedSite));
      
      ASM_BOOKMARK_END("UnrolledSIMDmethod");
      
      return *this;
    }
    
    /// Loop over all sites
    template <typename F>
    void sitesLoop(F&& f) const
    {
      ThreadPool::loopSplit(0,fusedVol,std::forward<F>(f));
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  namespace resources
  {
    /// Instantiate the proper loop splitter
    ///
    /// Forward definition
    template <StorLoc SL>
    struct GpuSitesLooper;
    
    /// Instantiate the proper loop splitter
    ///
    /// CPU case
    template <>
    struct GpuSitesLooper<StorLoc::ON_CPU>
    {
      /// Execute the loop
      template <typename F>
      static void exec(const int min,const int max,F&& f)
      {
	ThreadPool::loopSplit(min,max,std::forward<F>(f));
      }
    };
    
    /// Instantiate the proper loop splitter
    ///
    /// GPU case
    template <>
    struct GpuSitesLooper<StorLoc::ON_GPU>
    {
      /// If compiling with cuda exec the loop via threads, divert to cpu case otherwise
      template <typename F>
      static void exec(const int min,const int max,F&& f)
      {
#ifdef USE_CUDA
	Gpu::cudaParallel(min,max,std::forward<F>(f));
#else
	GpuSitesLooper<StorLoc::ON_CPU>::exec(min,max,std::forward<F>(f));
#endif
      }
    };
  }
  
  /// GPU version of  su3 field
  template <typename Fund,
	    StorLoc SL>
  struct GpuSU3Field : public SU3Field<GpuSU3Field<Fund,SL>>
  {
    /// Returns the name of the type
    static std::string nameOfType()
    {
      return std::string("GpuSU3Field<")+NAME_OF_TYPE(Fund)+","+storLocTag<SL>()+">";
    }
    
    /// Base type
    using BaseType=
      Fund;
    
    /// Volume
    const int vol;
    
    /// Store wether this is a reference
    const bool isRef;
    
    /// Internal data
    Fund* data;
    
    /// Index function
    HOST DEVICE
    int index(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(iSite+vol*(icol2+NCOL*icol1));
    }
    
    /// Access to data
    HOST DEVICE
    const Fund& operator()(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(operator());
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(site);
    
    /// Create knowning volume
    GpuSU3Field(const int& vol) : vol(vol),isRef(false)
    {
      /// Compute size
      const int size=index(0,NCOL,0,0);
      
      data=(Fund*)memoryManager<SL>()->template provide<Fund>(size);
    }
    
    /// Copy constructor
    HOST DEVICE
    GpuSU3Field(const GpuSU3Field& oth) : vol(oth.vol),isRef(true),data(oth.data)
    {
    }
    
    /// Destroy
    HOST DEVICE
    ~GpuSU3Field()
    {
#ifndef COMPILING_FOR_DEVICE
      if(not isRef)
	memoryManager<SL>()->release(data);
#endif
    }
    
    /// Loop over all sites
    template <typename F>
    void sitesLoop(F&& f) const
    {
      resources::GpuSitesLooper<SL>::exec(0,vol,std::forward<F>(f));
    }
  };
  
  namespace resources
  {
    /// Assign from a non-simd version
    template <typename F,
	      typename OF>
    SimdSU3Field<F,StorLoc::ON_CPU>& deepCopy(SimdSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      for(int iSite=0;iSite<res.fusedVol*simdLength<F>;iSite++)
    	{
    	  /// Index of the simd fused sites
    	  const int iFusedSite=iSite/simdLength<F>;
	  
    	  /// Index of the simd component
    	  const int iSimdComp=iSite%simdLength<F>;
	  
    	  for(int ic1=0;ic1<NCOL;ic1++)
    	    for(int ic2=0;ic2<NCOL;ic2++)
   	      for(int ri=0;ri<2;ri++)
   		res(iFusedSite,ic1,ic2,ri)[iSimdComp]=oth(iSite,ic1,ic2,ri);
    	}
      
      return res;
    }
    
    /// Assign from a non-simd version
    template <typename F,
	      typename OF>
    CpuSU3Field<F,StorLoc::ON_CPU>& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      for(int iSite=0;iSite<res.vol;iSite++)
    	{
    	  for(int ic1=0;ic1<NCOL;ic1++)
    	    for(int ic2=0;ic2<NCOL;ic2++)
    	      for(int ri=0;ri<2;ri++)
    		res(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
    	}
      
      return res;
    }
    
    /// Assign from SIMD version, with possible different type
    template <typename F,
	      typename OF>
    auto& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const SimdSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      for(int iFusedSite=0;iFusedSite<oth.fusedVol;iFusedSite++)
	for(int ic1=0;ic1<NCOL;ic1++)
	  for(int ic2=0;ic2<NCOL;ic2++)
	    for(int ri=0;ri<2;ri++)
	      for(int iSimdComp=0;iSimdComp<simdLength<OF>;iSimdComp++)
		{
		  const int iSite=iSimdComp+simdLength<OF>*iFusedSite;
		
		  res(iSite,ic1,ic2,ri)=oth(iFusedSite,ic1,ic2,ri)[iSimdComp];
  	      }
      
      return res;
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Copy an SU3 field within GPU, with GPU layout, and possible different types
    template <typename F,
	      typename OF>
    GpuSU3Field<F,StorLoc::ON_GPU>& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const GpuSU3Field<OF,StorLoc::ON_GPU>& oth)
      {
	CRASHER<<"Must be done with kernel"<<endl;
	return res;
      }
    
    /// Copy an SU3 field from GPU with GPU layout to CPU with GPU layout, with the same type
    template <typename F>
    GpuSU3Field<F,StorLoc::ON_CPU>&deepCopy(GpuSU3Field<F,StorLoc::ON_CPU>& res,const GpuSU3Field<F,StorLoc::ON_GPU>& oth)
    {
      /// Size to be copied \todo please move somewhere more meaningful
      const int64_t size=oth.vol*sizeof(F)*NCOL*NCOL*2;
      
#ifdef USE_CUDA
      DECRYPT_CUDA_ERROR(cudaMemcpy(res.data,oth.data,size,cudaMemcpyDeviceToHost),"Copying %ld bytes from gpu to cpu",size);
#else
      memcpy(res.data,oth.data,size);
#endif
      
      return res;
    }
    
    /// Copy an SU3 field from GPU with CPU layout to CPU with CPU layout, with the same type
    ///
    /// \todo template and revert to above
    template <typename F>
    CpuSU3Field<F,StorLoc::ON_CPU>& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<F,StorLoc::ON_GPU>& oth)
    {
      /// Size to be copied
      const int64_t size=oth.vol*sizeof(F)*NCOL*NCOL*2;
      
#ifdef USE_CUDA
      DECRYPT_CUDA_ERROR(cudaMemcpy(res.data,oth.data,size,cudaMemcpyDeviceToHost),"Copying %ld bytes from gpu to cpu",size);
#else
      memcpy(res.data,oth.data,size);
#endif
      
      return res;
    }
    
    /// Copy an SU3 field from CPU with GPU layout to GPU with GPU layout, with the same type
    template <typename F>
    GpuSU3Field<F,StorLoc::ON_GPU>& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const GpuSU3Field<F,StorLoc::ON_CPU>& oth)
      {
	/// Size to be copied \todo please move somewhere more meaningful
	const int64_t size=oth.vol*sizeof(F)*NCOL*NCOL*2;
	
#ifdef USE_CUDA
	DECRYPT_CUDA_ERROR(cudaMemcpy(res.data,oth.data,size,cudaMemcpyHostToDevice),"Copying %ld bytes from cpu to gpu",size);
#else
	memcpy(res.data,oth.data,size);
#endif
	
	return res;
      }
    
    /// Copy an SU3 field from CPU with CPU layout to GPU with CPU layout, with the same type
    template <typename F>
    CpuSU3Field<F,StorLoc::ON_GPU>& deepCopy(CpuSU3Field<F,StorLoc::ON_GPU>& res,const CpuSU3Field<F,StorLoc::ON_CPU>& oth)
      {
	/// Size to be copied \todo please move somewhere more meaningful
	const int64_t size=oth.vol*sizeof(F)*NCOL*NCOL*2;
	
#ifdef USE_CUDA
	DECRYPT_CUDA_ERROR(cudaMemcpy(res.data,oth.data,size,cudaMemcpyHostToDevice),"Copying %ld bytes from cpu to gpu",size);
#else
	memcpy(res.data,oth.data,size);
#endif
	
	return res;
      }
    
    /// Copy an SU3 field from CPU with CPU layout to GPU with GPU layout
    template <typename F,
	      typename OF>
    GpuSU3Field<F,StorLoc::ON_GPU>& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      /// Temporary storage
      GpuSU3Field<F,StorLoc::ON_CPU> tmp(res.vol);
      
      tmp.deepCopy(oth);
      res.deepCopy(tmp);
      
      return res;
    }
    
    /// Copy an SU3 field from GPU with GPU layout to CPU with CPU layout
    template <typename F,
	      typename OF>
    CpuSU3Field<F,StorLoc::ON_CPU>& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const GpuSU3Field<OF,StorLoc::ON_GPU>& oth)
    {
      /// Temporary storage
      GpuSU3Field<OF,StorLoc::ON_CPU> tmp(res.vol);
      
      tmp.deepCopy(oth);
      res.deepCopy(tmp);
      
      return res;
    }
    
    /// Copy an SU3 field from CPU with GPU layout to CPU with CPU layout, possibly with different types
    template <typename F,
	      typename OF>
    CpuSU3Field<F,StorLoc::ON_CPU>& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const GpuSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      oth.sitesLoop([=](const int& iSite) mutable
		    {
		      UNROLLED_FOR(ic1,NCOL)
			UNROLLED_FOR(ic2,NCOL)
			  UNROLLED_FOR(ri,2)
			    res(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
		          UNROLLED_FOR_END;
		        UNROLLED_FOR_END;
		      UNROLLED_FOR_END;
		    });
      return res;
    }
    
    /// Copy an SU3 field from CPU with CPU layout to CPU with GPU layout, possibly with different type
    ///
    /// \todo this of course is the same, so should be templated
    template <typename F,
	      typename OF>
    GpuSU3Field<F,StorLoc::ON_CPU>& deepCopy(GpuSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
    {
      oth.sitesLoop([=](const int& iSite) mutable
		    {
		      UNROLLED_FOR(ic1,NCOL)
			UNROLLED_FOR(ic2,NCOL)
			  UNROLLED_FOR(ri,2)
			    res(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
		          UNROLLED_FOR_END;
		        UNROLLED_FOR_END;
		      UNROLLED_FOR_END;
		    });
      
      return res;
    }
  }
  
  /// Dispatch the correct copier
  template <typename T>
  template <typename O>
  T& SU3Field<T>::deepCopy(const O& oth)
  {
    return
      resources::deepCopy(this->crtp(),oth);
  }
  
  /////////////////////////////////////////////////////////////////
  
  /// Serializable version
  template <typename Fund>
  using IoSU3Field=
	       CpuSU3Field<Fund,StorLoc::ON_CPU>;
  
#ifdef USE_CUDA
  
  /// USe GPU version of the field
  template <typename Fund>
  using OptSU3Field=
	       GpuSU3Field<Fund,StorLoc::ON_GPU>;
  
#else
  
  /// USe SIMD version of the field
  template <typename Fund>
  using OptSU3Field=
	       SimdSU3Field<Fund,StorLoc::ON_CPU>;
  
#endif
}

#endif
