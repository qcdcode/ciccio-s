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
    auto& deepCopy(const O& oth);
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
  
  /// GPU version of  su3 field
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
    int index(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iSite));
    }
    
    /// Access to data
    const Fund& operator()(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
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
    
    /// Destroy
    ~CpuSU3Field()
    {
      if(not isRef)
	memoryManager<SL>()->release(data);
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
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD version of the field
  template <typename Fund,
	    StorLoc SL>
  struct SimdSU3Field : public SU3Field<SimdSU3Field<Fund,SL>>
  {
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
    int index(const int& iFusedSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iFusedSite));
    }
    
    /// Access to data
    const Simd<Fund>& operator()(const int& iFusedSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iFusedSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
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
    
    /// Creates starting from the physical volume
    SimdSU3Field(const SimdSU3Field& oth) : fusedVol(oth.fusedVol),isRef(true),data(oth.data)
    {
    }
    
    /// Destroy
    ~SimdSU3Field()
    {
      if(not isRef)
	cpuMemoryManager->release(data);
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
    void sitesLoop(F&& f)
    {
      ThreadPool::loopSplit(0,fusedVol,std::forward<F>(f));
    }
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// GPU version of  su3 field
  template <typename Fund,
	    StorLoc SL>
  struct GpuSU3Field : public SU3Field<GpuSU3Field<Fund,SL>>
  {
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
    int index(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return reim+2*(iSite+vol*(icol2+NCOL*icol1));
    }
    
    /// Access to data
    const Fund& operator()(const int& iSite,const int& icol1,const int& icol2,const int& reim) const
    {
      return data[index(iSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
    PROVIDE_ALSO_NON_CONST_METHOD(site);
    
    /// Create knowning volume
    GpuSU3Field(const int& vol) : vol(vol),isRef(false)
    {
      /// Compute size
      const int size=index(vol,0,0,0);
      
      data=(Fund*)memoryManager<SL>()->template provide<Fund>(size);
    }
    
    /// Destroy
    ~GpuSU3Field()
    {
      if(not isRef)
	memoryManager<SL>()->release(data);
    }
    
    /// Sum the product of the two passed fields
    INLINE_FUNCTION GpuSU3Field& sumProd(const GpuSU3Field& oth1,const GpuSU3Field& oth2)
    {
      ASM_BOOKMARK_BEGIN("UnrolledGPUmethod");
      for(int iSite=0;iSite<this->vol;iSite++)
      	{
      	  auto& a=this->GPUSite(iSite);
      	  const auto& b=oth1.GPUSite(iSite);
      	  const auto& c=oth2.GPUSite(iSite);
	  
	  a.sumProd(b,c);
	}
      ASM_BOOKMARK_END("UnrolledGPUmethod");
      
      return *this;
    }
    
    /// Loop over all sites
    template <typename F>
    void sitesLoop(F&& f)
    {
      CRASHER<<"Need kernel launcher"<<endl;
      //ThreadPool::loopSplit(0,fusedVol,std::forward<F>(f));
    }
  };
  
  namespace resources
  {
    /// Assign from a non-simd version
    template <typename F,
	      typename OF>
    auto& deepCopy(SimdSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
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
    auto& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<OF,StorLoc::ON_CPU>& oth)
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
    
    /// Copy an SU3 field within GPU, with GPU layout, and possible different types
    template <typename F,
	      typename OF>
    auto& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const GpuSU3Field<OF,StorLoc::ON_GPU>& oth)
      {
	CRASHER<<"Must be done with kernel"<<endl;
	return res;
      }
    
    /// Copy an SU3 field from CPU with GPU layout to GPU with GPU layout, with the same type
    template <typename F>
    auto& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const GpuSU3Field<F,StorLoc::ON_CPU>& oth)
      {
	CRASHER<<"To be fixed"<<endl;
	return res;
      }
    
    /// Copy an SU3 field from CPU with CPU layout to GPU with GPU layout, with the same type
    template <typename F>
    auto& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const CpuSU3Field<F,StorLoc::ON_CPU>& oth)
    {
      CRASHER<<"To be fixed"<<endl;
      return res;
    }
    
    /// Copy an SU3 field from GPU with GPU layout to CPU with CPU layout, with the same type
    template <typename F>
    auto& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const GpuSU3Field<F,StorLoc::ON_GPU>& oth)
    {
      CRASHER<<"To be fixed"<<endl;
      return res;
    }
    
  // /// Copy a SU3 field, from CPU with CPU layout to GPU with GPU layout, with the same type
    // template <typename F>
    // auto& deepCopy(GpuSU3Field<F,StorLoc::ON_GPU>& res,const CpuSU3Field<F,StorLoc::ON_CPU>& oth)
    //   {
    // 	CRASHER<<"To be fixed"<<endl;
    // 	return res;
    //   }
    
    /// Copy a SU3 field, from GPU with GPU layout to CPU with CPU layout, with the same type
    // template <typename F>
    // auto& deepCopy(CpuSU3Field<F,StorLoc::ON_CPU>& res,const CpuSU3Field<F,StorLoc::ON_GPU>& oth)
    //   {
    // 	CRASHER<<"To be fixed"<<endl;
    // 	return res;
    //   }
    
  // /// Assign from a different (or not) version, on the gpu
  // template <typename F,
  // 	    typename OF>
  // GpuSU3Field<StorLoc::ON_GPU,F>& deepCopy(GpuSU3Field<StorLoc::ON_GPU,F>& res,const GpuSU3Field<StorLoc::ON_GPU,OF>& oth)
  //   {
  //     CRASHER<<"To be fixed"<<endl;
  //     for(int ic1=0;ic1<NCOL;ic1++)
  // 	for(int ic2=0;ic2<NCOL;ic2++)
  // 	  for(int iSite=0;iSite<vol;iSite++)
  // 	    for(int ri=0;ri<2;ri++)
  // 	      res(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
      
  //     return res;
  //   }
    
  
  // /// Assign from CPU to GPU
  // template <typename F>
  // GpuSU3Field<StorLoc::ON_GPU,F>& GpuSU3Field<StorLoc::ON_GPU,F>::deepCopy(const GpuSU3Field<StorLoc::ON_CPU,F>& oth)
  //   {
  //     CRASHER<<"To be fixed"<<endl;
  //     for(int ic1=0;ic1<NCOL;ic1++)
  // 	for(int ic2=0;ic2<NCOL;ic2++)
  // 	  for(int iSite=0;iSite<vol;iSite++)
  // 	    for(int ri=0;ri<2;ri++)
  // 	      (*this)(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
      
  //     return *this;
  //   }
  
    
  }
  
  /// Dispatch the correct copier
  template <typename T>
  template <typename O>
  auto& SU3Field<T>::deepCopy(const O& oth)
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
