#ifndef _SU3FIELD_HPP
#define _SU3FIELD_HPP

#include "base/memoryManager.hpp"
#include "dataTypes/su3.hpp"

namespace ciccios
{
  /// SIMD su3 field
  ///
  /// Forward definition
  template <typename Fund>
  struct SimdSu3Field;
  
  /// Trivial su3 field
  template <StorLoc SL,
	    typename Fund>
  struct CpuSU3Field
  {
    /// Volume
    const int vol;
    
    /// Store wether this is a reference
    const bool isRef;
    
    /// Internal data
    Fund* data;
    
    /// Index function
    int index(int iSite,int icol1,int icol2,int reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iSite));
    }
    
    /// Access to data
    const Fund& operator()(int iSite,int icol1,int icol2,int reim) const
    {
      return data[index(iSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
    /// Access to the fused site
    const SU3<Complex<Fund>>& site(const int iSite) const
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
    
    /// Assign from a non-simd version
    CpuSU3Field& deepCopy(const CpuSU3Field<StorLoc::ON_CPU,Fund>& oth)
    {
      for(int iSite=0;iSite<vol;iSite++)
	{
	  for(int ic1=0;ic1<NCOL;ic1++)
	    for(int ic2=0;ic2<NCOL;ic2++)
	      for(int ri=0;ri<2;ri++)
		(*this)(iSite,ic1,ic2,ri)=oth(iSite,ic1,ic2,ri);
	}
      
      return *this;
    }
    
    /// Assign from a SIMD version
    CpuSU3Field& deepCopy(const SimdSu3Field<Fund>& oth);
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD version of the field
  template <typename Fund>
  struct SimdSu3Field
  {
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
    SimdSu3Field(int vol) : fusedVol(vol/simdLength<Fund>),isRef(false)
    {
      /// Compute the size
      int size=index(fusedVol,0,0,0);
      
      data=cpuMemoryManager->template provide<Simd<Fund>>(size);
    }
    
    /// Creates starting from the physical volume
    SimdSu3Field(const SimdSu3Field& oth) : fusedVol(oth.fusedVol),isRef(true),data(oth.data)
    {
    }
    
    /// Destroy
    ~SimdSu3Field()
    {
      if(not isRef)
	cpuMemoryManager->release(data);
    }
    
    /// Assign from a non-simd version
    SimdSu3Field& deepCopy(const CpuSU3Field<StorLoc::ON_CPU,Fund>& oth)
    {
      for(int iSite=0;iSite<fusedVol*simdLength<Fund>;iSite++)
	{
	  /// Index of the simd fused sites
	  const int iFusedSite=iSite/simdLength<Fund>;
	  
	  /// Index of the simd component
	  const int iSimdComp=iSite%simdLength<Fund>;
	  
	  for(int ic1=0;ic1<NCOL;ic1++)
	    for(int ic2=0;ic2<NCOL;ic2++)
	      for(int ri=0;ri<2;ri++)
		(*this)(iFusedSite,ic1,ic2,ri)[iSimdComp]=oth(iSite,ic1,ic2,ri);
	}
      
      return *this;
    }
    
    /// Sum the product of the two passed fields
    INLINE_FUNCTION SimdSu3Field& sumProd(const SimdSu3Field& oth1,const SimdSu3Field& oth2)
    {
      ASM_BOOKMARK_BEGIN("UnrolledSIMDmethod");
      
      for(int iFusedSite=0;iFusedSite<this->fusedVol;iFusedSite++)
	this->simdSite(iFusedSite).sumProd(oth1.simdSite(iFusedSite),oth2.simdSite(iFusedSite));
      
      ASM_BOOKMARK_END("UnrolledSIMDmethod");
      
      return *this;
    }
  };
  
  /// Assign from SIMD version
  template <StorLoc SL,
	    typename Fund>
  CpuSU3Field<SL,Fund>& CpuSU3Field<SL,Fund>::deepCopy(const SimdSu3Field<Fund>& oth)
  {
    for(int iFusedSite=0;iFusedSite<oth.fusedVol;iFusedSite++)
      
      for(int ic1=0;ic1<NCOL;ic1++)
	for(int ic2=0;ic2<NCOL;ic2++)
	  for(int ri=0;ri<2;ri++)
	    for(int iSimdComp=0;iSimdComp<simdLength<Fund>;iSimdComp++)
	      {
		const int iSite=iSimdComp+simdLength<Fund>*iFusedSite;
		
		(*this)(iSite,ic1,ic2,ri)=oth(iFusedSite,ic1,ic2,ri)[iSimdComp];
	      }
    
    return *this;
  }
}

#endif
