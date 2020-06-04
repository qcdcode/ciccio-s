#ifndef _GAUGECONF_HPP
#define _GAUGECONF_HPP

#include "base/memoryManager.hpp"
#include "dataTypes/SU3.hpp"

namespace ciccios
{
  /// SIMD gauge conf
  ///
  /// Forward definition
  template <typename Fund>
  struct SimdGaugeConf;
  
  /// Trivial gauge conf
  template <StorLoc SL,
	    typename Fund>
  struct CPUGaugeConf
  {
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
    
    /// Create knowning volume
    CPUGaugeConf(int vol)
    {
      /// Compute size
      const int size=index(vol,0,0,0);
      
      data=(Fund*)memoryManager<SL>()->template provide<Fund>(size);
    }
    
    /// Destroy
    ~CPUGaugeConf()
    {
      memoryManager<SL>()->release(data);
    }
    
    /// Assign from a SIMD version
    CPUGaugeConf& operator=(const SimdGaugeConf<Fund>& oth);
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD version of the conf
  template <typename Fund>
  struct SimdGaugeConf
  {
    /// Volume
    const int fusedVol;
    
    /// Internal data
    Simd<Fund>* data;
    
    /// Index to internal data
    int index(const int iFusedSite,const int icol1,const int icol2,const int reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*iFusedSite));
    }
    
    /// Access to data
    const Simd<Fund>& operator()(const int iFusedSite,const int icol1,const int icol2,const int reim) const
    {
      return data[index(iFusedSite,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
    /// Access to the fused site
    const SimdSU3<Fund>& simdSite(const int iFusedSite) const
    {
      const Simd<Fund>& ref=(*this)(iFusedSite,0,0,0);
      
      return *reinterpret_cast<const SimdSU3<Fund>*>(&ref);
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(simdSite);
    
    /// Creates starting from the physical volume
    SimdGaugeConf(int vol) : fusedVol(vol/simdLength<Fund>)
    {
      int size=index(fusedVol,0,0,0);
      
      data=cpuMemoryManager->template provide<Simd<Fund>>(size);
    }
    
    /// Destroy
    ~SimdGaugeConf()
    {
      cpuMemoryManager->release(data);
    }
    
    /// Assign from a non-simd version
    SimdGaugeConf& operator=(const CPUGaugeConf<StorLoc::ON_CPU,Fund>& oth)
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
    
    /// Sum the product of the two passed conf
    SimdGaugeConf& sumProd(const SimdGaugeConf& oth1,const SimdGaugeConf& oth2)
    {
      ASM_BOOKMARK_BEGIN("AltUnrolled method");
      for(int iFusedSite=0;iFusedSite<this->fusedVol;iFusedSite++)
      	{
      	  auto a=this->simdSite(iFusedSite);
      	  const auto& b=oth1.simdSite(iFusedSite);
      	  const auto& c=oth2.simdSite(iFusedSite);
	  
	  unrollLoopAlt<3>([&](const int& ir){
			     unrollLoopAlt<3>([&](const int& ic){
						unrollLoopAlt<3>([&](const int& i){
								   a[ir][ic].sumProd(b[ir][i],c[i][ic]);
								 }
						  );}
			       );});
	  
	  // // 	    a.sumProd(b,c);
	  
	  this->simdSite(iFusedSite)=a;
	}
      ASM_BOOKMARK_END("AltUnrolled method");
      
      return *this;
    }
  };
  
  /// Assign from SIMD version
  template <StorLoc SL,
	    typename Fund>
  CPUGaugeConf<SL,Fund>& CPUGaugeConf<SL,Fund>::operator=(const SimdGaugeConf<Fund>& oth)
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
