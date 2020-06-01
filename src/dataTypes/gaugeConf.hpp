#ifndef _GAUGECONF_HPP
#define _GAUGECONF_HPP

#include "base/memoryManager.hpp"
#include "dataTypes/SU3.hpp"

namespace ciccios
{
  /// SIMD gauge conf
  ///
  /// Forward definition
  struct SimdGaugeConf;
  
  /// Trivial gauge conf
  template <StorLoc SL>
  struct CPUGaugeConf
  {
    /// Internal data
    double* data;
    
    /// Index function
    int index(int ivol,int mu,int icol1,int icol2,int reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*(mu+NDIM*ivol)));
    }
    
    /// Access to data
    const double& operator()(int ivol,int mu,int icol1,int icol2,int reim) const
    {
      return data[index(ivol,mu,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
    /// Create knowning volume
    CPUGaugeConf(int vol)
    {
      /// Compute size
      const int size=index(vol,0,0,0,0);
      
      data=(double*)memoryManager<SL>()->template provide<double>(size);
    }
    
    /// Destroy
    ~CPUGaugeConf()
    {
      memoryManager<SL>()->release(data);
    }
    
    /// Assign from a SIMD version
    CPUGaugeConf& operator=(const SimdGaugeConf& oth);
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// SIMD version of the conf
  struct SimdGaugeConf
  {
    /// Volume
    const int simdVol;
    
    /// Internal data
    Simd* data;
    
    int index(int ivol,int mu,int icol1,int icol2,int reim) const
    {
      return reim+2*(icol2+NCOL*(icol1+NCOL*(mu+NDIM*ivol)));
    }
    
    /// Access to data
    const Simd& operator()(int ivol,int mu,int icol1,int icol2,int reim) const
    {
      return data[index(ivol,mu,icol1,icol2,reim)];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(operator());
    
    /// Creates starting from the physical volume
    SimdGaugeConf(int vol) : simdVol(vol/simdLength)
    {
      int size=index(simdVol,0,0,0,0);
      
      data=cpuMemoryManager->template provide<Simd>(size);
    }
    
    /// Destroy
    ~SimdGaugeConf()
    {
      cpuMemoryManager->release(data);
    }
    
    /// Assign from a non-simd version
    SimdGaugeConf& operator=(const CPUGaugeConf<StorLoc::ON_CPU>& oth)
    {
      for(int iSite=0;iSite<simdVol*simdLength;iSite++)
	{
	  /// Index of the simd fused sites
	  const int iSimdSite=iSite/simdLength;
	  
	  /// Index of the simd component
	  const int iSimdComp=iSite%simdLength;
	  
	  for(int mu=0;mu<NDIM;mu++)
	    for(int ic1=0;ic1<NCOL;ic1++)
	      for(int ic2=0;ic2<NCOL;ic2++)
		for(int ri=0;ri<2;ri++)
		  (*this)(iSimdSite,mu,ic1,ic2,ri)[iSimdComp]=oth(iSite,mu,ic1,ic2,ri);
	}
      
      return *this;
    }
    
    /// Sum the prodcut of the two passed conf
    SimdGaugeConf& sumProd(const SimdGaugeConf&oth1,const SimdGaugeConf& oth2)
    {
      ASM_BOOKMARK("here");
      
      /// Take reference to the actual data, to convert to the arithmetic-aware datatype
      auto a=(SimdQuadSU3*)(this->data);
      auto b=(SimdQuadSU3*)(oth1.data);
      auto c=(SimdQuadSU3*)(oth2.data);
      
      //#pragma omp parallel for
      for(int iSimdSite=0;iSimdSite<this->simdVol;iSimdSite++)
	a[iSimdSite]+=b[iSimdSite]*c[iSimdSite];
      
      ASM_BOOKMARK("there");
      
      return *this;
    }
  };
  
  /// Assign from SIMD version
  template <>
  CPUGaugeConf<StorLoc::ON_CPU>& CPUGaugeConf<StorLoc::ON_CPU>::operator=(const SimdGaugeConf& oth)
  {
    for(int iSimdSite=0;iSimdSite<oth.simdVol;iSimdSite++)
      for(int mu=0;mu<NDIM;mu++)
	for(int ic1=0;ic1<NCOL;ic1++)
	  for(int ic2=0;ic2<NCOL;ic2++)
	    for(int ri=0;ri<2;ri++)
	      for(int iSimdComp=0;iSimdComp<simdLength;iSimdComp++)
		{
		  const int iSite=iSimdComp+simdLength*iSimdSite;
		  
		  (*this)(iSite,mu,ic1,ic2,ri)=oth(iSimdSite,mu,ic1,ic2,ri)[iSimdComp];
		}
    
    return *this;
  }
}

#endif
