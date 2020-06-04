#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <eigen3/Eigen/Dense>
#include <chrono>

#include "ciccio-s.hpp"

using namespace ciccios;

using Fund=double;

/////////////////////////////////////////////////////////////////

/// Unroll loops with metaprogramming
void unrolledSumProd(SimdGaugeConf<Fund>& simdConf1,const SimdGaugeConf<Fund>& simdConf2,const SimdGaugeConf<Fund>& simdConf3)
{
  ASM_BOOKMARK_BEGIN("AltUnrolled");
  
  //#pragma omp parallel for
  for(int iFusedSite=0;iFusedSite<simdConf1.fusedVol;iFusedSite++)
    {
      auto a=simdConf1.simdSite(iFusedSite);
      const auto& b=simdConf2.simdSite(iFusedSite);
      const auto& c=simdConf3.simdSite(iFusedSite);
      
      //così fa 77 vmovapd, se invece usiamo la versione dentro a gaugeconf ne fa 117, prova a spostare quanto sotto così comìè
      
      unrollLoopAlt<NCOL>([&](const int& i){
			 unrollLoopAlt<NCOL>([&](const int& k){
					    unrollLoopAlt<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      simdConf1.simdSite(iFusedSite)=a;
    }
  
  ASM_BOOKMARK_END("AltUnrolled");
}

/// Unroll loops with metaprogramming
template <StorLoc SL=StorLoc::ON_CPU>
void unrolledSumProd(CPUGaugeConf<SL,Fund>& conf1,const CPUGaugeConf<SL,Fund>& conf2,const CPUGaugeConf<SL,Fund>& conf3)
{
  ASM_BOOKMARK_BEGIN("AltUnrolledCPU");
  
  //#pragma omp parallel for
  for(int iSite=0;iSite<conf1.vol;iSite++)
    {
      auto a=conf1.site(iSite);
      const auto& b=conf2.site(iSite);
      const auto& c=conf3.site(iSite);
      
      //così fa 77 vmovapd, se invece usiamo la versione dentro a gaugeconf ne fa 117, prova a spostare quanto sotto così comìè
      
      unrollLoopAlt<NCOL>([&](const int& i){
			 unrollLoopAlt<NCOL>([&](const int& k){
					    unrollLoopAlt<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      conf1.site(iSite)=a;
    }
  
  ASM_BOOKMARK_END("AltUnrolledCPU");
}

template <int FMA=0>
void test(const int vol,const int nIters=10000)
{
  /// Number of flops per site
  const double nFlopsPerSite=8.0*NCOL*NCOL*NCOL;
  
  /// Number of GFlops in total
  const double nGFlops=nFlopsPerSite*nIters*vol/(1<<30);
  
  /// Prepare the configuration in the CPU format
  CPUGaugeConf<StorLoc::ON_CPU,Fund> conf(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  conf(iSite,ic1,ic2,ri)=ri+2*(ic2+NCOL*(ic1+NCOL*iSite));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<double>>)/(1<<20)<<endl;
  
  /////////////////////////////////////////////////////////////////
  
  /// Allocate three confs, this could be short-circuited through cast operator
  CPUGaugeConf<StorLoc::ON_CPU,Fund> conf1(vol),conf2(vol),conf3(vol);
  conf1=conf;
  conf2=conf;
  conf3=conf;
  
  /// Takes note of starting moment
  Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    unrolledSumProd(conf1,conf2,conf3);
  
  /// Takes note of ending moment
  Instant end=takeTime();
  
  
  {
    /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=nGFlops/timeInSec;
  LOGGER<<"GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf1(0,0,0,0)<<" "<<conf1(0,0,0,1)<<endl;
  }
  /////////////////////////////////////////////////////////////////
  
  /// Allocate three confs, this could be short-circuited through cast operator
  SimdGaugeConf<Fund> simdConf1(vol),simdConf2(vol),simdConf3(vol);
  simdConf1=conf;
  simdConf2=conf;
  simdConf3=conf;
  
  /// Takes note of starting moment
  start=takeTime();
  
  for(int i=0;i<nIters;i++)
    if(FMA%2==1)
      simdConf1.sumProd(simdConf2,simdConf3);
    else
      unrolledSumProd(simdConf1,simdConf2,simdConf3);
  
  /// Takes note of ending moment
  end=takeTime();
  
  // Copy back
  conf=simdConf1;
  
  {
    /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=nGFlops/timeInSec;
  LOGGER<<"Fantasy GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf(0,0,0,0)<<" "<<conf(0,0,0,1)<<endl;
  }
  /////////////////////////////////////////////////////////////////
  
  /// Eigen equivalent of 4xSU3
  using EQSU3=Eigen::Matrix<std::complex<Fund>,NCOL,NCOL>;
  
  /// Allocate three confs through Eigen
  std::vector<EQSU3,Eigen::aligned_allocator<EQSU3>> a(vol),b(vol),c(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
    {
      const int y=ic2+NCOL*(ic1+NCOL*iSite);
      a[iSite](ic1,ic2)=
	b[iSite](ic1,ic2)=
	c[iSite](ic1,ic2)={0.0+2*y,1.0+2*y};
    }
  
  start=takeTime();
  for(int i=0;i<nIters;i++)
    for(int i=0;i<vol;i++)
      {
	//ASM_BOOKMARK("EIG_BEGIN");
	a[i]+=b[i]*c[i];
	//ASM_BOOKMARK("EIG_END");
      }
  
  end=takeTime();
  {
    const double timeInSec=timeDiffInSec(end,start);
    const double gFlopsPerSec=nGFlops/timeInSec;
    
    LOGGER<<"Eigen GFlops/s: "<<gFlopsPerSec<<endl;
    LOGGER<<"Check: "<<a[0](0,0)<<endl;
  }
  LOGGER<<endl;
}

/// Internal main
void inMain()
{
  for(int volLog2=4;volLog2<20;volLog2++)
    {
      const int vol=1<<volLog2;
      test(vol);
    }
}

int main(int narg,char **arg)
{
  initCiccios(narg,arg,inMain);
  
  finalizeCiccios();
  
  return 0;
}
