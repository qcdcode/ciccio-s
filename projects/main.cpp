#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <eigen3/Eigen/Dense>
#include <chrono>

#include "ciccio-s.hpp"

using namespace ciccios;

/// Type used for the test
using Fund=double;

/////////////////////////////////////////////////////////////////

/// Unroll loops with metaprogramming, SIMD version
void unrolledSumProd(SimdGaugeConf<Fund>& simdConf1,const SimdGaugeConf<Fund>& simdConf2,const SimdGaugeConf<Fund>& simdConf3)
{
  ASM_BOOKMARK_BEGIN("UnrolledSIMD");
  
  //#pragma omp parallel for
  for(int iFusedSite=0;iFusedSite<simdConf1.fusedVol;iFusedSite++)
    {
      auto a=simdConf1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
      const auto& b=simdConf2.simdSite(iFusedSite);
      const auto& c=simdConf3.simdSite(iFusedSite);
      
      unrollFor<NCOL>([&](const int& i){
			 unrollFor<NCOL>([&](const int& k){
					    unrollFor<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      simdConf1.simdSite(iFusedSite)=a;
    }
  
  ASM_BOOKMARK_END("UnrolledSIMD");
}

/// Unroll loops with metaprogramming, scalar version
template <StorLoc SL=StorLoc::ON_CPU>
void unrolledSumProd(CPUGaugeConf<SL,Fund>& conf1,const CPUGaugeConf<SL,Fund>& conf2,const CPUGaugeConf<SL,Fund>& conf3)
{
  ASM_BOOKMARK_BEGIN("UnrolledCPU");
  
  //#pragma omp parallel for
  for(int iSite=0;iSite<conf1.vol;iSite++)
    {
      auto a=conf1.site(iSite); // Same as above
      const auto& b=conf2.site(iSite);
      const auto& c=conf3.site(iSite);
      
      unrollFor<NCOL>([&](const int& i){
			 unrollFor<NCOL>([&](const int& k){
					    unrollFor<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      conf1.site(iSite)=a;
    }
  
  ASM_BOOKMARK_END("AltUnrolledCPU");
}

/// Perform the non-simd CPU version of a+=b*c
void cpuTest(CPUGaugeConf<StorLoc::ON_CPU,Fund>& conf,const int nIters,const double gFlops)
{
  /// Allocate three confs, and copy inside
  CPUGaugeConf<StorLoc::ON_CPU,Fund> conf1(conf.vol),conf2(conf.vol),conf3(conf.vol);
  conf1=conf;
  conf2=conf;
  conf3=conf;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    unrolledSumProd(conf1,conf2,conf3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf1(0,0,0,0)<<" "<<conf1(0,0,0,1)<<endl;
}

void simdTest(CPUGaugeConf<StorLoc::ON_CPU,Fund>& conf,const int nIters,const double gFlops)
{
  /// Allocate three confs, this could be short-circuited through cast operator
  SimdGaugeConf<Fund> simdConf1(conf.vol),simdConf2(conf.vol),simdConf3(conf.vol);
  simdConf1=conf;
  simdConf2=conf;
  simdConf3=conf;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    if(0) // Thsi performs "much worse" even if doing the same, most likely for aliasing issue
      simdConf1.sumProd(simdConf2,simdConf3);
    else
      unrolledSumProd(simdConf1,simdConf2,simdConf3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CPUGaugeConf<StorLoc::ON_CPU,Fund> confRes(conf.vol);
  confRes=simdConf1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"SIMD GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<confRes(0,0,0,0)<<" "<<confRes(0,0,0,1)<<endl;
}

void test(const int vol,const int nIters=10000)
{
  /// Number of flops per site
  const double nFlopsPerSite=8.0*NCOL*NCOL*NCOL;
  
  /// Number of GFlops in total
  const double gFlops=nFlopsPerSite*nIters*vol/(1<<30);
  
  /// Prepare the configuration in the CPU format
  CPUGaugeConf<StorLoc::ON_CPU,Fund> conf(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  conf(iSite,ic1,ic2,ri)=ri+2*(ic2+NCOL*(ic1+NCOL*iSite));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<double>>)/(1<<20)<<endl;
  
  cpuTest(conf,nIters,gFlops);
  
  simdTest(conf,nIters,gFlops);
  
  /////////////////////////////////////////////////////////////////
  
  /// Eigen equivalent of SU3
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
  
  const Instant start=takeTime();
  for(int i=0;i<nIters;i++)
    for(int i=0;i<vol;i++)
      {
	ASM_BOOKMARK_BEGIN("EIGEN");
	a[i]+=b[i]*c[i];
	ASM_BOOKMARK_END("EIGEN");
      }
  
  const Instant end=takeTime();
  
  const double timeInSec=timeDiffInSec(end,start);
  const double gFlopsPerSec=gFlops/timeInSec;
  
  LOGGER<<"Eigen GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<a[0](0,0)<<endl;
  
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
