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
void unrolledSumProd(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
  ASM_BOOKMARK_BEGIN("UnrolledSIMD");
  
  //#pragma omp parallel for
  for(int iFusedSite=0;iFusedSite<simdField1.fusedVol;iFusedSite++)
    {
      auto a=simdField1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
      const auto& b=simdField2.simdSite(iFusedSite);
      const auto& c=simdField3.simdSite(iFusedSite);
      
      unrollFor<NCOL>([&](const int& i){
			 unrollFor<NCOL>([&](const int& k){
					    unrollFor<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      simdField1.simdSite(iFusedSite)=a;
    }
  
  ASM_BOOKMARK_END("UnrolledSIMD");
}

/// Unroll loops with metaprogramming, scalar version
template <StorLoc SL=StorLoc::ON_CPU>
void unrolledSumProd(CpuSU3Field<SL,Fund>& field1,const CpuSU3Field<SL,Fund>& field2,const CpuSU3Field<SL,Fund>& field3)
{
  ASM_BOOKMARK_BEGIN("UnrolledCPU");
  
  //#pragma omp parallel for
  for(int iSite=0;iSite<field1.vol;iSite++)
    {
      auto a=field1.site(iSite); // Same as above
      const auto& b=field2.site(iSite);
      const auto& c=field3.site(iSite);
      
      unrollFor<NCOL>([&](const int& i){
			 unrollFor<NCOL>([&](const int& k){
					    unrollFor<NCOL>([&](const int& j)
							     {
							       a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      field1.site(iSite)=a;
    }
  
  ASM_BOOKMARK_END("AltUnrolledCPU");
}

/// Perform the non-simd CPU version of a+=b*c
void cpuTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int nIters,const double gFlops)
{
  /// Allocate three fields, and copy inside
  CpuSU3Field<StorLoc::ON_CPU,Fund> field1(field.vol),field2(field.vol),field3(field.vol);
  field1=field;
  field2=field;
  field3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    unrolledSumProd(field1,field2,field3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<field1(0,0,0,0)<<" "<<field1(0,0,0,1)<<endl;
}

void simdTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSu3Field<Fund> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1=field;
  simdField2=field;
  simdField3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    if(0) // Thsi performs "much worse" even if doing the same, most likely for aliasing issue
      simdField1.sumProd(simdField2,simdField3);
    else
      unrolledSumProd(simdField1,simdField2,simdField3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<StorLoc::ON_CPU,Fund> fieldRes(field.vol);
  fieldRes=simdField1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"SIMD GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<fieldRes(0,0,0,0)<<" "<<fieldRes(0,0,0,1)<<endl;
}

void test(const int vol,const int nIters=10000)
{
  /// Number of flops per site
  const double nFlopsPerSite=8.0*NCOL*NCOL*NCOL;
  
  /// Number of GFlops in total
  const double gFlops=nFlopsPerSite*nIters*vol/(1<<30);
  
  /// Prepare the fieldiguration in the CPU format
  CpuSU3Field<StorLoc::ON_CPU,Fund> field(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  field(iSite,ic1,ic2,ri)=ri+2*(ic2+NCOL*(ic1+NCOL*iSite));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<double>>)/(1<<20)<<endl;
  
  cpuTest(field,nIters,gFlops);
  
  simdTest(field,nIters,gFlops);
  
  /////////////////////////////////////////////////////////////////
  
  /// Eigen equivalent of SU3
  using EQSU3=Eigen::Matrix<std::complex<Fund>,NCOL,NCOL>;
  
  /// Allocate three fields through Eigen
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
