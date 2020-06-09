#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#ifdef USE_EIGEN
 #include <eigen3/Eigen/Dense>
#endif

#include <chrono>

#include "ciccio-s.hpp"

using namespace ciccios;

/////////////////////////////////////////////////////////////////

PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMD,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMD,float)

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
ALWAYS_INLINE void unrolledSumProd(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
  BOOKMARK_BEGIN_UnrolledSIMD(Fund{});
  
  //#pragma omp parallel for // To be done when thread pool exists
  for(int iFusedSite=0;iFusedSite<simdField1.fusedVol;iFusedSite++)
    {
      auto a=simdField1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
      const auto &b=simdField2.simdSite(iFusedSite);
      const auto &c=simdField3.simdSite(iFusedSite);
      
      unrollFor<NCOL>([&](const int& i){
			 unrollFor<NCOL>([&](const int& k){
					    unrollFor<NCOL>([&](const int& j)
							    {
							      a[i][j].sumProd(b[i][k],c[k][j]);
							     });});});
      simdField1.simdSite(iFusedSite)=a;
    }
  
  BOOKMARK_END_UnrolledSIMD(Fund{});
}

PROVIDE_ASM_DEBUG_HANDLE(UnrolledCPU,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledCPU,float)

/// Unroll loops with metaprogramming, scalar version
template <typename Fund,
	  StorLoc SL=StorLoc::ON_CPU>
void unrolledSumProd(CpuSU3Field<SL,Fund>& field1,const CpuSU3Field<SL,Fund>& field2,const CpuSU3Field<SL,Fund>& field3)
{
  BOOKMARK_BEGIN_UnrolledCPU(Fund{});
  
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
  
  BOOKMARK_END_UnrolledCPU(Fund{});
}

/// Perform the non-simd CPU version of a+=b*c
template <typename Fund>
void cpuTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, and copy inside
  CpuSU3Field<StorLoc::ON_CPU,Fund> field1(field.vol),field2(field.vol),field3(field.vol);
  field1=field;
  field2=field;
  field3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    unrolledSumProd(field1,field2,field3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"CPU \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<field1(0,0,0,0)<<" "<<field1(0,0,0,1)<<endl;
}

/// Issue the test on SIMD field
template <typename Fund>
void simdTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSu3Field<Fund> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1=field;
  simdField2=field;
  simdField3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
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
  LOGGER<<"SIMD \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<endl;
}

#ifdef USE_EIGEN

PROVIDE_ASM_DEBUG_HANDLE(Eigen,double)
PROVIDE_ASM_DEBUG_HANDLE(Eigen,float)

/// Test eigen
template <typename Fund>
void eigenTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Copy volume
  const int vol=field.vol;
  
  /// Eigen equivalent of SU3
  using EQSU3=Eigen::Matrix<std::complex<Fund>,NCOL,NCOL>;
  
  /// Allocate three fields through Eigen
  std::vector<EQSU3,Eigen::aligned_allocator<EQSU3>> a(vol),b(vol),c(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	a[iSite](ic1,ic2)=
	  b[iSite](ic1,ic2)=
	  c[iSite](ic1,ic2)=
	  {field(iSite,ic1,ic2,RE),field(iSite,ic1,ic2,IM)};
  
  /// Starting instant
  const Instant start=takeTime();
  for(int64_t i=0;i<nIters;i++)
    {
      BOOKMARK_BEGIN_Eigen(Fund{});
      for(int iSite=0;iSite<vol;iSite++)
	a[iSite]+=b[iSite]*c[iSite];
      BOOKMARK_END_Eigen(Fund{});
    }
  
  /// Ending instant
  const Instant end=takeTime();
  
  const double timeInSec=timeDiffInSec(end,start);
  const double gFlopsPerSec=gFlops/timeInSec;
  
  LOGGER<<"Eigen\t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<a[0](0,0).real()<<" "<<a[0](0,0).imag()<<(std::is_same<float,Fund>::value?" (might differ by rounding)":"")<<endl;
}

#endif

/// Perform the tests
template <typename Fund>
void test(const int vol)
{
  /// Number of iterations
  const int nIters=40000000/vol;
  
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
	  field(iSite,ic1,ic2,ri)=(ri+2*(ic2+NCOL*(ic1+NCOL*iSite)))/Fund((NCOL*NCOL*2)*(iSite+1));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<Fund>>)/(1<<20)<<endl;
  
  cpuTest(field,nIters,gFlops);
  
#ifdef USE_EIGEN
  eigenTest(field,nIters,gFlops);
#endif
  
  simdTest(field,nIters,gFlops);
  
  /////////////////////////////////////////////////////////////////
  
  LOGGER<<endl;
}

template <typename Fund>
void testType()
{
  LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
  LOGGER<<"                      "<<nameOfType(Fund{})<<" version"<<endl;
  LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
  
  for(int volLog2=4;volLog2<20;volLog2++)
    {
      const int vol=1<<volLog2;
      test<Fund>(vol);
    }
}

/// Internal main
void inMain()
{
  testType<float>();
  
  testType<double>();
}

int main(int narg,char **arg)
{
  initCiccios(narg,arg,inMain);
  
  finalizeCiccios();
  
  return 0;
}
