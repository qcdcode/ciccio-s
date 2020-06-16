#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#ifdef USE_EIGEN
 #include <eigen3/Eigen/Dense>
#endif

#include <chrono>
#include <omp.h>

#include "ciccio-s.hpp"

using namespace ciccios;

/////////////////////////////////////////////////////////////////

PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDpool,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDpool,float)



/// Unroll loops with metaprogramming, SIMD version
template <typename F>
INLINE_FUNCTION void unrolledSumProdPool(SU3Field<F>& _field1,const SU3Field<F>& _field2,const SU3Field<F>& _field3)
{
  auto& field1=_field1.crtp();
  const auto& field2=_field2.crtp();
  const auto& field3=_field3.crtp();
  
  /// Fundamental type
  using Fund=typename F::BaseType;
  
  field1.sitesLoop([=](const int threadId,const int iFusedSite) mutable
		   {
		     BOOKMARK_BEGIN_UnrolledSIMDpool(Fund{});
		     
		     UNROLLED_FOR(i,NCOL)
		       UNROLLED_FOR(k,NCOL)
		         UNROLLED_FOR(j,NCOL)
		           {
			     auto& f1r=field1(iFusedSite,i,j,RE);
			     auto& f1i=field1(iFusedSite,i,j,IM);
			     
			     const auto& f2r=field2(iFusedSite,i,k,RE);
			     const auto& f2i=field2(iFusedSite,i,k,IM);
			     
			     const auto& f3r=field3(iFusedSite,k,j,RE);
			     const auto& f3i=field3(iFusedSite,k,j,IM);
			     
			     f1r+=f2r*f3r;
			     f1r-=f2i*f3i;
			     
			     f1i+=f2r*f3i;
			     f1i+=f2i*f3r;
			   }
		         UNROLLED_FOR_END;
		       UNROLLED_FOR_END;
		     UNROLLED_FOR_END;
		     
		     BOOKMARK_END_UnrolledSIMDpool(Fund{});
		   }
    );
}

/////////////////////////////////////////////////////////////////

PROVIDE_ASM_DEBUG_HANDLE(UnrolledCPU,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledCPU,float)

/// Unroll loops with metaprogramming, scalar version
template <typename Fund,
	  StorLoc SL=StorLoc::ON_CPU>
void unrolledSumProd(CpuSU3Field<Fund,SL>& field1,const CpuSU3Field<Fund,SL>& field2,const CpuSU3Field<Fund,SL>& field3)
{
  BOOKMARK_BEGIN_UnrolledCPU(Fund{});
  
  //#pragma omp parallel for
  for(int iSite=0;iSite<field1.vol;iSite++)
    {
      auto a=field1.site(iSite); // Same as above
      const auto& b=field2.site(iSite);
      const auto& c=field3.site(iSite);
      
      UNROLLED_FOR(i,NCOL)
	UNROLLED_FOR(k,NCOL)
	  UNROLLED_FOR(j,NCOL)
	    a.get(i,j).sumProd(b.get(i,k),c.get(k,j));
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      field1.site(iSite)=a;
    }
  
  BOOKMARK_END_UnrolledCPU(Fund{});
}

/// Perform the non-simd CPU version of a+=b*c
template <typename Fund>
void cpuTest(CpuSU3Field<Fund,StorLoc::ON_CPU>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, and copy inside
  CpuSU3Field<Fund,StorLoc::ON_CPU> field1(field.vol),field2(field.vol),field3(field.vol);
  field1.deepCopy(field);
  field2.deepCopy(field);
  field3.deepCopy(field);
  
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
void simdTest(CpuSU3Field<Fund,StorLoc::ON_CPU>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSU3Field<Fund,StorLoc::ON_CPU> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1.deepCopy(field);
  simdField2.deepCopy(field);
  simdField3.deepCopy(field);
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    unrolledSumProdPool(simdField1,simdField2,simdField3);
  ThreadPool::waitThatAllWorkersWaitForWork();
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<Fund,StorLoc::ON_CPU> fieldRes(field.vol);
  fieldRes.deepCopy(simdField1);
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"SIMD"<<" \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<" time: "<<timeInSec<<endl;
}

/////////////////////////////////////////////////////////////////

/// Issue the test on SIMD field
template <typename Fund>
void gpuTest(CpuSU3Field<Fund,StorLoc::ON_CPU>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  GpuSU3Field<Fund,StorLoc::ON_GPU> field1(field.vol),field2(field.vol),field3(field.vol);
  field1.deepCopy(field);
  field2.deepCopy(field);
  field3.deepCopy(field);
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    unrolledSumProdPool(field1,field2,field3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<Fund,StorLoc::ON_CPU> fieldRes(field.vol);
  fieldRes.deepCopy(field1);
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"GPU"<<" \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<" time: "<<timeInSec<<endl;
}

/////////////////////////////////////////////////////////////////

#ifdef USE_EIGEN

PROVIDE_ASM_DEBUG_HANDLE(Eigen,double)
PROVIDE_ASM_DEBUG_HANDLE(Eigen,float)

/// Test eigen
template <typename Fund>
void eigenTest(CpuSU3Field<Fund,StorLoc::ON_CPU>& field,const int64_t nIters,const double gFlops)
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
  
  LOGGER<<"Eigen\t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<a[0](0,0).real()<<" "<<a[0](0,0).imag()<<(std::is_same<float,Fund>::value?" (might differ by rounding)":"")<<" time: "<<timeInSec<<endl;
}

#endif

/// Perform the tests
template <typename Fund>
void test(const int vol)
{
  /// Number of iterations
  const int64_t nIters=40000000ULL/vol;
  
  /// Number of flops per site
  const double nFlopsPerSite=8.0*NCOL*NCOL*NCOL;
  
  /// Number of GFlops in total
  const double gFlops=nFlopsPerSite*nIters*vol/(1<<30);
  
  /// Prepare the fieldiguration in the CPU format
  CpuSU3Field<Fund,StorLoc::ON_CPU> field(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  field(iSite,ic1,ic2,ri)=(ri+2*(ic2+NCOL*(ic1+NCOL*iSite)))/Fund((NCOL*NCOL*2)*(iSite+1));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<Fund>>)/(1<<20)<<endl;
  
  simdTest(field,nIters,gFlops);
  
  cpuTest(field,nIters,gFlops);
  
  gpuTest(field,nIters,gFlops);
  
#ifdef USE_EIGEN
  eigenTest(field,nIters,gFlops);
#endif
  
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

int main(int narg,char **arg)
{
  initCiccios(narg,arg);
  
  testType<float>();
  
  testType<double>();
  
  finalizeCiccios();
  
  return 0;
}
