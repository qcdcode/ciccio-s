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

PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMD,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMD,float)

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
INLINE_FUNCTION void unrolledSumProd(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
  BOOKMARK_BEGIN_UnrolledSIMD(Fund{});
  
  const int it=omp_get_thread_num();
  const int nt=omp_get_num_threads();
  const int v=(simdField1.fusedVol+nt-1)/nt;
  for(int iFusedSite=v*it;iFusedSite<std::min(v*(it+1),simdField1.fusedVol);iFusedSite++)
    {
      auto a=simdField1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
      const auto &b=simdField2.simdSite(iFusedSite);
      const auto &c=simdField3.simdSite(iFusedSite);
      
      UNROLLED_FOR(i,NCOL)
	UNROLLED_FOR(k,NCOL)
	  UNROLLED_FOR(j,NCOL)
	   a.get(i,j).sumProd(b.get(i,k),c.get(k,j));
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      simdField1.simdSite(iFusedSite)=a;
    }
  
  BOOKMARK_END_UnrolledSIMD(Fund{});
}

/////////////////////////////////////////////////////////////////

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
INLINE_FUNCTION void unrolledSumProdOMP(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
#pragma omp parallel for // To be done when thread pool exists
  for(int iFusedSite=0;iFusedSite<simdField1.fusedVol;iFusedSite++)
    {
      auto a=simdField1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
      const auto &b=simdField2.simdSite(iFusedSite);
      const auto &c=simdField3.simdSite(iFusedSite);
      
      UNROLLED_FOR(i,NCOL)
	UNROLLED_FOR(k,NCOL)
	  UNROLLED_FOR(j,NCOL)
	   a.get(i,j).sumProd(b.get(i,k),c.get(k,j));
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
      
      simdField1.simdSite(iFusedSite)=a;
    }
}

/////////////////////////////////////////////////////////////////

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
INLINE_FUNCTION void unrolledSumProdPool(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
  threadPool->loopSplit(0,simdField1.fusedVol,
		       [&](const int& threadId,const int& iFusedSite) INLINE_ATTRIBUTE
		       {
			 auto a=simdField1.simdSite(iFusedSite); // This copy gets compiled away, and no alias is induced
			 const auto &b=simdField2.simdSite(iFusedSite);
			 const auto &c=simdField3.simdSite(iFusedSite);
			 
			 UNROLLED_FOR(i,NCOL)
			   UNROLLED_FOR(k,NCOL)
			     UNROLLED_FOR(j,NCOL)
			       a.get(i,j).sumProd(b.get(i,k),c.get(k,j));
			     UNROLLED_FOR_END;
			   UNROLLED_FOR_END;
			 UNROLLED_FOR_END;
			 
			 simdField1.simdSite(iFusedSite)=a;
		       }
			);
}

/////////////////////////////////////////////////////////////////

PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDAliasing,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDAliasing,float)

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
INLINE_FUNCTION void unrolledSumProdAliasing(SimdSu3Field<Fund>& simdField1,const SimdSu3Field<Fund>& simdField2,const SimdSu3Field<Fund>& simdField3)
{
  BOOKMARK_BEGIN_UnrolledSIMDAliasing(Fund{});
  
  //#pragma omp parallel for // To be done when thread pool exists
  simdField1.sumProd(simdField2,simdField3);
  
  BOOKMARK_END_UnrolledSIMDAliasing(Fund{});
}

/////////////////////////////////////////////////////////////////

PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDRestrict,double)
PROVIDE_ASM_DEBUG_HANDLE(UnrolledSIMDRestrict,float)

template <typename Fund>
struct A
{
  SimdComplex<Fund> data[NCOL*NCOL];
};

template <typename Fund>
struct B
{
  SimdComplex<Fund> data[NCOL*NCOL];
};

template <typename Fund>
struct C
{
  SimdComplex<Fund> data[NCOL*NCOL];
};

/// Unroll loops with metaprogramming, SIMD version
template <typename Fund>
INLINE_FUNCTION void unrolledSumProdRestrict(SimdSu3Field<Fund>&  simdField1,const SimdSu3Field<Fund>&  simdField2,const SimdSu3Field<Fund>&  simdField3)
{
  BOOKMARK_BEGIN_UnrolledSIMDRestrict(Fund{});
  
#define COMPLEX_SUM_PROD(A,B,C)						\
  a->data[(A)].real+=b->data[(B)].real*c->data[(C)].real;		\
  a->data[(A)].real-=b->data[(B)].imag*c->data[(C)].imag;		\
  a->data[(A)].imag+=b->data[(B)].real*c->data[(C)].imag;		\
  a->data[(A)].imag+=b->data[(B)].imag*c->data[(C)].real
  
#define S(A,B,C)				\
  COMPLEX_SUM_PROD(B+NCOL*A,C+NCOL*A,B+NCOL*C)
  
  //#pragma omp parallel for // To be done when thread pool exists
  for(int iFusedSite=0;iFusedSite<simdField1.fusedVol;iFusedSite++)
    {
      auto a=(A<Fund>*)&simdField1.simdSite(iFusedSite);
      const auto b=(const B<Fund>*)&simdField2.simdSite(iFusedSite);
      const auto c=(const C<Fund>*)&simdField3.simdSite(iFusedSite);
      
      UNROLLED_FOR(i,NCOL)
	UNROLLED_FOR(k,NCOL)
	  UNROLLED_FOR(j,NCOL)
	    S(i,j,k);
          UNROLLED_FOR_END;
        UNROLLED_FOR_END;
      UNROLLED_FOR_END;
    }
  BOOKMARK_END_UnrolledSIMDRestrict(Fund{});
}

/////////////////////////////////////////////////////////////////

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
template <int way,
	  typename Fund>
void simdTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSu3Field<Fund> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1=field;
  simdField2=field;
  simdField3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  switch(way)
    {
    case 0:
  #pragma omp parallel
      for(int64_t i=0;i<nIters;i++)
	unrolledSumProd(simdField1,simdField2,simdField3);
      break;
    case 1:
      for(int64_t i=0;i<nIters;i++)
	unrolledSumProdOMP(simdField1,simdField2,simdField3);
      break;
    case 2:
      for(int64_t i=0;i<nIters;i++)
	unrolledSumProdPool(simdField1,simdField2,simdField3);
    }
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<StorLoc::ON_CPU,Fund> fieldRes(field.vol);
  fieldRes=simdField1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"SIMD"<<way<<" \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<" time: "<<timeInSec<<endl;
}

/// Issue the test on SIMD field
template <typename Fund>
void simdAliasingTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSu3Field<Fund> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1=field;
  simdField2=field;
  simdField3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    unrolledSumProdAliasing(simdField1,simdField2,simdField3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<StorLoc::ON_CPU,Fund> fieldRes(field.vol);
  fieldRes=simdField1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"alSIMD \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<" time: "<<timeInSec<<endl;
}

/// Issue the test on SIMD field
template <typename Fund>
void simdRestrictTest(CpuSU3Field<StorLoc::ON_CPU,Fund>& field,const int64_t nIters,const double gFlops)
{
  /// Allocate three fields, this could be short-circuited through cast operator
  SimdSu3Field<Fund> simdField1(field.vol),simdField2(field.vol),simdField3(field.vol);
  simdField1=field;
  simdField2=field;
  simdField3=field;
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    unrolledSumProdRestrict(simdField1,simdField2,simdField3);
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  // Copy back
  CpuSU3Field<StorLoc::ON_CPU,Fund> fieldRes(field.vol);
  fieldRes=simdField1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<"reSIMD \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,RE)<<" "<<fieldRes(0,0,0,IM)<<" time: "<<timeInSec<<endl;
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
  
  LOGGER<<"Eigen\t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<a[0](0,0).real()<<" "<<a[0](0,0).imag()<<(std::is_same<float,Fund>::value?" (might differ by rounding)":"")<<" time: "<<timeInSec<<endl;
}

#endif

/// Perform the tests
template <typename Fund>
void test(const int vol)
{
  /// Number of iterations
  const int64_t nIters=400000000ULL/vol;
  
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
  
  simdTest<0>(field,nIters,gFlops);
  
  simdTest<1>(field,nIters,gFlops);
  
  simdTest<2>(field,nIters,gFlops);
  
  simdAliasingTest(field,nIters,gFlops);
  
  simdRestrictTest(field,nIters,gFlops);
  
  cpuTest(field,nIters,gFlops);
  
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
