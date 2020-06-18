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

PROVIDE_ASM_DEBUG_HANDLE(SU3FieldsSumProd,double)
PROVIDE_ASM_DEBUG_HANDLE(SU3FieldsSumProd,float)

/// Compute a+=b*c
template <typename F>
INLINE_FUNCTION void su3FieldsSumProd(SU3Field<F>& _field1,const SU3Field<F>& _field2,const SU3Field<F>& _field3)
{
  /// Cast to actual type, to be moved inside operator()
  auto& field1=_field1.crtp();
  const auto& field2=_field2.crtp();
  const auto& field3=_field3.crtp();
  
  /// Fundamental type
  using Fund=typename F::BaseType;
  
  field1.sitesLoop([=] HOST DEVICE (const int iSite) mutable
		   {
		     BOOKMARK_BEGIN_SU3FieldsSumProd(Fund{});
		     
		     UNROLLED_FOR(i,NCOL)
		       UNROLLED_FOR(k,NCOL)
		         UNROLLED_FOR(j,NCOL)
		           {
			     auto& f1r=field1(iSite,i,j,RE);
			     auto& f1i=field1(iSite,i,j,IM);
			     
			     const auto f2r=field2(iSite,i,k,RE);
			     const auto f2i=field2(iSite,i,k,IM);
			     
			     const auto f3r=field3(iSite,k,j,RE);
			     const auto f3i=field3(iSite,k,j,IM);
			     
			     f1r+=f2r*f3r;
			     f1r-=f2i*f3i;
			     
			     f1i+=f2r*f3i;
			     f1i+=f2i*f3r;
			   }
		         UNROLLED_FOR_END;
		       UNROLLED_FOR_END;
		     UNROLLED_FOR_END;
		     
		     BOOKMARK_END_SU3FieldsSumProd(Fund{});
		   }
    );
}

/// Perform the test using Field as intermediate type
template <typename Field,
	  typename Fund>
void test(const CpuSU3Field<Fund,StorLoc::ON_CPU>& field,const int64_t nIters)
{
  /// Number of flops per site
  const double nFlopsPerSite=8.0*NCOL*NCOL*NCOL;
  
  /// Number of GFlops in total
  const double gFlops=nFlopsPerSite*nIters*field.vol/(1<<30);
  
  /// Allocate three fields, and copy inside
  Field field1(field.vol),field2(field.vol),field3(field.vol);
  field1.deepCopy(field);
  field2.deepCopy(field);
  field3.deepCopy(field);
  
  /// Takes note of starting moment
  const Instant start=takeTime();
  
  for(int64_t i=0;i<nIters;i++)
    su3FieldsSumProd(field1,field2,field3);
  ThreadPool::waitThatAllWorkersWaitForWork();
  
  /// Takes note of ending moment
  const Instant end=takeTime();
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  // Copy back
  CpuSU3Field<Fund,StorLoc::ON_CPU> fieldRes(field.vol);
  fieldRes.deepCopy(field1);
  
  /// Compute performances
  const double gFlopsPerSec=gFlops/timeInSec;
  LOGGER<<NAME_OF_TYPE(Field)<<" \t GFlops/s: "<<gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,0)<<" "<<fieldRes(0,0,0,1)<<" time: "<<timeInSec<<endl;
}

/// Perform the tests
template <typename Fund>
void test(const int vol,         ///< Volume to simulate
	  const int workReducer) ///< Reduce worksize to make a quick test
{
  /// Number of iterations
  const int64_t nIters=400000000LL/vol/workReducer;
  
  /// Prepare the field configuration in the CPU format
  CpuSU3Field<Fund,StorLoc::ON_CPU> field(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  field(iSite,ic1,ic2,ri)=(ri+2*(ic2+NCOL*(ic1+NCOL*iSite)))/Fund((NCOL*NCOL*2)*(iSite+1));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<Fund>>)/(1<<20)<<endl;
  
  forEachInTuple(std::tuple<
		 SimdSU3Field<Fund,StorLoc::ON_CPU>*,
		 CpuSU3Field<Fund,StorLoc::ON_CPU>*,
		 GpuSU3Field<Fund,StorLoc::ON_GPU>*>{},
		 [&](auto t)
		 {
		   /// Field type tp be used in the test
		   using F=
		     std::remove_reference_t<decltype(*t)>;
		   
		   test<F>(field,nIters);
		 });
  
  /////////////////////////////////////////////////////////////////
  
  LOGGER<<endl;
}

// template <typename Fund>
// void testType

int main(int narg,char **arg)
{
  initCiccios(narg,arg);
  
  int workReducer=1;
  if(narg>=2)
    workReducer=atoi(arg[1]);
  
  forEachInTuple(std::tuple<float,double>{},
		 [&](auto t)
		 {
		   using Fund=decltype(t);
		   LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
		   LOGGER<<"                      "<<NAME_OF_TYPE(Fund)<<" version"<<endl;
		   LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
		   
		   for(int volLog2=4;volLog2<20;volLog2++)
		     {
		       const int vol=1<<volLog2;
		       test<Fund>(vol,workReducer);
		     }
		 });
  // testType<float>(workReducer);
  
  // testType<double>(workReducer);
  
  finalizeCiccios();
  
  return 0;
}
