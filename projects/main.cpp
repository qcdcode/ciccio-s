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

#if 0

// Provides the assembly comment useful to catch the produced code for each data type
PROVIDE_ASM_DEBUG_HANDLE(sumProd,CpuSU3Field<float,StorLoc::ON_CPU>*);
PROVIDE_ASM_DEBUG_HANDLE(sumProd,GpuSU3Field<float,StorLoc::ON_GPU>*);
PROVIDE_ASM_DEBUG_HANDLE(sumProd,SimdSU3Field<float,StorLoc::ON_CPU>*);
PROVIDE_ASM_DEBUG_HANDLE(sumProd,CpuSU3Field<double,StorLoc::ON_CPU>*);
PROVIDE_ASM_DEBUG_HANDLE(sumProd,GpuSU3Field<double,StorLoc::ON_GPU>*);
PROVIDE_ASM_DEBUG_HANDLE(sumProd,SimdSU3Field<double,StorLoc::ON_CPU>*);

/// Compute a+=b*c
///
/// Arguments are caught as generic \a SU3Field so allow for static
/// polymorphism, then they must be cast to the actual type. This
/// might be hidden with some macro, if wanted, which prepends with _
/// the name of the argument, to be matched with an internal cast
/// which strips the _. We might also catch arguments with their
/// actual type, but we would loose control on the fact that they are
/// su3field
template <typename F1,
	  typename F2,
	  typename F3>
INLINE_FUNCTION void su3FieldsSumProd(SU3Field<F1>& _field1,const SU3Field<F2>& _field2,const SU3Field<F3>& _field3)
{
  // Cast to the actual type
  F1& field1=_field1;
  const F2& field2=_field2;
  const F3& field3=_field3;
  
  field1.sitesLoop(KERNEL_LAMBDA_BODY(const int iSite)
		   {
		     // This puts a bookmark in the assembly to check
		     // the compiler implementation
		     BOOKMARK_BEGIN_sumProd((F2*){});
		     
		     /// Take access to the looping site, so the
		     /// compiler needs not to recompute the full
		     /// index for each color components, and it has a
		     /// clearer view which makes it easier to produce
		     /// optimized code
		     auto f1=field1.site(iSite);
		     auto f2=field2.site(iSite);
		     auto f3=field3.site(iSite);
		     
		     // This could be moved to a dedicated routine but it he
		     
		     UNROLLED_FOR(i,NCOL)
		       UNROLLED_FOR(k,NCOL)
		         UNROLLED_FOR(j,NCOL)
		           {
			     // Unroll the complex product, since with
			     // gpu we have torn apart real and
			     // imaginay part
			     
			     /// Result real and imaginary part
			     auto& f1r=f1(i,j,RE);
			     auto& f1i=f1(i,j,IM);
			     
			     /// First opeand, real and imaginary
			     const auto f2r=f2(i,k,RE);
			     const auto f2i=f2(i,k,IM);
			     
			     /// Second operand, real and imaginary
			     const auto f3r=f3(k,j,RE);
			     const auto f3i=f3(k,j,IM);
			     
			     // Adds the real part
			     f1r+=f2r*f3r;
			     f1r-=f2i*f3i;
			     
			     // Adds the imaginary part of the product
			     f1i+=f2r*f3i;
			     f1i+=f2i*f3r;
			   }
		         UNROLLED_FOR_END;
		       UNROLLED_FOR_END;
		     UNROLLED_FOR_END;
		     
		     // End of the bokkmarked assembly section
		     BOOKMARK_END_sumProd((F2*){});
		   }
    );
}

/// Perform the test using Field as intermediate type
///
/// Allocates three copies of the field, and pass to the kernel
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
  LOGGER<<"Volume: "<<field.vol<<" dataset: "<<3*(double)field.vol*sizeof(SU3<Complex<Fund>>)/(1<<20) << " precision: " <<
    NAME_OF_TYPE(Fund) << " field: " << NAME_OF_TYPE(Field)<<" \t GFlops/s: "<<
    gFlopsPerSec<<"\t Check: "<<fieldRes(0,0,0,0)<<" "<<fieldRes(0,0,0,1)<<" time: "<<timeInSec<<endl;
}

template <typename I>
struct Coord;

template <typename...>
struct a;

template <typename...Args>
struct a<std::tuple<Coord<Args>...>>
{
  std::tuple<Coord<Args>*...> fa;
  
  static void print()
  {
    forEachInTuple(std::tuple<Args...>{},[](auto t){LOGGER<<typeid(t).name()<<endl;});
  }
};

#endif

template<typename T,
	 typename...Args>
constexpr int tuple_element_index_helper()
{
  int is[]={std::is_same<T,Args>::value...};
  
  for (int i = 0; i < (int)sizeof...(Args); i++)
    if (is[i])
      return i;

  return sizeof...(Args);
}

#if 0
/// Perform the tests on the given type (double/float)
template <typename Fund>           // Fundamental datatype
void test(const int vol,         ///< Volume to simulate
	  const int workReducer) ///< Reduce worksize to make a quick test
{
  a<std::tuple<Coord<int>,Coord<uint64_t>>>::print();

  constexpr int i=tuple_element_index_helper<int, int,double>();
  
  LOGGER<<std::get<i>(std::make_tuple(23320,9))<<endl;
  
  /// Number of iterations
  const int64_t nIters=400000000LL/vol/workReducer;
  
  /// Prepare the field configuration in the CPU format
  CpuSU3Field<Fund,StorLoc::ON_CPU> field(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int ic1=0;ic1<NCOL;ic1++)
      for(int ic2=0;ic2<NCOL;ic2++)
	for(int ri=0;ri<2;ri++)
	  field(iSite,ic1,ic2,ri)=
	    (ri+2*(ic2+NCOL*(ic1+NCOL*iSite)))/Fund((NCOL*NCOL*2)*(iSite+1));
  
  LOGGER<<"Volume: "<<vol<<" dataset: "<<3*(double)vol*sizeof(SU3<Complex<Fund>>)/(1<<20)<<endl;
  
  // Loop over three different layout and storage
  forEachInTuple(std::tuple<
		 SimdSU3Field<Fund,StorLoc::ON_CPU>*,
		 CpuSU3Field<Fund,StorLoc::ON_CPU>*,
		 GpuSU3Field<Fund,StorLoc::ON_GPU>*>{},
		 [&](auto t)
		 {
		   /// Field type to be used in the test
		   using F=
		     std::remove_reference_t<decltype(*t)>;
		   
		   test<F>(field,nIters);
		 });
  
  /////////////////////////////////////////////////////////////////
  
  LOGGER<<endl;
}

#endif

const SpaceTime LocVol{4};

using T=Tens<TensComps<SpaceTime,ColRow,ColCln,Compl>,double,StorLoc::ON_CPU>;

auto& subs1(T& t,int i,int j,int k)
{
  //std::index_sequence<T::hasComp<SpinRow>,T::hasMoreThanOneComps>& u="";
  
  ASM_BOOKMARK_BEGIN("subs1");
  auto& m=t[clRow(j)][clCln(k)][complComp(0)][spaceTime(i)];
  ASM_BOOKMARK_END("subs1");
  
  return m;
}

auto& subs2(T& t,int i,int j,int k)
{
  ASM_BOOKMARK_BEGIN("subs2");
  auto& m=t[spaceTime(i)][clRow(j)][clCln(k)][complComp(0)];
  ASM_BOOKMARK_END("subs2");
  
  return m;
}

auto& fund(T& t,int i,int j,int k)
{
  ASM_BOOKMARK_BEGIN("fund");
  
  auto& m=t.data.data.data[k+NColComp*(j+NColComp*i)];
  
  ASM_BOOKMARK_END("fund");
  
  return m;
}

void testOffsets()
{
  T t(LocVol);
  
  ThreadPool::loopSplit(spaceTime(0),LocVol,
			KERNEL_LAMBDA_BODY(const SpaceTime iVol)
			{
			  t;
			  // for(ColRow ic1{0};ic1<NColComp;ic1++)
			  //   for(ColCln ic2{0};ic2<NColComp;ic1++)
			  //     for(Compl ri{0};ri<2;ri++)
			  // 	t[iVol][ic1][ic2][ri]=
			  // 	  (ri+2*(ic2+NColComp*(ic1+NColComp*iVol)))/double((NColComp*NColComp*2)*(iVol+1));
			}
			);
  
  
  int i,j,k;
  std::cin>>i>>j>>k;
  LOGGER<<&subs1(t,i,j,k)<<" "<<&subs2(t,i,j,k)<<" "<<&fund(t,i,j,k)<<endl;
  
  //LOGGER<<"AAA "<<&k<<" "<<&r<<" "<<&s<<endl;
  
  //t[sp(0)];
  LOGGER<<t.data.stackAllocated<<endl;
  
}

/// inMmain is the actual main, which is where the main thread of the
/// pool is sent to work while the workers are sent in the background
void inMain(int narg,char **arg)
{
  testOffsets();
  
  SpinRow a;
  a=1;
    
  /// Workreducer is useful for speeding up the test
  int workReducer=1;
  
  if(narg>=2)
    {
      workReducer=atoi(arg[1]);
      LOGGER<<"WorkReducer: "<<workReducer<<endl;
    }
  
  // Loop ofer float and double
  forEachInTuple(std::tuple<float,double>{},
		 [&](auto t)
		 {
		   using Fund=decltype(t);
		   LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
		   LOGGER<<"                      "<<NAME_OF_TYPE(Fund)<<" version"<<endl;
		   LOGGER<<"/////////////////////////////////////////////////////////////////"<<endl;
		   
		   for(int volLog2=4;volLog2<20;volLog2++)
		     {
#if 0
		       const int vol=1<<volLog2;
		       test<Fund>(vol,workReducer);
#endif
		     }
		 });
  
}

/// This might be moved to the library, and \a inMain expected
int main(int narg,char **arg)
{
  initCiccios(inMain,narg,arg);
  
  finalizeCiccios();
  
  return 0;
}
