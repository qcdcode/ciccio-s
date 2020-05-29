#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <chrono>

#include <immintrin.h>

#include "ciccio-s.hpp"

using Simd=__m256d;
constexpr int simdSize=sizeof(Simd)/sizeof(double);

using namespace ciccios;

/// Position where to store the data: device or host
enum class StorLoc{ON_CPU
#ifdef USE_CUDA
		   ,ON_GPU
#endif
};

// constexpr StorLoc DEFAULT_STOR_LOC=
// #ifdef USE_CUDA
// 	    StorLoc::ON_GPU
// #else
// 	    StorLoc::ON_CPU
// #endif
// 	    ;

/// Wraps the memory manager
template <StorLoc>
struct MemoryManageWrapper;

/// Use memory manager
template <>
struct MemoryManageWrapper<StorLoc::ON_CPU>
{
  static auto& get()
  {
    return cpuMemoryManager;
  }
};

#ifdef USE_CUDA
/// Use memory manager
template <>
struct MemoryManageWrapper<StorLoc::ON_GPU>
{
  static auto& get()
  {
    return gpuMemoryManager;
  }
};
#endif

/// Gets the appropriate memory manager
template <StorLoc SL>
auto memoryManager()
{
  return MemoryManageWrapper<SL>::get();
}

constexpr int NDIM=4;
constexpr int NCOL=3;

struct SimdGaugeConf;

template <StorLoc SL>
struct CPUGaugeConf
{
  double* data;
  
  int index(int ivol,int mu,int icol1,int icol2,int reim) const
  {
    return reim+2*(icol2+NCOL*(icol1+NCOL*(mu+NDIM*ivol)));
  }
  
  const double& operator()(int ivol,int mu,int icol1,int icol2,int reim) const
  {
    return data[index(ivol,mu,icol1,icol2,reim)];
  }
  
  double& operator()(int ivol,int mu,int icol1,int icol2,int reim)
  {
    return data[index(ivol,mu,icol1,icol2,reim)];
  }
  
  CPUGaugeConf(int vol)
  {
    int size=index(vol,0,0,0,0);
    
    data=(double*)memoryManager<SL>()->template provide<double>(size);
  }
  
  ~CPUGaugeConf()
  {
    memoryManager<SL>()->release(data);
  }
  
  CPUGaugeConf& operator=(const SimdGaugeConf& oth);
};

/////////////////////////////////////////////////////////////////


struct SimdGaugeConf
{
  const int simdVol;
  
  Simd* data;
  
  int index(int ivol,int mu,int icol1,int icol2,int reim) const
  {
    return reim+2*(icol2+NCOL*(icol1+NCOL*(mu+NDIM*ivol)));
  }
  
  const Simd& operator()(int ivol,int mu,int icol1,int icol2,int reim) const
  {
    return data[index(ivol,mu,icol1,icol2,reim)];
  }
  
  Simd& operator()(int ivol,int mu,int icol1,int icol2,int reim)
  {
    return data[index(ivol,mu,icol1,icol2,reim)];
  }
  
  SimdGaugeConf(int vol) : simdVol(vol/simdSize)
  {
    int size=index(simdVol,0,0,0,0);
    
    data=cpuMemoryManager->template provide<Simd>(size);
  }
  
  ~SimdGaugeConf()
  {
    cpuMemoryManager->release(data);
  }
  
  SimdGaugeConf& operator=(const CPUGaugeConf<StorLoc::ON_CPU>& oth)
  {
    for(int iSite=0;iSite<simdVol*simdSize;iSite++)
      {
	const int iSimdSite=iSite/simdSize;
	const int iSimdComp=iSite%simdSize;
	
	for(int mu=0;mu<NDIM;mu++)
	  for(int ic1=0;ic1<NCOL;ic1++)
	    for(int ic2=0;ic2<NCOL;ic2++)
	      for(int ri=0;ri<2;ri++)
		
		(*this)(iSimdSite,mu,ic1,ic2,ri)[iSimdComp]=oth(iSite,mu,ic1,ic2,ri);
      }
    
    return *this;
  }
  
  SimdGaugeConf& operator*=(const SimdGaugeConf& oth)
  {
    ASM_BOOKMARK("here");
    
    // long int a=0;
    //#pragma omp parallel for
    for(int iSimdSite=0;iSimdSite<oth.simdVol;iSimdSite++)
      for(int mu=0;mu<NDIM;mu++)
    	for(int ic1=0;ic1<NCOL;ic1++)
    	  for(int ic2=0;ic2<NCOL;ic2++)
    	    for(int ri=0;ri<2;ri++)
    	      {
		(*this)(iSimdSite,mu,ic1,ic2,ri)*=oth(iSimdSite,mu,ic1,ic2,ri);
    // for(int i=0;i<oth.simdVol*NDIM*NCOL*NCOL*2;i++)
    //   {
    // 	this->data[i]*=oth.data[i];
    // 	a++;
    //   }
    
    // LOGGER<<"Flops: "<<a<<endl;
		ASM_BOOKMARK("there");
	      }
    
    return *this;
  }
  
  SimdGaugeConf& operator+=(const SimdGaugeConf& oth)
  {
    
    // long int a=0;
    //#pragma omp parallel for
    for(int iSimdSite=0;iSimdSite<oth.simdVol;iSimdSite++)
      for(int mu=0;mu<NDIM;mu++)
    	for(int ic1=0;ic1<NCOL;ic1++)
    	  for(int ic2=0;ic2<NCOL;ic2++)
    	    for(int ri=0;ri<2;ri++)
	      {
		ASM_BOOKMARK("here");
		
		  // auto& a=(*this)(iSimdSite,mu,ic1,ic2,ri);
		  // auto& b=(*this)(iSimdSite,mu,ic1,ic2,ri);
		  // auto& c=oth(iSimdSite,mu,ic1,ic2,ri);
		  // a=_mm256_add_pd(b,c);
		  (*this)(iSimdSite,mu,ic1,ic2,ri)+=oth(iSimdSite,mu,ic1,ic2,ri);
    // for(int i=0;i<oth.simdVol*NDIM*NCOL*NCOL*2;i++)
    //   {
    // 	this->data[i]*=oth.data[i];
    // 	a++;
    //   }
    
    // LOGGER<<"Flops: "<<a<<endl;
    ASM_BOOKMARK("there");
	      }
    return *this;
  }
};

template <>
CPUGaugeConf<StorLoc::ON_CPU>& CPUGaugeConf<StorLoc::ON_CPU>::operator=(const SimdGaugeConf& oth)
{
  for(int iSimdSite=0;iSimdSite<oth.simdVol;iSimdSite++)
    for(int mu=0;mu<NDIM;mu++)
      for(int ic1=0;ic1<NCOL;ic1++)
	for(int ic2=0;ic2<NCOL;ic2++)
	  for(int ri=0;ri<2;ri++)
	    for(int iSimdComp=0;iSimdComp<simdSize;iSimdComp++)
	      {
		const int iSite=iSimdComp+simdSize*iSimdSite;
		
		(*this)(iSite,mu,ic1,ic2,ri)=oth(iSimdSite,mu,ic1,ic2,ri)[iSimdComp];
	      }
  return *this;
}

/////////////////////////////////////////////////////////////////

/// Measure time
using Instant=std::chrono::time_point<std::chrono::steady_clock>;

inline Instant takeTime()
{
  return std::chrono::steady_clock::now();
}

/// Difference between two times
using Duration=decltype(Instant{}-Instant{});

double milliDiff(const Instant& end,const Instant& start)
{
  return std::chrono::duration<double,std::milli>(end-start).count();
}

void test(const int vol)
{
  CPUGaugeConf<StorLoc::ON_CPU> conf(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int mu=0;mu<NDIM;mu++)
      for(int ic1=0;ic1<NCOL;ic1++)
	for(int ic2=0;ic2<NCOL;ic2++)
	  for(int ri=0;ri<2;ri++)
	    conf(iSite,mu,ic1,ic2,ri)=1.1;
  
  SimdGaugeConf simdConf1(vol);
  SimdGaugeConf simdConf2(vol);
  simdConf1=conf;
  simdConf2=conf;
  
  Instant start=takeTime();

  const int nIters=10;
  for(int i=0;i<nIters;i++)
    simdConf1+=simdConf2;
  
  Instant end=takeTime();
  
  conf=simdConf1;
  const double timeInSec=milliDiff(end,start)/1000.0;
  const double nFlopsPerSite=2.0*NCOL*NCOL*NDIM,nGFlops=nFlopsPerSite*nIters*vol/1e9,gFlopsPerSec=nGFlops/timeInSec;
  LOGGER<<"Time in s: "<<timeInSec<<endl;
  LOGGER<<"nFlopsPerSite: "<<nFlopsPerSite<<endl;
  LOGGER<<"nGFlops: "<<nGFlops<<endl;
  LOGGER<<"GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf(0,0,0,0,0)<<endl;
  
  // conf=simdConf;
  // simdConf=conf;//(0,0,0,0,0)[0]=0.0;
}

/// Factorizes a number with a simple algorithm
void initCiccios(int& narg,char **&arg)
{
  
  initRanks(narg,arg);
  
  printBanner();
  
  printVersionAndCompileFlags(LOGGER);
  
  possiblyWaitToAttachDebugger();
  
  //CRASHER<<"Ciao"<<" amico"<<endl;
  
  cpuMemoryManager=new CPUMemoryManager;
  
  for(int volLog2=4;volLog2<20;volLog2++)
    {
      const int vol=1<<volLog2;
      test(vol);
    }
  
  delete cpuMemoryManager;
  
  LOGGER<<endl<<"Ariciao!"<<endl<<endl;
  
  finalizeRanks();
  
  //     print_banner();
    
//     //print version and configuration and compilation time
//     master_printf("\nInitializing NISSA, git hash: " GIT_HASH ", last commit at " GIT_TIME " with message: \"" GIT_LOG "\"\n");
//     master_printf("Configured at %s with flags: %s\n",compile_info[0],compile_info[1]);
//     master_printf("Compiled at %s of %s\n",compile_info[2],compile_info[3]);
    
// #ifdef USE_CUDA
//     init_cuda();
// #endif
    
//     //initialize the first vector of nissa
//     initialize_main_vect();
    
//     //initialize the memory manager
//     cpu_memory_manager=new CPUMemoryManager;
// #ifdef USE_CUDA
//     gpu_memory_manager=new GPUMemoryManager;
// #endif
    
    
//     //check endianness
//     check_endianness();
//     if(little_endian) master_printf("System endianness: little (ordinary machine)\n");
//     else master_printf("System endianness: big (BG, etc)\n");
    
//     //set scidac mapping
//     scidac_mapping[0]=0;
//     for(int mu=1;mu<NDIM;mu++) scidac_mapping[mu]=NDIM-mu;
    
//     for(int mu=0;mu<NDIM;mu++) all_dirs[mu]=1;
//     for(int mu=0;mu<NDIM;mu++)
//       for(int nu=0;nu<NDIM;nu++)
// 	{
// 	  only_dir[mu][nu]=(mu==nu);
// 	  all_other_dirs[mu][nu]=(mu!=nu);
// 	  all_other_spat_dirs[mu][nu]=(mu!=nu and nu!=0);
// 	}
//     //perpendicular dir
// #if NDIM >= 2
//     for(int mu=0;mu<NDIM;mu++)
//       {
// 	int nu=0;
// 	for(int inu=0;inu<NDIM-1;inu++)
// 	  {
// 	    if(nu==mu) nu++;
// 	    perp_dir[mu][inu]=nu;
// #if NDIM >= 3
// 	    int rho=0;
// 	    for(int irho=0;irho<NDIM-2;irho++)
// 	      {
// 		for(int t=0;t<2;t++) if(rho==mu||rho==nu) rho++;
// 		perp2_dir[mu][inu][irho]=rho;
// #if NDIM >= 4
// 		int sig=0;
// 		for(int isig=0;isig<NDIM-3;isig++)
// 		  {
// 		    for(int t=0;t<3;t++) if(sig==mu||sig==nu||sig==rho) sig++;
// 		    perp3_dir[mu][inu][irho][isig]=sig;
// 		    sig++;
// 		  } //sig
// #endif
// 		rho++;
// 	      } //rho
// #endif
// 	    nu++;
// 	  } //nu
// #endif
//       } //mu
    
//     //print fft implementation
// #if FFT_TYPE == FFTW_FFT
//     master_printf("Fast Fourier Transform: FFTW3\n");
// #else
//     master_printf("Fast Fourier Transform: NATIVE\n");
// #endif
    
// #if HIGH_PREC_TYPE == GMP_HIGH_PREC
//     mpf_precision=NISSA_DEFAULT_MPF_PRECISION;
// #endif
    
// #ifdef USE_HUGEPAGES
//     use_hugepages=NISSA_DEFAULT_USE_HUGEPAGES;
// #endif
    
//     //set default value for parameters
//     perform_benchmark=NISSA_DEFAULT_PERFORM_BENCHMARK;
//     verbosity_lv=NISSA_DEFAULT_VERBOSITY_LV;
//     use_128_bit_precision=NISSA_DEFAULT_USE_128_BIT_PRECISION;
//     use_eo_geom=NISSA_DEFAULT_USE_EO_GEOM;
//     use_Leb_geom=NISSA_DEFAULT_USE_LEB_GEOM;
//     warn_if_not_disallocated=NISSA_DEFAULT_WARN_IF_NOT_DISALLOCATED;
//     warn_if_not_communicated=NISSA_DEFAULT_WARN_IF_NOT_COMMUNICATED;
//     use_async_communications=NISSA_DEFAULT_USE_ASYNC_COMMUNICATIONS;
//     for(int mu=0;mu<NDIM;mu++) fix_nranks[mu]=0;
    
// #ifdef USE_VNODES
//     vnode_paral_dir=NISSA_DEFAULT_VNODE_PARAL_DIR;
// #endif

// #ifdef USE_DDALPHAAMG
//     master_printf("Linked with DDalphaAMG\n");
// #endif

// #ifdef USE_QUDA
// 	master_printf("Linked with QUDA, version: %d.%d.%d\n",QUDA_VERSION_MAJOR,QUDA_VERSION_MINOR,QUDA_VERSION_SUBMINOR);
// #endif
    
// #ifdef USE_EIGEN
//     master_printf("Linked with Eigen\n");
// #endif
    
// #ifdef USE_PARPACK
//     master_printf("Linked with Parpack\n");
// #endif
    
// #ifdef USE_EIGEN_EVERYWHERE
//     master_printf("Using Eigen everywhere\n");
// #endif
    
// #ifdef USE_PARPACK
//     use_parpack=NISSA_DEFAULT_USE_PARPACK;
// #endif
    
// #ifdef USE_GMP
//     master_printf("Linked with GMP\n");
// #endif
    
//     //put 0 as minimal request
//     recv_buf_size=0;
//     send_buf_size=0;
    
//     //read the configuration file, if present
//     read_nissa_config_file();
    
//     //setup the high precision
//     init_high_precision();
    
// #ifdef USE_DDALPHAAMG
//     DD::read_DDalphaAMG_pars();
// #endif
    
//     //initialize the base of the gamma matrices
//     init_base_gamma();
    
//     master_printf("Nissa initialized!\n");
    
//     const char DEBUG_LOOP_STRING[]="WAIT_TO_ATTACH";
//     if(getenv(DEBUG_LOOP_STRING)!=NULL)
//       debug_loop();
//     else
//       master_printf("To wait attaching the debugger please export: %s\n",DEBUG_LOOP_STRING);
}

int main(int narg,char **arg)
{
  initCiccios(narg,arg);
  
  return 0;
}
