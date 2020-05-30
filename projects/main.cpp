#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <eigen3/Eigen/Dense>
#include <chrono>

#include <immintrin.h>

#include "ciccio-s.hpp"

#define SIMD_TYPE M256D

#if SIMD_TYPE == M256D

using Simd=__m256d;

#else

using Simd=std::array<double,1>;
inline Simd operator+=(Simd& a, const Simd& b)
{
  a[0]+=b[0];
  return a;
}

inline Simd operator*(const Simd&a, const Simd& b)
{
  Simd c;
  c[0]=a[0]*b[0];
  return c;
}

#endif

constexpr int simdSize=sizeof(Simd)/sizeof(double);

using namespace ciccios;



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

template <typename T>
struct Complex : public std::array<T,2>
{
  Complex operator*(const Complex& oth) const
  {
    const Complex& t=*this;
    Complex out;
    
    out[0]=t[0]*oth[0]-t[1]*oth[1];
    out[1]=t[0]*oth[1]+t[1]*oth[0];
    
    return out;
  }
  Complex& operator+=(const Complex& oth)
  {
    Complex& t=*this;
    
    t[0]+=oth[0];
    t[1]+=oth[1];
    
    return t;
  }
};

using SimdComplex=Complex<Simd>;

template <typename T,
	  int N>
struct ArithmeticArray : public std::array<T,N>
{
  template <typename U>
  auto operator*(const ArithmeticArray<U,N>& oth) const
  {
    ArithmeticArray<decltype(T()*U()),N> out={};
    
    for(int i=0;i<N;i++)
      out[i]+=(*this)[i]*oth[i];
    
    return out;
  }
  
  template <typename U>
  auto operator+=(const ArithmeticArray<U,N>& oth)
  {
    for(int i=0;i<N;i++)
      (*this)[i]+=oth[i];
    
    return *this;
  }
};

template <typename T,
	  int N>
struct ArithmeticMatrix : public std::array<std::array<T,N>,N>
{
  template <typename U>
  auto operator*(const ArithmeticMatrix<U,N>& oth) const
  {
    ArithmeticMatrix<decltype(T()*U()),N> out={};
    
#pragma unroll
    for(int ir=0;ir<N;ir++)
#pragma unroll
      for(int ic=0;ic<N;ic++)
#pragma unroll
	for(int i=0;i<N;i++)
	  {
	    ASM_BOOKMARK("FMA BEGIN");
	    out[ir][ic]+=(*this)[ir][i]*oth[i][ic];
	    ASM_BOOKMARK("FMA END");
	  }
    
    return out;
  }
  
  template <typename U>
  auto operator+=(const ArithmeticMatrix<U,N>& oth)
  {
#pragma unroll
    for(int ir=0;ir<N;ir++)
#pragma unroll
      for(int ic=0;ic<N;ic++)
	(*this)[ir][ic]+=oth[ir][ic];
    
    return *this;
  }
};

// template <typename T>
// using Color=ArithmeticArray<T,NCOL>;

template <typename T>
using SU3=ArithmeticMatrix<T,NCOL>;

template <typename T>
using QuadSU3=ArithmeticArray<SU3<T>,NDIM>;

using SimdQuadSU3=QuadSU3<SimdComplex>;

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
  
  SimdGaugeConf& sumProd(const SimdGaugeConf&oth1,const SimdGaugeConf& oth2)
  {
    ASM_BOOKMARK("here");
    
    auto a=(SimdQuadSU3*)(this->data);
    auto b=(SimdQuadSU3*)(oth1.data);
    auto c=(SimdQuadSU3*)(oth2.data);
    
    // long int a=0;
    //#pragma omp parallel for
    for(int iSimdSite=0;iSimdSite<this->simdVol;iSimdSite++)
      a[iSimdSite]+=b[iSimdSite]*c[iSimdSite];
    
    // LOGGER<<"Flops: "<<a<<endl;
    ASM_BOOKMARK("there");
    
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
		
		(*this)(iSimdSite,mu,ic1,ic2,ri)+=oth(iSimdSite,mu,ic1,ic2,ri);
		
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
  SimdGaugeConf simdConf3(vol);
  simdConf1=conf;
  simdConf2=conf;
  simdConf3=conf;
  
  Instant start=takeTime();
  
  const int nIters=100;
  for(int i=0;i<nIters;i++)
    simdConf1.sumProd(simdConf2,simdConf3);
  
  Instant end=takeTime();
  
  conf=simdConf1;
  const double timeInSec=timeDiffInSec(end,start);
  const double nFlopsPerSite=7.0*NCOL*NCOL*NCOL*NDIM,nGFlops=nFlopsPerSite*nIters*vol/1e9,gFlopsPerSec=nGFlops/timeInSec;
  LOGGER<<"Volume: "<<vol<<endl;
  LOGGER<<"Fantasy GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf(0,0,0,0,0)<<" "<<conf(0,0,0,0,1)<<endl;
  
  using EQSU3=std::array<Eigen::Matrix<std::complex<double>,NCOL,NCOL>,NDIM>;
  
  std::vector<EQSU3,Eigen::aligned_allocator<EQSU3>> a(vol),b(vol),c(vol);
  for(int i=0;i<vol;i++)
    for(int mu=0;mu<NDIM;mu++)
      {
	a[i][mu].fill({1.1,1.1});
	b[i][mu].fill({1.1,1.1});
	c[i][mu].fill({1.1,1.1});
      }
  
  start=takeTime();
  for(int i=0;i<nIters;i++)
    for(int i=0;i<vol;i++)
      for(int mu=0;mu<NDIM;mu++)
	{
	  ASM_BOOKMARK("EIG_BEGIN");
	  a[i][mu]+=b[i][mu]*c[i][mu];
	  ASM_BOOKMARK("EIG_END");
	}
  
  end=takeTime();
  {
    const double timeInSec=timeDiffInSec(end,start);
    const double nFlopsPerSite=7.0*NCOL*NCOL*NCOL*NDIM,nGFlops=nFlopsPerSite*nIters*vol/1e9,gFlopsPerSec=nGFlops/timeInSec;
  // LOGGER<<"Time in s: "<<timeInSec<<endl;
  // LOGGER<<"nFlopsPerSite: "<<nFlopsPerSite<<endl;
  // LOGGER<<"nGFlops: "<<nGFlops<<endl;
    LOGGER<<"Eigen GFlops/s: "<<gFlopsPerSec<<endl;
    LOGGER<<"Check: "<<a[0][0](0,0)<<endl;
  }
  LOGGER<<endl;
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
}

int main(int narg,char **arg)
{
  initCiccios(narg,arg);
  
  return 0;
}
