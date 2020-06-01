#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <eigen3/Eigen/Dense>
#include <chrono>

#include "ciccio-s.hpp"

using namespace ciccios;

/////////////////////////////////////////////////////////////////

void test(const int vol,const int nIters=100)
{
  /// Number of flops per site
  const double nFlopsPerSite=7.0*NCOL*NCOL*NCOL*NDIM;
  
  /// Number of GFlops in total
  const double nGFlops=nFlopsPerSite*nIters*vol/1e9;
  
  /// Prepare the configuration in the CPU format
  CPUGaugeConf<StorLoc::ON_CPU> conf(vol);
  for(int iSite=0;iSite<vol;iSite++)
    for(int mu=0;mu<NDIM;mu++)
      for(int ic1=0;ic1<NCOL;ic1++)
	for(int ic2=0;ic2<NCOL;ic2++)
	  for(int ri=0;ri<2;ri++)
	    conf(iSite,mu,ic1,ic2,ri)=1.1;
  
  /// Allocate three confs, this could be short-circuited through cast operator
  SimdGaugeConf simdConf1(vol),simdConf2(vol),simdConf3(vol);
  simdConf1=conf;
  simdConf2=conf;
  simdConf3=conf;
  
  /// Takes note of starting moment
  Instant start=takeTime();
  
  for(int i=0;i<nIters;i++)
    simdConf1.sumProd(simdConf2,simdConf3);
  
  /// Takes note of ending moment
  Instant end=takeTime();
  
  // Copy back
  conf=simdConf1;
  
  /// Compute time
  const double timeInSec=timeDiffInSec(end,start);
  
  /// Compute performances
  const double gFlopsPerSec=nGFlops/timeInSec;
  LOGGER<<"Volume: "<<vol<<endl;
  LOGGER<<"Fantasy GFlops/s: "<<gFlopsPerSec<<endl;
  LOGGER<<"Check: "<<conf(0,0,0,0,0)<<" "<<conf(0,0,0,0,1)<<endl;
  
  /////////////////////////////////////////////////////////////////
  
  /// Eigen equivalent of 4xSU3
  using EQSU3=std::array<Eigen::Matrix<std::complex<double>,NCOL,NCOL>,NDIM>;
  
  /// Allocate three confs through Eigen
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
    const double gFlopsPerSec=nGFlops/timeInSec;
    
    LOGGER<<"Eigen GFlops/s: "<<gFlopsPerSec<<endl;
    LOGGER<<"Check: "<<a[0][0](0,0)<<endl;
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
