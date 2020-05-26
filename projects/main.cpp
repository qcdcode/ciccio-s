#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include "ciccio-s.hpp"

using namespace ciccios;


void initCiccios(int& narg,char **&arg)
{
  //init base things
  
  initRanks(narg,arg);
  
  printBanner();
  
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
