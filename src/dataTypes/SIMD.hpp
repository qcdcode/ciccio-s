#ifndef _SIMD_HPP
#define _SIMD_HPP

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#ifndef DISABLE_X86_INTRINSICS
 #include <immintrin.h>
#endif

#include <cstring>

#include <dataTypes/arithmeticTensor.hpp>

// \todo Rename Simd into something more appropriate

namespace ciccios
{
  /// Kinds of instruction set
  enum InstSet{NONE,MMX,AVX,AVX512};
  
  namespace resources
  {
    /// SIMD type for a given instruction set and fundamental
    ///
    /// Forward definition
    template <InstSet IS,
	      typename Fund>
    struct Simd;
    
    /// Provides the proper datatype for the given fundamental and instruction set
#define PROVIDE_SIMD(INST_SET,FUND,TYPE...)		\
    /*! SIMD FUND for instruction set INST_SET */	\
    template <>						\
    struct Simd<INST_SET,FUND>				\
    {							\
      /*! Provide the type*/				\
      using Type=TYPE;					\
    }
    
    PROVIDE_SIMD(NONE,float,ArithmeticArray<float,1>);
    PROVIDE_SIMD(NONE,double,ArithmeticArray<double,1>);
    
#ifndef DISABLE_X86_INTRINSICS
    
    PROVIDE_SIMD(AVX,float,__m256);
    PROVIDE_SIMD(AVX,double,__m256d);
    
    PROVIDE_SIMD(MMX,float,__m128);
    PROVIDE_SIMD(MMX,double,__m128d);
    
    PROVIDE_SIMD(AVX512,float,__m512);
    PROVIDE_SIMD(AVX512,double,__m512d);
    
#endif
    
#undef PROVIDE_SIMD
    
    /// Actual intinsic to be used
    template <typename Fund>
    using ActualSimd=typename resources::Simd<SIMD_INST_SET,Fund>::Type;
  }
  
  /// Length of a SIMD vector
  template <typename Fund>
  constexpr int simdLength=
    sizeof(resources::ActualSimd<Fund>)/sizeof(Fund);
  
  /// Simd datatype
  template <typename Fund>
  using Simd=
#ifndef COMPILING_FOR_DEVICE
    resources::ActualSimd<Fund>
#else
    ArithmeticArray<Fund,simdLength<Fund>>
#endif
    ;
}

#endif
