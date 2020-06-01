#ifndef _SU3_HPP
#define _SU3_HPP

#include "dataTypes/arithmeticTensor.hpp"
#include "dataTypes/complex.hpp"
#include "dataTypes/SIMD.hpp"

// Temporary file to setup su3 datatype

namespace ciccios
{
  /// Number of dimension
  constexpr int NDIM=4;
  
  /// Number of colors
  constexpr int NCOL=3;
  
  /// NCOL x NCOL matrix
  template <typename T>
  using SU3=ArithmeticMatrix<T,NCOL>;
  
  /// Four SU3 matrices
  template <typename T>
  using QuadSU3=ArithmeticArray<SU3<T>,NDIM>;
  
  /// Simd version of 4 SU3 matrices
  using SimdQuadSU3=QuadSU3<SimdComplex>;
}

#endif
