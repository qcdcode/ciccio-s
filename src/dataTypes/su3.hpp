#ifndef _SU3_HPP
#define _SU3_HPP

/// \file su3.hpp
///
/// Temporary file to setup su3 datatype

#include "dataTypes/arithmeticTensor.hpp"
#include "dataTypes/complex.hpp"
#include "dataTypes/SIMD.hpp"

namespace ciccios
{
  /// Number of colors
  constexpr int NCOL=3;
  
  /// NCOL x NCOL matrix
  template <typename T>
  using SU3=ArithmeticMatrix<T,NCOL>;
  
  /// Simd version of SU3 matrices
  template <typename Fund>
  using SimdSU3=SU3<SimdComplex<Fund>>;
}

#endif
