#ifndef _COMPL_SUBSCRIBE_HPP
#define _COMPL_SUBSCRIBE_HPP

/// \file complSubscribe.hpp
///
/// \brief Provides a simple complex subscribe

#include <tensors/component.hpp>

namespace ciccios
{
  /// Real part index
  [[ maybe_unused ]]
  static constexpr auto RE=
    complComp(0);
  
  /// Imaginary part index
  [[ maybe_unused ]]
  static constexpr auto IM=
    complComp(1);
  
  /// Provides a simple complex subscribe
  template <typename T>
  struct ComplexSubscribe
  {
    PROVIDE_DEFEAT_METHOD(T);
    
    /// Returns the real part
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    decltype(auto) real()
      const
    {
      return
	this->deFeat()[RE];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(real);
    
    /// Returns the imaginary part
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
    decltype(auto) imag()
      const
    {
      return
	this->deFeat()[IM];
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_GPU(imag);
  };
}

#endif
