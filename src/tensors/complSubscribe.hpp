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
    
    /// The usual trick mess up slice IsConst detection
#define PROVIDE_RE_OR_IM_CONST_OR_NOT(REAL_OR_IMAG,RE_OR_IM,CONST_ATTR)	\
    /*! Returns the real part */					\
    INLINE_FUNCTION constexpr CUDA_HOST_DEVICE				\
    decltype(auto) REAL_OR_IMAG()					\
      CONST_ATTR							\
    {									\
      return								\
	this->deFeat()[RE_OR_IM];					\
    }
    
#define PROVIDE_RE_OR_IM_CONST_AND_NOT(REAL_OR_IMAG,RE_OR_IM)		\
    PROVIDE_RE_OR_IM_CONST_OR_NOT(REAL_OR_IMAG,RE_OR_IM,/* not const */) \
    PROVIDE_RE_OR_IM_CONST_OR_NOT(REAL_OR_IMAG,RE_OR_IM,const)		\
    
    PROVIDE_RE_OR_IM_CONST_AND_NOT(real,RE)
    PROVIDE_RE_OR_IM_CONST_AND_NOT(imag,RE)
    
#undef PROVIDE_RE_OR_IM_CONST_AND_NOT
#undef PROVIDE_RE_OR_IM_CONST_OR_NOT
  };
}

#endif
