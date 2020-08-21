#ifndef _MATH_HPP
#define _MATH_HPP

#include <functional>

#include <gpu/cudaMacros.hpp>

namespace ciccios
{
  /// Combine the the passed list of values
  template <typename F,
	    typename T,
	    typename...Ts>
  CUDA_HOST_DEVICE
  constexpr T binaryCombine(F&& f,
			    const T& init,
			    Ts&&...list)
  {
    /// Result
    T out=init;
    
    T l[]{(T)list...};
    
    for(auto& i : l)
      out=f(out,i);
    
    return out;
  }
  
  ///Product of the arguments
  template <typename T,
	    typename...Ts>
  CUDA_HOST_DEVICE
  constexpr auto productAll(Ts&&...t)
  {
    return binaryCombine(std::multiplies<>(),T{1},std::forward<Ts>(t)...);
  }
  
  ///Sum of the arguments
  template <typename T,
	    typename...Ts>
  CUDA_HOST_DEVICE
  constexpr auto sumAll(Ts&&...t)
  {
    return binaryCombine(std::plus<>(),T{0},std::forward<Ts>(t)...);
  }
}

#endif
