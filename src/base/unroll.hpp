#ifndef _UNROLL_HPP
#define _UNROLL_HPP

#include <utility>

namespace ciccios
{
  /// Force the compiler to inline the function
  ///
  /// \todo This is not very portable, let us investigate about other
  /// compilers
#define ALWAYS_INLINE                           \
  __attribute__((always_inline)) inline
  
  /////////////////////////////////////////////////////////////////
  
  namespace resources
  {
    /// Wraps the function to be called
    ///
    /// Return an integer, to allow variadic expansion of the
    /// unrolling without any recursion
    template <typename F,
	      typename...Args>
    ALWAYS_INLINE int call(F&& f,Args&&...args)
    {
      f(std::forward<Args>(args)...);
      
      return 0;
    }
    
    /// Unroll a loop
    ///
    /// Actual implementation
    template <int...Is,
	      typename F>
    ALWAYS_INLINE void unrollFor(std::integer_sequence<int,Is...>,F f)
    {
      /// Dummy initialized list, discarded at compile time
      ///
      /// The attribute avoids compiler warning.
      [[ maybe_unused ]]auto list={call(f,Is)...};
    }
  }
  
  /// Unroll a loop, wrapping the actual implementation
  template <int N,
	    typename F>
  ALWAYS_INLINE void unrollFor(const F& f)
  {
    resources::unrollFor(std::make_integer_sequence<int, N>{},f);
  }
}

#endif
