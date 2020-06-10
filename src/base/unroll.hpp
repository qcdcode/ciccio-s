#ifndef _UNROLL_HPP
#define _UNROLL_HPP

#include <utility>

namespace ciccios
{
  /// Force the compiler to inline
  ///
  /// \todo This is not very portable, let us investigate about other
  /// compilers
#define INLINE_ATTRIBUTE                           \
  __attribute__((always_inline))
  
  /// Force the compiler to inline a function
#define INLINE_FUNCTION				\
  INLINE_ATTRIBUTE inline
  
  /////////////////////////////////////////////////////////////////
  
  namespace resources
  {
    /// Wraps the function to be called
    ///
    /// Return an integer, to allow variadic expansion of the
    /// unrolling without any recursion
    template <typename F,
	      typename...Args>
    INLINE_FUNCTION int call(F&& f,Args&&...args)
    {
      f(std::forward<Args>(args)...);
      
      return 0;
    }
    
    /// Unroll a loop
    ///
    /// Actual implementation
    template <int...Is,
	      typename F>
    INLINE_FUNCTION void unrolledFor(std::integer_sequence<int,Is...>,F f)
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
  INLINE_FUNCTION void unrolledFor(const F& f)
  {
    resources::unrolledFor(std::make_integer_sequence<int, N>{},f);
  }
  
  /// Create an unrolled for
  ///
  /// Hides the complexity
#define UNROLLED_FOR(I,N)			\
  unrolledFor<N>([&](const auto& I) INLINE_ATTRIBUTE {
  
  /// Finish an unrolled for
#define UNROLLED_FOR_END })
}

#endif
