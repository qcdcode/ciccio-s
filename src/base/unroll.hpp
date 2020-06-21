#ifndef _UNROLL_HPP
#define _UNROLL_HPP

/// \file unroll.hpp
///
/// \brief Provides functions to unroll loops

#include <tuple>
#include <utility>

#include <gpu/cuda.hpp>

namespace ciccios
{
  namespace resources
  {
    /// Wraps the function to be called
    ///
    /// Return an integer, to allow variadic expansion of the
    /// unrolling without any recursion
    template <typename F,
	      typename...Args>
    INLINE_FUNCTION HOST DEVICE
    int call(F&& f,          ///< Function to be called
	     Args&&...args)  ///< Calling arguments
    {
      f(std::forward<Args>(args)...);
      
      return 0;
    }
    
    /// Unroll a loop
    ///
    /// Actual implementation
    template <int...Is,
	      typename F>
    INLINE_FUNCTION HOST DEVICE
    void unrolledForInternal(std::integer_sequence<int,Is...>,F&& f)
    {
      /// Dummy initialized list, discarded at compile time
      ///
      /// The attribute avoids compiler warning.
      [[ maybe_unused ]]
	auto list=
	{call(std::forward<F>(f),Is)...};
    }
  }
  
  /// Unroll a loop, wrapping the actual implementation
  template <int N,
	    typename F>
  INLINE_FUNCTION
  void unrolledFor(F&& f)
  {
    resources::unrolledForInternal(std::make_integer_sequence<int,N>{},std::forward<F>(f));
  }
  
#ifdef COMPILING_FOR_DEVICE
  
  /// Uses nvcc builtin
  ///
  /// \todo move it to a dedicated macro to call the proper bultin
#define UNROLLED_FOR(I,N)	\
  PRAGMA(unroll N)				\
  for(int I=0;I<N;I++)				\
    {
  
 #define UNROLLED_FOR_END			\
  }
  
#else
  
  /// Create an unrolled for
  ///
  /// Hides the complexity
 #define UNROLLED_FOR(I,N)			\
  unrolledFor<N>([&](const auto& I) INLINE_ATTRIBUTE {
  
  /// Finish an unrolled for
 #define UNROLLED_FOR_END })
  
#endif
  
  /////////////////////////////////////////////////////////////////
  
  namespace resources
  {
    /// Wraps the function to be called
    ///
    /// Identycal to the \c call function above, but it works only on
    /// host, as the auto lambda is not supported by nvcc
    template <typename F,
	      typename...Args>
    INLINE_FUNCTION
    int tupleCall(F&& f,Args&&...args)
    {
      f(std::forward<Args>(args)...);
      
      return 0;
    }
    
    /// Loop on tuple arguments
    ///
    /// Actual implementation
    template <int...Is,
	      typename Tp,
	      typename F>
    INLINE_FUNCTION
    void forEachInTupleInternal(std::integer_sequence<int,Is...>,Tp&& tp,F&& f)
    {
      /// Dummy initialized list, discarded at compile time
      ///
      /// The attribute avoids compiler warning.
      [[ maybe_unused ]]
	auto list=
	{tupleCall(std::forward<F>(f),std::get<Is>(tp))...};
    }
  }
  
  /// Loop on tuple arguments
  template <typename Tp,
	    typename F>
  INLINE_FUNCTION
  void forEachInTuple(Tp&& tp,F&& f)
  {
    resources::forEachInTupleInternal(std::make_integer_sequence<int,std::tuple_size<std::remove_reference_t<Tp>>::value>{},
				      std::forward<Tp>(tp),std::forward<F>(f));
  }
}

#endif
