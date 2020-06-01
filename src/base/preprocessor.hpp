#ifndef _PREPROCESSOR_HPP
#define _PREPROCESSOR_HPP

/// Double macro to force symbol expansion
#define TO_STRING_INTERNAL(x) #x

/// Convert to a string
#define TO_STRING(x) TO_STRING_INTERNAL(x)

/// Instantiate the line number in a string
#define LINE_AS_STRING TO_STRING(__LINE__)

/// A string containing file location and line, as a string
#define HERE __FILE__ ":" LINE_AS_AS_STRING

/// Force unroll
#ifdef __GNUC__
 #define UNROLL(N) \
  PRAGMA(GCC unroll N)
#elif defined  __clang__
 #define UNROLL(N) \
  PRAGMA(unroll)
#endif
 
#endif
