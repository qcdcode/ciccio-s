#ifndef _PREPROCESSOR_HPP
#define _PREPROCESSOR_HPP

/// \file preprocessor.hpp
///
/// Implement some macro to concatenate, unroll, stringify etc

#define TO_STRING_INTERNAL(x) #x

/// Put a pragma
#define PRAGMA(s)				\
  _Pragma(#s)

/// Convert to a string
#define TO_STRING(x) TO_STRING_INTERNAL(x)

/// Instantiate the line number in a string
#define LINE_AS_STRING TO_STRING(__LINE__)

/// A string containing file location and line, as a string
#define HERE __FILE__ ":" LINE_AS_AS_STRING

/// Concatenate two tokens
///
/// Internal implementation
#define _CONCAT2_IMPL(X,Y)			\
  X ## Y

/// Concatenate two tokens
///
///  Wrapper to beat CPP
#define _CONCAT2_WRAP(X,Y)			\
  _CONCAT2_IMPL(X,Y)

/// Concatenate two tokens
///
/// User accessible implementation
#define CONCAT2(X,Y)				\
  _CONCAT2_WRAP(X,Y)

/// Concatenate three tokens
#define CONCAT3(X,Y,Z)				\
  CONCAT2(CONCAT2(X,Y),Z)

/////////////////////////////////////////////////////////////////

/// Merges two tokens using a _
#define NAME2(X,Y)				\
  CONCAT3(X,_,Y)

/// Merges three tokens with two _
#define NAME3(X,Y,Z)				\
  NAME2(X,NAME2(Y,Z))

/// Merges four tokens with three _
#define NAME4(X,Y,W,Z)				\
  NAME2(X,NAME3(Y,W,Z))
  
/////////////////////////////////////////////////////////////////
  
/// Force unroll
#ifdef __GNUC__
 #define UNROLL(N) \
  PRAGMA(GCC unroll N)
#elif defined  __clang__
 #define UNROLL(N) \
  PRAGMA(unroll)
#endif
 
#endif
