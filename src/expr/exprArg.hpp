#ifndef _EXPR_ARG_HPP
#define _EXPR_ARG_HPP

/// \file exprArg.hpp
///
/// \brief Implements a conditional type for args

#include <base/metaProgramming.hpp>

namespace ciccios
{
  /// Returns a type or its reference
  template <typename T>
  using ExprArg=
    RefIf<T::takeAsArgByRef,T>;
}

#endif
