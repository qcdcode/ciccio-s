#ifndef _METAPROGRAMMING_HPP
#define _METAPROGRAMMING_HPP

#include <type_traits>
#include <utility>

#include <gpu/cudaMacros.hpp>

namespace ciccios
{
  /// Returns the argument as a constant
  template <typename T>
  constexpr const T& asConst(T& t) noexcept
  {
    return t;
  }
  
  /// Provides a SFINAE to be used in template par list
  ///
  /// This follows
  /// https://stackoverflow.com/questions/32636275/sfinae-with-variadic-templates
  /// as in this example
  /// \code
  /// template <typename D,
  ///           SFINAE_ON_TEMPLATE_ARG(std::is_same<D,int>::value)>
  /// void foo(D i) {} // fails if D is not int
  /// \endcode
#define SFINAE_ON_TEMPLATE_ARG(...)	\
  std::enable_if_t<(__VA_ARGS__),void*> =nullptr
  
  /// Returns true if T is a const lvalue reference
  template <typename T>
  constexpr bool is_const_lvalue_reference_v=std::is_lvalue_reference<T>::value and std::is_const<std::remove_reference_t<T>>::value;
  
  /// Returns the type without "const" attribute if it is a reference
  template <typename T>
  HOST DEVICE
  decltype(auto) remove_const_if_ref(T&& t)
  {
    using Tv=std::remove_const_t<std::remove_reference_t<T>>;
    
    return (std::conditional_t<is_const_lvalue_reference_v<T>,Tv&,Tv>)t;
  }
  
  /// If the type is an l-value reference, provide the type T&, otherwise wih T
  template <typename T>
  using ref_or_val_t=std::conditional_t<std::is_lvalue_reference<T>::value,T&,T>;
  
  /// First part of the non-const method provider
#define _PROVIDE_ALSO_NON_CONST_METHOD_BEGIN				\
  /*! Overload the \c NAME const method passing all args             */ \
  template <typename...Ts> /* Type of all arguments                  */
  
  /// Body OF non-const method provider
#define _PROVIDE_ALSO_NON_CONST_METHOD_BODY(NAME)			\
  decltype(auto) NAME(Ts&&...ts) /*!< Arguments                      */ \
  {									\
    return remove_const_if_ref(asConst(*this).NAME(std::forward<Ts>(ts)...)); \
  }
  
  /// Provides also a non-const version of the method \c NAME
  ///
  /// See Scott Meyers
  /// https://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
  /// One or more const method NAME must be already present, the
  /// correct one will be chosen perfectly forwarding the arguments
  ///
  ///Example
  ///
  /// \code
  // class ciccio
  /// {
  ///   double e{0};
  ///
  /// public:
  ///
  ///   const double& get() const
  ///   {
  ///     return e;
  ///   }
  ///
  ///   PROVIDE_ALSO_NON_CONST_METHOD(get);
  /// };
  /// \endcode
#define PROVIDE_ALSO_NON_CONST_METHOD(NAME)				\
  _PROVIDE_ALSO_NON_CONST_METHOD_BEGIN					\
  _PROVIDE_ALSO_NON_CONST_METHOD_BODY(NAME)
  
#define PROVIDE_ALSO_NON_CONST_METHOD_GPU(NAME)				\
  _PROVIDE_ALSO_NON_CONST_METHOD_BEGIN					\
  HOST DEVICE								\
  _PROVIDE_ALSO_NON_CONST_METHOD_BODY(NAME)
  
  /// Implements the CRTP pattern
  template <typename T>
  struct Crtp
  {
    /// Cast to the base type
    ///
    /// This is customarily done by ~ operator, but I don't like it
    const T& crtp() const
    {
      return *static_cast<const T*>(this);
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD(crtp);
  };
}

#endif
