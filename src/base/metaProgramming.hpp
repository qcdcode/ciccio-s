#ifndef _METAPROGRAMMING_HPP
#define _METAPROGRAMMING_HPP

/// \file metaProgramming.hpp
///
/// \brief Implements many metaprogramming techniques
///
/// \todo Move it to a dedicated directory and separate
/// functionalities into more topical files

#include <type_traits>
#include <utility>

#include <gpu/cudaMacros.hpp>

namespace ciccios
{
  /////////////////////////////////////////////////////////////////
  
  /// Provides a SFINAE to be used in template par list
  ///
  /// This follows
  /// https://stackoverflow.com/questions/32636275/sfinae-with-variadic-templates
  /// as in this example
  /// \code
  /// template <typename D,
  ///           ENABLE_TEMPLATE_IF(std::is_same<D,int>::value)>
  /// void foo(D i) {} // fails if D is not int
  /// \endcode
#define ENABLE_TEMPLATE_IF(...)	\
  std::enable_if_t<(__VA_ARGS__),void*> =nullptr
  
  /////////////////////////////////////////////////////////////////
  
  /// Define a member detecter named hasMember_TAG
  ///
  /// Example:
  ///
  /// \code
  /// DEFINE_HAS_MEMBER(ciccio);
  ///
  /// struct fuffa
  /// {
  ///    int ciccio();
  /// };
  ///
  /// int main()
  /// {
  ///   bool has=hasMember_ciccio(fuffa);
  ///   return 0;
  /// }
  /// \endcode
#define PROVIDE_HAS_MEMBER(TAG)						\
  namespace impl							\
  {									\
    /*! Detect if \c Type has member (variable or method) TAG */	\
    /*!                                                       */	\
    /*! Forward definition                                    */	\
    template <typename Type,						\
	      bool IsClass>						\
      struct HasMember_ ## TAG;						\
    									\
    /*! Detect if \c Type has member (variable or method) TAG */	\
    /*!                                                       */	\
    /*! Internal implementation for class                     */	\
    template <typename Type>						\
      struct HasMember_ ## TAG<Type,true>				\
    {									\
      /*! Internal class of size 1, used if Type has the method */	\
      using Yes=char[1];						\
      									\
      /*! Internal class of size 2 used if Type has not the method */	\
      using No=char[2];							\
      									\
      /*! Internal class which does implement the method TAG */		\
      struct Fallback							\
      {									\
	/*! Member actually implemented */			        \
	int TAG;							\
      };								\
      									\
      /*! This class inherits from both Type and Fallback, so it will  */ \
      /*! certainly have the method TAG, possibly two                  */ \
      struct Derived :							\
	      public Type,						\
	      public Fallback						\
	{								\
	};								\
      									\
      /*! This type can be instantiated only if the U type is          */ \
      /*! unambiguosly understood.*/					\
      template <typename U,						\
		U>							\
    struct Check;							\
      									\
      /*! Forward definition of test function, taking a pointer to the */ \
      /*! type of TAG as argument, returning No. The instantiation     */ \
      /*! will fail if Base have two member TAG implemented, which     */ \
      /*! means that Type has the member                               */ \
      template <typename U>						\
	static No& test(Check<int Fallback::*,&U::TAG>*);		\
      									\
      /*! Forward definition of test function, taking a pointer to the */ \
      /*! type of TAG as argument, returning Yes. The instantiation   */ \
      /*! will work when the other fails, which means that Type does  */ \
      /*! have the member                                             */ \
      template <typename U>						\
	static Yes& test(...);						\
      									\
    public:								\
      /*! Result of the check, comparing the size of return type of */	\
      /*! test with the size of yes */					\
      static constexpr bool value=					\
	sizeof(test<Derived>(nullptr))==sizeof(Yes);			\
    };									\
  									\
    /*! Detect if \c Type has member (variable or method) TAG */	\
    /*!                                                       */	\
    /*! Internal implementation for not class                 */	\
    template <typename Type>						\
      struct HasMember_ ## TAG<Type,false>				\
    {									\
    public:								\
      /*! Result of the check, always false */				\
      static constexpr bool value=					\
	false;								\
    };									\
  }									\
									\
  /*! Detect if \c Type has member (variable or method) TAG          */	\
  template <typename Type>						\
  [[ maybe_unused ]]							\
  constexpr bool hasMember_ ## TAG=					\
    impl::HasMember_ ## TAG<Type,std::is_class<Type>::value>::value
  
  /////////////////////////////////////////////////////////////////
  
  /// Returns the argument as a constant
  template <typename T>
  constexpr const T& asConst(T& t) noexcept
  {
    return t;
  }
  
  /////////////////////////////////////////////////////////////////
  
  /// Remove \c const qualifier from any reference
  template <typename T,
	    ENABLE_TEMPLATE_IF(not std::is_pointer<T>::value)>
  constexpr T& asMutable(const T& v) noexcept
  {
    return const_cast<T&>(v);
  }
  
  /// Remove \c const qualifier from any pointer
  template <typename T>
  constexpr T* asMutable(const T* v) noexcept
  {
    return (T*)(v);
  }
  
  /// Return the type T or const T if B is true
  template <bool B,
	    typename T>
  using ConstIf=
    std::conditional_t<B,const T,T>;
  
  /// If the type is an l-value reference, provide the type T&, otherwise wih T
  template <typename T>
  using ref_or_val_t=
    std::conditional_t<std::is_lvalue_reference<T>::value,T&,T>;
  
  /// First part of the non-const method provider
#define _PROVIDE_ALSO_NON_CONST_METHOD_BEGIN				\
  /*! Overload the \c NAME const method passing all args             */ \
  template <typename...Ts> /* Type of all arguments                  */
  
  /// Body OF non-const method provider
#define _PROVIDE_ALSO_NON_CONST_METHOD_BODY(NAME)			\
  decltype(auto) NAME(Ts&&...ts) /*!< Arguments                      */ \
  {									\
    return asMutable(asConst(*this).NAME(std::forward<Ts>(ts)...)); \
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
  CUDA_HOST_DEVICE							\
  _PROVIDE_ALSO_NON_CONST_METHOD_BODY(NAME)
  
  /// Introduces the body of a loop
#if defined USE_CUDA
# define KERNEL_LAMBDA_BODY(A)			\
  [=] CUDA_HOST_DEVICE (A) mutable
#else
 # define KERNEL_LAMBDA_BODY(A)\
  [&] (A) __attribute__((always_inline))
#endif

  /// Dummy type that eats any argument
  template <typename...>
  struct DummyType
  {
  };
}

#endif
