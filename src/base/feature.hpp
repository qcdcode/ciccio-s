#ifndef _FEATURE_HPP
#define _FEATURE_HPP

#include <gpu/cudaMacros.hpp>

namespace ciccios
{
  /// Provides a simple struct with the given name, to be used to catch a feature
#define DEFINE_FEATURE(NAME) \
  struct NAME		     \
  {			     \
  }
  
  /// Define a group of feature parametrized, all parametrized by a FeatName feature
#define DEFINE_FEATURE_GROUP(GROUP_FEATURE_NAME)	\
  template <typename FeatName,				\
	    typename Defeated>				\
  struct GROUP_FEATURE_NAME				\
  {							\
    PROVIDE_DEFEAT_METHOD(Defeated);			\
  }
  
  /// Provides the method to cast to the featuring class
#define PROVIDE_DEFEAT_METHOD(T)			\
  /*! Cast to the base type, with const attribute */	\
  CUDA_HOST_DEVICE					\
  operator const T&() const				\
  {							\
    return *static_cast<const T*>(this);		\
  }							\
  							\
  /*! Cast to the base type */						\
  /*!                       */						\
  /*! Cannot be achieved with the preprocessor macro, since */		\
  /*! the name of the method is weird */				\
  CUDA_HOST_DEVICE							\
  operator T&()								\
  {									\
    return *static_cast<T*>(this);					\
  }									\
  									\
  /*! Cast to the featuring class */					\
  CUDA_HOST_DEVICE							\
  const T& defeat() const						\
  {									\
    return *this;							\
  }									\
  									\
  PROVIDE_ALSO_NON_CONST_METHOD_GPU(defeat)
  
  /// Import method from the feature class
#define IMPORT_FEATURE_METHOD(A...)				\
  /*! Calls A in the base class */				\
  template <typename...Args>					\
  CUDA_HOST_DEVICE						\
  decltype(auto) A(Args&&...args) const				\
  {								\
    return (*this)().A(std::forward<Args>(args)...);		\
  }
}

#endif
