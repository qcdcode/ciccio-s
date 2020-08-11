#ifndef _FIELDTENS_PROVIDER_HPP
#define _FIELDTENS_PROVIDER_HPP

/// \file fieldTensProvider.hpp
///
/// \brief Implements the remapping of field components into tensor,
/// constructor etc

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <fields/fieldTraits.hpp>

namespace ciccios
{
  /// Components for a given layout
  template <typename SPComp,
	    typename TC,
	    typename F,
	    StorLoc SL,
	    FieldLayout FL>
  struct FieldTensProvider
  {
    /// Field components
    using FC=
      FieldTraits<SPComp,TC,F,FL>;
    
    /// Components
    using Comps=
      typename FC::Comps;
    
    /// Fundamenal type
    using Fund=
      typename FC::Fund;
    
    /// Tensor type
    using T=
      Tens<Comps,Fund,SL,Stackable::CANNOT_GO_ON_STACK>;
    
    /// Tensor
    T t;
    
    /// Provide subscribe method
#define PROVIDE_SUBSCRIBE_OPERATOR(CONST_ATTR)				\
    /*! Subscribe a component via CRTP */				\
    template <typename OTC>						\
    decltype(auto) operator[]						\
    (const TensCompFeat<IsTensComp,OTC>& c) CONST_ATTR			\
    {									\
      return								\
	t[c.deFeat()];							\
    }
    
    PROVIDE_SUBSCRIBE_OPERATOR(/* non const*/);
    PROVIDE_SUBSCRIBE_OPERATOR(const);
    
#undef PROVIDE_SUBSCRIBE_OPERATOR
    
    /// Create from the components sizes
    template <typename...TD,
	      FieldLayout TFL=FL,
	      ENABLE_THIS_TEMPLATE_IF(sizeof...(TD)+1==
				      std::tuple_size<typename T::DynamicComps>::value)>
    FieldTensProvider(const TensCompFeat<IsTensComp,SPComp>& spaceTime,
		      const TensCompFeat<IsTensComp,TD>&...dynCompSize) :
      t(FC::adaptSpaceTime(spaceTime.deFeat()),dynCompSize.dFeat()...)
    {
    }
  };
}

#endif
