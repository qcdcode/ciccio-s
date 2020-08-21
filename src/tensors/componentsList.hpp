#ifndef _COMPONENTS_LIST_HPP
#define _COMPONENTS_LIST_HPP

#include <tuple>

/// \file componentsList.hpp
///
/// \brief Implements a list of components

#include <utilities/tuple.hpp>
#include <tensors/component.hpp>

namespace ciccios
{
  /// Collection of components
  template <typename...Tc>
  using TensComps=
    std::tuple<Tc...>;
  
  /// Returns the number of components of a tensComp
  template <typename T>
  constexpr int nOfComps=
    std::tuple_size<typename T::Comps>::value;
  
  namespace impl
  {
    /// Provides the result of filtering from a list of components the Row or Column
    ///
    /// Forward definition
    template <RwCl F,
	      typename TC>
    struct _TensCompsFilterRwCl;
    
    /// Cannot use directly the TupleFilter, because of some template template limitation
    template <RwCl F,
	      typename...Tc>
    struct _TensCompsFilterRwCl<F,TensComps<Tc...>>
    {
      /// Predicate to filter
      ///
      /// Forward definition
      template <typename T>
      struct Filter;
      
      /// Predicate to filter
      template <typename S,
		RwCl RC,
		int Which>
      struct Filter<TensComp<S,RC,Which>>
      {
	/// Predicate result, counting whether the type match
	static constexpr
	bool value=
	  (RC==F);
      };
      
      /// Returned type
      typedef TupleFilter<Filter,TensComps<Tc...>> type;
    };
  }
  
  /// Filter all Row components
  template <typename TC>
  using TensCompsFilterRow=
    typename impl::_TensCompsFilterRwCl<RwCl::ROW,TC>::type;
  
  /// Filter all Col components
  template <typename TC>
  using TensCompsFilterCln=
    typename impl::_TensCompsFilterRwCl<RwCl::CLN,TC>::type;
  
  /// Filter all Any components
  template <typename TC>
  using TensCompsFilterAny=
    typename impl::_TensCompsFilterRwCl<RwCl::ANY,TC>::type;
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Transposes a list of components
    ///
    /// Actual implementation, forward declaration
    template <typename TC>
    struct _TensCompsTransp;
    
    /// Transposes a list of components
    ///
    /// Actual implementation
    template <typename...S,
	      RwCl...RC,
	      int...Which>
    struct _TensCompsTransp<TensComps<TensComp<S,RC,Which>...>>
    {
      /// Resulting type
      using type=
	TensComps<TensComp<S,transp<RC>,Which>...>;
    };
  }
  
  /// Transposes a list of components
  template <typename TC>
  using TensCompsTransp=
    typename impl::_TensCompsTransp<TC>::type;
}

#endif
