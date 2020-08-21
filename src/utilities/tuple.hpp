#ifndef _TUPLE_HPP
#define _TUPLE_HPP

/// \file tuple.hpp
///
/// \brief Functionalities on top of std::tuple

#include <tuple>

#include <utilities/math.hpp>

namespace ciccios
{
  namespace impl
    {
      /// Helper to filter a tuple on the basis of a predicate
      ///
      /// Filter a single element, forward declaration
      template <bool,
		typename>
      struct _TupleFilterEl;
      
      /// Helper to filter a tuple on the basis of a predicate
      ///
      /// Filter a single element: True case, in which the type passes
      /// the filter
      template <typename T>
      struct _TupleFilterEl<true,T>
      {
	/// Helper type, used to cat the results
	using type=
	  std::tuple<T>;
	
	/// Filtered value
	const type value;
	
	/// Construct, taking a tuple type and filtering the valid casis
	template <typename Tp>
	_TupleFilterEl(Tp&& t) : ///< Tuple to filter
	  value{std::get<T>(t)}
	{
	}
      };
      
      /// Helper to filter a tuple on the basis of a predicate
      ///
      /// Filter a single element: False case, in which the type does
      /// not pass the filter
      template <typename T>
      struct _TupleFilterEl<false,T>
      {
	/// Helper empty type, used to cat the results
	using type=std::tuple<>;
	
	/// Empty value
	const type value{};
	
	/// Construct without getting the type
	template <typename Tp>
	_TupleFilterEl(Tp&& t) ///< Tuple to filter
	{
	}
      };
  }
  
  /// Returns a tuple, filtering out on the basis of a predicate
  template <template <class> class F,          // Predicate to be applied on the types
	    typename...T>                      // Types contained in the tuple to be filtered
  auto tupleFilter(const std::tuple<T...>& t) ///< Tuple to filter
  {
    return
      std::tuple_cat(impl::_TupleFilterEl<F<T>::value,T>{t}.value...);
  }
  
  /// Type obtained applying the predicate filter F on the tuple T
  template <template <class> class F,
	    typename T>
  using TupleFilter=
    decltype(tupleFilter<F>(*(T*)nullptr));
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Directly provides the result of filtering out from a tuple a give
    ///
    /// Forward definition
    template <typename F,
	      typename Tp>
    struct _TupleFilterOut;
    
    /// Cannot use directly the TupleFilter, because of some template template limitation
    template <typename...Fs,
	      typename...Tps>
    struct _TupleFilterOut<std::tuple<Fs...>,std::tuple<Tps...>>
    {
      /// Predicate to filter out
      template <typename T>
      struct Filter
      {
	/// Predicate result, counting whether the type match
	static constexpr bool value=
	  (sumAll<int>(std::is_same<T,Fs>::value...)==0);
      };
      
      /// Returned type
      typedef TupleFilter<Filter,std::tuple<Tps...>> type;
    };
  }
  
  /// Directly provides the result of filtering out the types of the tuple F from Tuple Tp
  template <typename F,
	    typename Tp>
  using TupleFilterOut=
    typename impl::_TupleFilterOut<F,Tp>::type;
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Predicate returning whether the type is present in the list a given number of times N
    ///
    /// Forward definition
    template <int N,       // Number of times that the type must be present
	      typename Tp> // Tuple in which to search
    struct _TypeIsInList;
    
    /// Predicate returning whether the type is present in the list
    template <int N,
	      typename...Tp>
    struct _TypeIsInList<N,std::tuple<Tp...>>
    {
      /// Internal implementation
      template <typename T>  // Type to search
      struct t
      {
	/// Predicate result
	static constexpr bool value=
	  (sumAll<int>(std::is_same<T,Tp>::value...)==N);
      };
    };
  }
  
  /// Returns whether the type T is in the tuple Tp N times
  template <typename T,
	    typename Tp,
	    int N=1>
  constexpr bool TupleHasType=
    impl::_TypeIsInList<N,Tp>::template t<T>::value;
  
  /// Returns a tuple containing all types common to the two tuples
  template <typename TupleToSearch,
	    typename TupleBeingSearched>
  using TupleCommonTypes=
    TupleFilter<impl::_TypeIsInList<1,TupleToSearch>::template t,TupleBeingSearched>;
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    template <typename T>
    /// Returns a tuple filled with a list of arguments
    ///
    /// Internal implementation, no more arguments to parse
    void _fillTuple(T&)
    {
    }
    
    /// Returns a tuple filled with a list of arguments
    ///
    /// Internal implementation, calls recursively
    template <typename T,
	      typename Head,
	      typename...Tail>
    void _fillTuple(T& t,                ///< Tuple to fill
		   const Head& head,    ///< Argument to fill
		   const Tail&...tail)  ///< Next arguments, filled recursively
    {
      std::get<Head>(t)=head;
      
      _fillTuple(t,tail...);
    }
  }
  
  /// Returns a tuple filled with the arguments of another tuple
  ///
  /// The arguments not passed are null-initialized
  template <typename T,     ///< Tuple type to be returned, to be provided
	    typename...Tp>  ///< Tuple arguments to be filled in
  T fillTuple(const std::tuple<Tp...>& in) ///< Tuple containing the arguments to be passed
  {
    /// Returned tuple
    T t;
    
    impl::_fillTuple(t,std::get<Tp>(in)...);
    
    return
      t;
  }
  
  namespace impl
  {
    template <typename I,
	      typename T>
    struct _TupleElOfList;
    
    template <std::size_t...Is,
	      typename T>
    struct _TupleElOfList<std::index_sequence<Is...>,T>
    {
      using type=
	std::tuple<std::tuple_element_t<Is,T>...>;
    };
  }
  
  /// Type of the tuple obtained removing last element
  template <typename Tp>
  using TupleAllButLast=
    typename impl::_TupleElOfList<std::make_index_sequence<std::tuple_size<Tp>::value-1>,Tp>::type;

  namespace impl
  {
    /// Type of the tuple obtained catting two tuples
    ///
    /// Forward declaration, internal implementation
    template <typename TP1,
	      typename TP2>
    struct _TupleCat;
    
    /// Type of the tuple obtained catting two tuples
    ///
    /// Internal implementation
    template <typename...Tp1,
	      typename...Tp2>
    struct _TupleCat<std::tuple<Tp1...>,
		    std::tuple<Tp2...>>
    {
      using type=
	std::tuple<Tp1...,Tp2...>;
    };
  }
  
  /// Type of the tuple obtained catting two tuples
  template <typename TP1,
	    typename TP2>
  using TupleCat=
    typename impl::_TupleCat<TP1,TP2>::type;
}

#endif
