#ifndef _TUPLE_HPP
#define _TUPLE_HPP

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
      struct TupleFilterEl;
      
      /// Helper to filter a tuple on the basis of a predicate
      ///
      /// Filter a single element: True case, in which the type passes
      /// the filter
      template <typename T>
      struct TupleFilterEl<true,T>
      {
	/// Helper type, used to cat the results
	using type=std::tuple<T>;
	
	/// Filtered value
	const type value;
	
	/// Construct, taking a tuple type and filtering the valid casis
	template <typename Tp>
	TupleFilterEl(Tp&& t) : ///< Tuple to filter
	  value{std::get<T>(t)}
	{
	}
      };
      
      /// Helper to filter a tuple on the basis of a predicate
      ///
      /// Filter a single element: False case, in which the type does
      /// not pass the filter
      template <typename T>
      struct TupleFilterEl<false,T>
      {
	/// Helper empty type, used to cat the results
	using type=std::tuple<>;
	
	/// Empty value
	const type value{};
	
	/// Construct without getting the type
	template <typename Tp>
	TupleFilterEl(Tp&& t) ///< Tuple to filter
	{
	}
      };
  }
  
  /// Returns a tuple, filtering out on the basis of a predicate
  template <template <class> class F,          // Predicate to be applied on the types
	    typename...T>                      // Types contained in the tuple to be filtered
  auto tupleFilter(const std::tuple<T...>& t) ///< Tuple to filter
  {
    return std::tuple_cat(impl::TupleFilterEl<F<T>::value,T>{t}.value...);
  }
  
  /// Type obtained applying the predicate filter F on the tuple T
  template <template <class> class F,
	    typename T>
  using TupleFilter=decltype(tupleFilter<F>(*(T*)nullptr));
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Directly provides the result of filtering out from a tuple a give
    ///
    /// Forward definition
    template <typename F,
	      typename Tp>
    struct TupleFilterOut;
    
    /// Cannot use directly the TupleFilter, because of some template template limitation
    template <typename...Fs,
	      typename...Tps>
    struct TupleFilterOut<std::tuple<Fs...>,std::tuple<Tps...>>
    {
      /// Predicate to filter out
      template <typename T>
      struct Filter
      {
	/// Predicate result, counting whether the type match
	static constexpr bool value=(sumAll<int>(std::is_same<T,Fs>::value...)==0);
      };
      
      /// Returned type
      typedef TupleFilter<Filter,std::tuple<Tps...>> type;
    };
  }
  
  /// Directly provides the result of filtering out the types of the tuple F from Tuple Tp
  template <typename F,
	    typename Tp>
  using TupleFilterOut=
    typename impl::TupleFilterOut<F,Tp>::type;
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Predicate returning whether the type is present in the list a given number of times N
    ///
    /// Forward definition
    template <int N,       // Number of times that the type must be present
	      typename Tp> // Tuple in which to search
    struct TypeIsInList;
    
    /// Predicate returning whether the type is present in the list
    template <int N,
	      typename...Tp>
    struct TypeIsInList<N,std::tuple<Tp...>>
    {
      /// Internal implementation
      template <typename T>  // Type to search
      struct t
      {
	/// Predicate result
	static constexpr bool value=(sumAll<int>(std::is_same<T,Tp>::value...)==N);
      };
    };
  }
  
  /// Returns a tuple containing all types common to the two tuples
  template <typename TupleToSearch,
	    typename TupleBeingSearched>
  using TupleCommonTypes=
    TupleFilter<impl::TypeIsInList<1,TupleToSearch>::template t,TupleBeingSearched>;
}

#endif
