#ifndef SYMPY_EXPR_H
#define SYMPY_EXPR_H

// Includes from SymEngine
#include "symengine/eval_double.h"
#include "symengine/real_double.h"
#include "symengine/parser.h"
#include "symengine/symbol.h"
#include "symengine/subs.h"

namespace nest
{

class SympyExpr
{
private:
  SymEngine::RCP< const SymEngine::Basic > expr_;
  std::vector< SymEngine::RCP< const SymEngine::Basic > > symbols_;

public:
  void parse( const std::string expr, const size_t n_inputs )
  {
    assert( expr.size() > 0 );

    expr_ = SymEngine::parse( expr );

    symbols_.resize( n_inputs );
    for ( size_t i = 0; i < n_inputs; ++i )
    {
      symbols_[ i ] = SymEngine::symbol( "x_" + std::to_string( i ) );
    }
  }

  double eval( const std::vector< double > args ) const
  {
    assert( args.size() == symbols_.size() );

    SymEngine::RCP< const SymEngine::Basic > local_expr = expr_;
    for ( size_t i = 0; i < args.size(); ++i )
    {
      local_expr = SymEngine::subs( local_expr, { { symbols_[ i ], SymEngine::real_double( args [ i ] ) } } );
    }
    return std::real( SymEngine::eval_complex_double( *local_expr ) );
  }
};

} // namespace nest

#endif
