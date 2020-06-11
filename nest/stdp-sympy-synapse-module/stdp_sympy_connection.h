/*
 *  stdp_sympy_connection.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef STDP_SYMPY_CONNECTION_H
#define STDP_SYMPY_CONNECTION_H

// C++ includes:
#include <cmath>
#include <string>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

#include "../include/sympy_expr.h"

using namespace nest;

namespace mynest
{

/** @BeginDocumentation
Name: stdp_sympy_synapse - Synapse type for spike-timing dependent
plasticity with configurable faciliation and depression terms.

Description:

stdp_sympy_synapse is a connector to create synapses with spike time
dependent plasticity with configurable facilitation and depresstion terms.

Parameters:

tau_plus   double - Time constant of STDP window, potentiation in ms
                    (tau_minus defined in post-synaptic neuron)
lambda     double - Step size
alpha      double - Asymmetry parameter (scales depressing increments as
                    alpha*lambda)
Wmax       double - Maximum allowed weight
expr_depress SympyExpr - Expression for depression branch
expr_facilitate SympyExpr - Expression for facilitation branch


Transmits: SpikeEvent


Author: Maximilian Schmidt

*/
// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template
template < typename targetidentifierT >
class STDPSympyConnection : public Connection< targetidentifierT >
{

public:
  typedef CommonSynapseProperties CommonPropertiesType;
  typedef Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  STDPSympyConnection();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  STDPSympyConnection( const STDPSympyConnection& );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( Event& e, thread t, const CommonSynapseProperties& cp );


  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    port
    handles_test_event( SpikeEvent&, rport )
    {
      return invalid_port_;
    }
  };

  void
  check_connection( Node& s,
    Node& t,
    rport receptor_type,
    const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  double
  facilitate_( double w, double kplus )
  {
    double delta_w = expr_facilitate_.eval( { w / Wmax_, kplus } );
    double norm_w = ( w / Wmax_ ) + lambda_ * delta_w;
    return norm_w < 1.0 ? norm_w * Wmax_ : Wmax_;
  }

  double
  depress_( double w, double kminus )
  {
    double delta_w = expr_depress_.eval( { w / Wmax_, kminus } );
    double norm_w = ( w / Wmax_ ) - alpha_ * lambda_ * delta_w;
    return norm_w > 0.0 ? norm_w * Wmax_ : 0.0;
  }

  // data members of each connection
  double weight_;
  double tau_plus_;
  double lambda_;
  double alpha_;
  double Wmax_;
  double Kplus_;
  SympyExpr expr_depress_;
  SympyExpr expr_facilitate_;
  std::string str_expr_depress_;
  std::string str_expr_facilitate_;
  long n_inputs_;

  double t_lastspike_;
};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
STDPSympyConnection< targetidentifierT >::send( Event& e,
  thread t,
  const CommonSynapseProperties& )
{
  // synapse STDP depressing/facilitation dynamics
  double t_spike = e.get_stamp().get_ms();

  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  Node* target = get_target( t );
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t1, t2] from post-synaptic neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;

  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
  // details.
  target->get_history( t_lastspike_ - dendritic_delay,
    t_spike - dendritic_delay,
    &start,
    &finish );
  // facilitation due to post-synaptic spikes since last pre-synaptic spike
  double minus_dt;
  while ( start != finish )
  {
    minus_dt = t_lastspike_ - ( start->t_ + dendritic_delay );
    ++start;
    // get_history() should make sure that
    // start->t_ > t_lastspike - dendritic_delay, i.e. minus_dt < 0
    assert( minus_dt < -1.0 * kernel().connection_manager.get_stdp_eps() );
    weight_ = facilitate_( weight_, Kplus_ * std::exp( minus_dt / tau_plus_ ) );
  }

  // depression due to new pre-synaptic spike
  weight_ =
    depress_( weight_, target->get_K_value( t_spike - dendritic_delay ) );

  e.set_receiver( *target );
  e.set_weight( weight_ );
  // use accessor functions (inherited from Connection< >) to obtain delay in
  // steps and rport
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  Kplus_ = Kplus_ * std::exp( ( t_lastspike_ - t_spike ) / tau_plus_ ) + 1.0;

  t_lastspike_ = t_spike;
}


template < typename targetidentifierT >
STDPSympyConnection< targetidentifierT >::STDPSympyConnection()
  : ConnectionBase()
  , weight_( 1.0 )
  , tau_plus_( 20.0 )
  , lambda_( 0.01 )
  , alpha_( 1.0 )
  , Wmax_( 100.0 )
  , Kplus_( 0.0 )
  , str_expr_depress_( "0." )
  , str_expr_facilitate_( "0." )
  , n_inputs_( 2 )
  , t_lastspike_( 0.0 )
{
  expr_depress_.parse( str_expr_depress_, n_inputs_ );
  expr_facilitate_.parse( str_expr_facilitate_, n_inputs_ );
}

template < typename targetidentifierT >
STDPSympyConnection< targetidentifierT >::STDPSympyConnection(
  const STDPSympyConnection< targetidentifierT >& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , tau_plus_( rhs.tau_plus_ )
  , lambda_( rhs.lambda_ )
  , alpha_( rhs.alpha_ )
  , Wmax_( rhs.Wmax_ )
  , Kplus_( rhs.Kplus_ )
  , expr_depress_( rhs.expr_depress_ )
  , expr_facilitate_( rhs.expr_facilitate_ )
  , str_expr_depress_( rhs.str_expr_depress_ )
  , str_expr_facilitate_( rhs.str_expr_facilitate_ )
  , n_inputs_( rhs.n_inputs_ )
  , t_lastspike_( rhs.t_lastspike_ )
{
}

template < typename targetidentifierT >
void
STDPSympyConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::tau_plus, tau_plus_ );
  def< double >( d, names::lambda, lambda_ );
  def< double >( d, names::alpha, alpha_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< long >( d, names::size_of, sizeof( *this ) );

  def< std::string >( d, "expr_depress", str_expr_depress_ );
  def< std::string >( d, "expr_facilitate", str_expr_facilitate_ );
  def< long >( d, "n_inputs", n_inputs_ );
}

template < typename targetidentifierT >
void
STDPSympyConnection< targetidentifierT >::set_status( const DictionaryDatum& d,
  ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::tau_plus, tau_plus_ );
  updateValue< double >( d, names::lambda, lambda_ );
  updateValue< double >( d, names::alpha, alpha_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
  if ( updateValue< std::string >( d, "expr_depress", str_expr_depress_ ) )
  {
    expr_depress_.parse( str_expr_depress_, n_inputs_ );
  }
  if ( updateValue< std::string >( d, "expr_facilitate", str_expr_facilitate_ ) )
  {
    expr_facilitate_.parse( str_expr_facilitate_, n_inputs_ );
  }

  // check if weight_ and Wmax_ has the same sign
  if ( not( ( ( weight_ >= 0 ) - ( weight_ < 0 ) )
         == ( ( Wmax_ >= 0 ) - ( Wmax_ < 0 ) ) ) )
  {
    throw BadProperty( "Weight and Wmax must have same sign." );
  }
}

} // of namespace mynest

#endif // of #ifndef STDP_SYMPY_CONNECTION_H
