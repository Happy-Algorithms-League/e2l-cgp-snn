/*
 *  stdp_homeostatic_connection.h
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

#ifndef STDP_HOMEOSTATIC_CONNECTION_H
#define STDP_HOMEOSTATIC_CONNECTION_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"
#include "histentry.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

namespace mynest
{

/** @BeginDocumentation
Name: stdp_homeostatic_synapse - Synapse type for spike-timing dependent
   plasticity with homeostasis term and no depression [1].
   Based on the stdp_synapse of NEST.

  Description:
   stdp_homeostatic_synapse is a connector to create synapses with spike time
   dependent plasticity (as defined in [1]). Here the weight dependence
   exponent can be set for potentiation.

  Parameters:
   tau_plus   double - Time constant of STDP window, potentiation in ms
                       (tau_minus defined in post-synaptic neuron)
   lambda     double - Step size
   mu_plus    double - Weight dependence exponent, potentiation
   Wmax       double - Maximum allowed weight
   Wout       double - Fixed homeostasis term added to the weight at every postsynaptic spike.

  Transmits: SpikeEvent

  References: [1] Masquelier, T. (2017). STDP allows close-to-optimal
   spatiotemporal spike pattern detection by single coincidence
   detector neurons. Neuroscience, 1–8.
   https://doi.org/10.1016/j.neuroscience.2017.06.032 

*/
// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template
template < typename targetidentifierT >
class STDPHomeostaticConnection : public nest::Connection< targetidentifierT >
{
private:
  double
  facilitate_( double w, double kplus )
  {
    const double norm_w = ( w / Wmax_ )
      + ( lambda_ * std::pow( 1.0 - ( w / Wmax_ ), mu_plus_ ) * kplus );
    const double w_new = norm_w * Wmax_ + wout_;
    if ( w_new > Wmax_ ) // too large
    {
      return Wmax_;
    }
    else if ( w_new < 0.0) // too small
    {
      return 0.0;
    }
    else
    {
      return w_new;
    }
  }


  // data members of each connection
  double weight_;
  double tau_plus_;
  double lambda_;
  double mu_plus_;
  double Wmax_;
  double Kplus_;

  double t_lastspike_;
  double wout_;
  
public:
  //! Type to use for representing common synapse properties
  using CommonPropertiesType = nest::CommonSynapseProperties;

  //! Shortcut for base class
  using ConnectionBase = nest::Connection< targetidentifierT >;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  STDPHomeostaticConnection();
  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  STDPHomeostaticConnection( const STDPHomeostaticConnection& );

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
  void set_status( const DictionaryDatum& d, nest::ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( nest::Event& e, nest::thread t, const CommonPropertiesType& cp );


  class ConnTestDummyNode : public nest::ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using nest::ConnTestDummyNodeBase::handles_test_event;
    nest::port
    handles_test_event( nest::SpikeEvent&, nest::rport )
    {
      return nest::invalid_port_;
    }
  };

    /**
   * Check that requested connection can be created.
   *
   * This function is a boilerplate function that should be included unchanged
   * in all synapse models. It is called before a connection is added to check
   * that the connection is legal. It is a wrapper that allows us to call
   * the "real" `check_connection_()` method with the `ConnTestDummyNode
   * dummy_target;` class for this connection type. This avoids a virtual
   * function call for better performance.
   *
   * @param s  Source node for connection
   * @param t  Target node for connection
   * @param receptor_type  Receptor type for connection
   */
  void
  check_connection( nest::Node& s, nest::Node& t, nest::rport receptor_type,
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

};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
STDPHomeostaticConnection< targetidentifierT >::send( nest::Event& e,
						      nest::thread t,
						      const CommonPropertiesType& )
{
  // synapse STDP depressing/facilitation dynamics
  double t_spike = e.get_stamp().get_ms();

  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  nest::Node* target = get_target( t );
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t1, t2] from post-synaptic neuron
  std::deque< nest::histentry >::iterator start;
  std::deque< nest::histentry >::iterator finish;

  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
  // details.
  target->get_history( t_lastspike_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish );

  // facilitation due to post-synaptic spikes since last pre-synaptic spike
  double minus_dt;
  while ( start != finish )
  {
    minus_dt = t_lastspike_ - ( start->t_ + dendritic_delay );
    ++start;
    // get_history() should make sure that
    // start->t_ > t_lastspike - dendritic_delay, i.e. minus_dt < 0
    assert( minus_dt < -1.0 * nest::kernel().connection_manager.get_stdp_eps() );
    weight_ = facilitate_( weight_, Kplus_ * std::exp( minus_dt / tau_plus_ ) );
  }

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
STDPHomeostaticConnection< targetidentifierT >::STDPHomeostaticConnection()
  : ConnectionBase()
  , weight_( 1.0 )
  , tau_plus_( 20.0 )
  , lambda_( 0.01 )
  , mu_plus_( 1.0 )
  , Wmax_( 100.0 )
  , Kplus_( 0.0 )
  , t_lastspike_( 0.0 )
  , wout_ (0.0)
{
}

template < typename targetidentifierT >
STDPHomeostaticConnection< targetidentifierT >::STDPHomeostaticConnection(
  const STDPHomeostaticConnection< targetidentifierT >& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , tau_plus_( rhs.tau_plus_ )
  , lambda_( rhs.lambda_ )
  , mu_plus_( rhs.mu_plus_ )
  , Wmax_( rhs.Wmax_ )
  , Kplus_( rhs.Kplus_ )
  , t_lastspike_( rhs.t_lastspike_ )
  , wout_ ( rhs.wout_)
{
}

template < typename targetidentifierT >
void
STDPHomeostaticConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, nest::names::weight, weight_ );
  def< double >( d, nest::names::tau_plus, tau_plus_ );
  def< double >( d, nest::names::lambda, lambda_ );
  def< double >( d, nest::names::mu_plus, mu_plus_ );
  def< double >( d, nest::names::Wmax, Wmax_ );
  def< double >( d, "Wout", wout_ );
  def< long >( d, nest::names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
STDPHomeostaticConnection< targetidentifierT >::set_status( const DictionaryDatum& d, nest::ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, nest::names::weight, weight_ );
  updateValue< double >( d, nest::names::tau_plus, tau_plus_ );
  updateValue< double >( d, nest::names::lambda, lambda_ );
  updateValue< double >( d, nest::names::mu_plus, mu_plus_ );
  updateValue< double >( d, nest::names::Wmax, Wmax_ );
  updateValue< double >( d, "Wout", wout_ );

  // check if weight_ and Wmax_ has the same sign
  if ( not( ( ( weight_ >= 0 ) - ( weight_ < 0 ) )
         == ( ( Wmax_ >= 0 ) - ( Wmax_ < 0 ) ) ) )
  {
    throw nest::BadProperty( "Weight and Wmax must have same sign." );
  }
}

} // of namespace nest

#endif // of #ifndef STDP_HOMEOSTATIC_CONNECTION_H
