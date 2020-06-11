/*
 *  us_sympy_synapse_module.cpp
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

#include "us_sympy_synapse_module.h"

// Generated includes:
#include "config.h"

// include headers with your own stuff
#include "us_sympy_connection.h"

// Includes from nestkernel:
#include "archiving_node.h"
#include "connection_manager_impl.h"
#include "connector_model_impl.h"
#include "dynamicloader.h"
#include "exceptions.h"
#include "genericmodel.h"
#include "genericmodel_impl.h"
#include "io_manager_impl.h"
#include "kernel_manager.h"
#include "model.h"
#include "model_manager_impl.h"
#include "nest.h"
#include "nest_impl.h"
#include "nestmodule.h"
#include "target_identifier.h"
#include "archiving_node.h"
#include "connection.h"

// Includes from libnestutil:
#include "numerics.h"
#include "propagator_stability.h"

// Includes from nestkernel:
#include "ring_buffer.h"

// Includes from sli:
#include "booldatum.h"
#include "integerdatum.h"
#include "sliexceptions.h"
#include "tokenarray.h"

// -- Interface to dynamic module loader ---------------------------------------

/*
 * There are three scenarios, in which USSymPySynapseModule can be loaded by NEST:
 *
 * 1) When loading your module with `Install`, the dynamic module loader must
 * be able to find your module. You make the module known to the loader by
 * defining an instance of your module class in global scope. (LTX_MODULE is
 * defined) This instance must have the name
 *
 * <modulename>_LTX_mod
 *
 * The dynamicloader can then load modulename and search for symbol "mod" in it.
 *
 * 2) When you link the library dynamically with NEST during compilation, a new
 * object has to be created. In the constructor the DynamicLoaderModule will
 * register your module. (LINKED_MODULE is defined)
 *
 * 3) When you link the library statically with NEST during compilation, the
 * registration will take place in the file `static_modules.h`, which is
 * generated by cmake.
 */
#if defined( LTX_MODULE ) | defined( LINKED_MODULE )
mynest::USSymPySynapseModule us_sympy_synapse_module_LTX_mod;
#endif
// -- DynModule functions ------------------------------------------------------

mynest::USSymPySynapseModule::USSymPySynapseModule()
{
#ifdef LINKED_MODULE
  // register this module at the dynamic loader
  // this is needed to allow for linking in this module at compile time
  // all registered modules will be initialized by the main app's dynamic loader
  nest::DynamicLoaderModule::registerLinkedModule( this );
#endif
}

mynest::USSymPySynapseModule::~USSymPySynapseModule() = default;

const std::string
mynest::USSymPySynapseModule::name() const
{
  return std::string( "USSymPy Synapse Module" ); // Return name of the module
}

const std::string
mynest::USSymPySynapseModule::commandstring() const
{
  // Instruct the interpreter to load us-sympy-synapse-module-init.sli
  return std::string( "(us-sympy-synapse-module-init) run" );
}

//-------------------------------------------------------------------------------------

void
mynest::USSymPySynapseModule::init( SLIInterpreter* i )
{
  /* Register a synapse type.
   */
  nest::register_secondary_connection_model< USSymPyConnection >(
    "us_sympy_synapse", RegisterConnectionModelFlags::HAS_DELAY );

} // USSymPySynapseModule::init()
