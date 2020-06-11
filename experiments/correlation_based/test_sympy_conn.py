import os
import sys

import nest

nest.ResetKernel()
nest.Install("stdp_sympy_synapse_module")

n1 = nest.Create("parrot_neuron")
n2 = nest.Create("iaf_psc_exp")

nest.Connect(n1, n2, syn_spec={"synapse_model": "stdp_sympy_synapse"})

nest.Simulate(1)
