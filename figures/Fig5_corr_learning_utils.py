import nest
import numpy as np
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr


def deltaW(synapse_type, synapse_params, delta_t_range, weights, delay):
    """Simulate weight change of a given synapse type for a range of
    weights and presynaptic spike times.

    Parameters
    ----------
    synapse_type : str
        Type of synapse, must specify a valid NEST synapse model.
    synapse_params : dict
        Parameter of the synapse. Note: receptor_type has to be set to 1.
    delta_t_range : numpy.ndarray or list
        List or array of presynaptic spike times
    weights : numpy.ndarray or list
        List or array of (initial) synaptic weights
    delay : float
        Delay of the synapse

    Returns
    -------
    tuple of lists
        Tuple with first item = delta_t_range and second item = change
        of synaptic weight induced by plasiticty
    """
    post_spike = 250.0
    deltaw = np.zeros((weights.size, delta_t_range.size))
    for i, weight in enumerate(weights):
        for j, delta_t in enumerate(delta_t_range):
            nest.ResetKernel()
            nest.SetKernelStatus({"total_num_virtual_procs": 1, "resolution": 0.1})
            nest.set_verbosity("M_FATAL")
            synapse_params["weight"] = weight
            nest.SetDefaults(synapse_type, synapse_params)
            spike_generator_pre = nest.Create("spike_generator")
            spike_generator_post = nest.Create("spike_generator")

            parrot_pre = nest.Create("parrot_neuron")
            parrot_post = nest.Create(
                "parrot_neuron", params={"tau_minus": synapse_params["tau_plus"]}
            )

            nest.Connect(spike_generator_pre, parrot_pre)
            nest.Connect(spike_generator_post, parrot_post)
            nest.Connect(parrot_pre, parrot_post, syn_spec={"synapse_model": synapse_type})
            pre_spike_times = [post_spike - delta_t, post_spike + 1000.0]
            post_spike_times = [post_spike]
            nest.SetStatus(spike_generator_pre, {"spike_times": pre_spike_times})
            nest.SetStatus(spike_generator_post, {"spike_times": post_spike_times})

            nest.Simulate(pre_spike_times[-1] + 10.0)

            new_weight = nest.GetStatus(nest.GetConnections(parrot_pre, parrot_post), "weight")[0]
            deltaw[i][j] = new_weight - synapse_params["weight"]

    return delta_t_range, deltaw


def analytical_learning_rule(dt, synapse_params):
    """Determine change of synaptic weight of the stdp_symyp_synapse.
    for a given different between pre- and post-synaptic spike.

    Parameters
    ----------
    dt : float
        Time different between pre- and post-synaptic weight.
    synapse_params : dict
        Parameters of stdp_sympy_synapse.

    Returns
    -------
    tuple of floats
        First item: weight change
        Second item: new weight
    """
    d = {}
    for key in ["alpha", "lambda", "Wmax", "weight", "delay"]:
        if key in synapse_params:
            d[key] = synapse_params[key]
        else:
            d[key] = nest.GetDefaults("stdp_sympy_synapse")[key]
    dt_prime = dt + d["delay"]
    x_0, x_1 = Symbol("x_0"), Symbol("x_1")
    if dt_prime < 0:
        stdp_trace = np.exp(-np.abs(dt_prime) / synapse_params["tau_plus"])
        expr_depress = parse_expr(synapse_params["expr_depress"])
        subs = []
        if "x_0" in synapse_params["expr_depress"]:
            subs.append([x_0, d["weight"] / d["Wmax"]])
        if "x_1" in synapse_params["expr_depress"]:
            subs.append((x_1, stdp_trace))
        delta_w = expr_depress.subs(subs)
        norm_w = (d["weight"] / d["Wmax"]) - d["alpha"] * d["lambda"] * delta_w
        if norm_w > 0.0:
            w_prime = norm_w * d["Wmax"]
        else:
            w_prime = 0.0
        delta_w = w_prime - d["weight"]

    else:
        stdp_trace = np.exp(-np.abs(dt_prime) / synapse_params["tau_plus"])
        subs = []
        if "x_0" in synapse_params["expr_facilitate"]:
            subs.append([x_0, d["weight"] / d["Wmax"]])
        if "x_1" in synapse_params["expr_facilitate"]:
            subs.append((x_1, stdp_trace))
        expr_facilitate = parse_expr(synapse_params["expr_facilitate"])
        delta_w = expr_facilitate.subs(subs)
        norm_w = (d["weight"] / d["Wmax"]) + d["lambda"] * delta_w
        w_prime = norm_w * d["Wmax"]
        if w_prime < 1.0:
            delta_w = w_prime - d["weight"]
        else:
            delta_w = d["Wmax"]
    return np.real(complex(delta_w)), w_prime
