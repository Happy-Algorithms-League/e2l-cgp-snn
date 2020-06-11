import nest
import numpy as np

"""
Parameters
"""

"""
NEST simulation

"""
nest.Install("stdp_sympy_synapse_module")

"""
Creation of input spikes
"""


def create_frozen_noise(num_neurons, T, rate, np_rng):
    """
    Create random spike trains for input neurons

    Parameters
    ----------
    num_neurons : int
        Number of input neurons
    T : float
        Simulation time
    rate : float
        Spike rate
    np_rng : numpy.random.RandomState
        Random number generator

    Returns
    -------
    List[np.ndarray]
        List of arrays containing the spike times for each neuron.
    """
    ISI = np_rng.exponential(
        scale=1.0 / (rate * 1e-3), size=(num_neurons, max(1, int(1.25 * rate * 1e-3 * T)))
    )
    times = np.round(np.cumsum(ISI, axis=1) + 0.1, 1)
    spike_times = [times[ii][times[ii] < T] for ii in range(num_neurons)]
    return spike_times


def build_poisson_input(inter_stim_spikes, stim_spikes, params, rng, interval="regular"):
    """
    Build the input to the task:
    cut out the intervals of pattern presentation from the
    inter_stim_spikes and insert the pattern (stim_spikes).

    Parameters
    ----------
    inter_stim_spikes : list of numpy.ndarrays
        Spike trains for time between stimulus presentations
    inter_stim_spikes : list of numpy.ndarrays
        Spike trains for stimulus presentations
    params : dict
        Experimental parameters
    interval : str
        'regular' : pattern at regular intervals
        'random': pattern appears at intervals distributed around the mean value

    Returns
    -------
    List[numpy.ndarray]
        List of spikes for each neuron.
    numpy.ndarray
        Array with all times between two pattern presentations.
    """

    # Compute reduction factor for stimulus neurons
    nu_stim = 1000.0 / (params["T_inter"] + params["T_stim"])
    nu_inter = params["bg_rate"] * params["T_inter"] / (params["T_inter"] + params["T_stim"])
    alpha = 1.0 - nu_stim / nu_inter
    assert alpha > 0

    total_spikes = []
    stim_neurons = []
    # Compute inter pattern intervals, can be random or regular
    if interval == "random":
        t = 0.1
        inter_times = []
        while t < params["T"]:
            inter_time = np.round(
                params["T_stim"] + rng.rand() * (params["T_inter"] - params["T_stim"]), 1
            )
            inter_times.append(inter_time)
            t += params["T_stim"] + inter_time
    elif interval == "regular":
        inter_times = [params["T_inter"]] * int(
            np.ceil(params["T"] / (params["T_stim"] + params["T_stim"]))
        )

    # Build spike trains
    for n in range(params["Npre"]):
        t = 0.1 + inter_times[0]
        i = 1
        sp = np.array([])
        ind = np.ones_like(inter_stim_spikes[n], dtype=np.bool)
        stim = np.array([])
        while t < params["T"]:
            inter_time = inter_times[i]
            t1 = t + params["T_stim"]
            ind = np.logical_and(
                ind,
                np.logical_not(
                    np.logical_and(inter_stim_spikes[n] > t, inter_stim_spikes[n] <= t1)
                ),
            )
            stim = np.append(stim, stim_spikes[n] + t)
            t += params["T_stim"] + inter_time
            i += 1
        sp = inter_stim_spikes[n][ind]
        if len(stim) > 0:
            sp = np.append(rng.choice(sp, size=int(sp.size * alpha)), stim)
        total_spikes.append(sp)
        if len(stim) > 0:
            stim_neurons.append(n)
    for n in range(params["Npre"]):
        total_spikes[n] = np.sort(total_spikes[n])
    return total_spikes, inter_times


"""
NEST simulation code
"""


def pattern_detection_experiment(
    synapse_model,
    synapse_params,
    params,
    input_spikes,
    return_weights=False,
    weight_array=None,
    threads=1,
    np_rng=None,
):
    """Run NEST simulation.

    Parameters
    ----------
    synapse_model: str
        Name of the synapse model.
    synapse_params : dict
        Dictionary holding the synapse parameters
    params : dict
        Dictionary holding the experiment parameters
    input_spikes : list
        List of input spikes for each input neurons.
    return_weights : bool, optional
        Whether to return a  Dataframe with the time evolution
        of all weights. Defaults to False.
    weight_array : np.ndarray, Optional
        2D numpy array specifying the weights into post-synaptic
        neuron i from presynaptic neuron j. Defaults to None, in which
        case the synaptic weight is the same for all synapses and
        taken from synapse_params.
    threads : int, Optional
        Number of threads to run NEST on. Defaults to 1.
    np_rng : int or numpy.random.RandomState
        Numpy random number generator or integer specifying the seed
        for an RNG. Defaults to None, in which case the seed takes a
        default value of 17893.

    Returns
    -------
    dict
        Dictionary of presynaptic spikes.
    dict
        Dictionary of postsynaptic spikes
    dict
        Recording of the membrane potential of the postsynaptic neuron.
    list (if return_weights is True)
        Initial synaptic weights
    list (if return_weights is True)
        Final synaptic weights
    dict (if return_weights is True)
        Time evolution of weights.
    """
    if np_rng is None:
        np_rng = np.random.RandomState(seed=17893)
    elif isinstance(np_rng, int):
        np_rng = np.random.RandomState(seed=np_rng)

    Npre = params["Npre"]
    Npost = params["Npost"]
    T = params["T"]
    base_rng = params["base_rng"]

    nest.set_verbosity("M_FATAL")
    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "grng_seed": base_rng,
            "rng_seeds": list(
                np.linspace(base_rng + 10, base_rng + 100000, threads, dtype=np.int)
            ),
            "local_num_threads": threads,
        }
    )
    nest.SetDefaults("iaf_psc_delta", params["neuron_params"])

    # Neurons
    n_post = nest.Create("iaf_psc_delta", Npost)
    nest.SetStatus(n_post, "V_m", np_rng.normal(0.0, 10.0, Npost))
    # Stimulus
    n_pre = nest.Create("spike_generator", Npre)
    nest.SetStatus(n_pre, [{"spike_times": input_spikes[ii]} for ii in range(Npre)])

    n_pre_parrots = nest.Create("parrot_neuron", Npre)
    nest.Connect(n_pre, n_pre_parrots, "one_to_one")

    synapse_model = synapse_model
    nest.SetDefaults(synapse_model, synapse_params)

    if synapse_model != "static_synapse":
        nest.CopyModel(synapse_model, synapse_model + "_rec")
        wr = nest.Create("weight_recorder")
        nest.SetDefaults(synapse_model + "_rec", {"weight_recorder": wr[0]})
    if synapse_model == "static_synapse":
        if weight_array is not None:
            for i, n in enumerate(n_post):
                [
                    nest.Connect(
                        n_pre_parrots[j],
                        n,
                        syn_spec={"synapse_model": synapse_model, "weight": weight_array[i][j]},
                    )
                    for j in range(len(n_pre_parrots))
                ]
        else:
            nest.Connect(n_pre_parrots, n_post, syn_spec={"synapse_model": synapse_model})
    else:
        if weight_array is not None:
            for i, n in enumerate(n_post):
                [
                    nest.Connect(
                        n_pre_parrots[j],
                        n,
                        syn_spec={"synapse_model": synapse_model, "weight": weight_array[i][j]},
                    )
                    for j in range(len(n_pre_parrots))
                ]
        else:
            nest.Connect(n_pre_parrots, n_post, syn_spec={"synapse_model": synapse_model + "_rec"})

    # Recording devices
    sd_pre = nest.Create("spike_detector")
    sd_post = nest.Create("spike_detector")
    nest.Connect(n_pre_parrots, sd_pre)
    nest.Connect(n_post, sd_post)

    vm_post = nest.Create("voltmeter")
    nest.SetStatus(vm_post, {"interval": 0.1})
    nest.Connect(vm_post, n_post)

    # Simulate
    weights_init, weights_final = [], []
    weights0 = nest.GetStatus(nest.GetConnections(target=n_post), "weight")
    nest.Simulate(T)
    weights1 = nest.GetStatus(nest.GetConnections(target=n_post, source=n_pre_parrots), "weight")

    weights_init.append(weights0)
    weights_final.append(weights1)

    if synapse_model != "static_synapse" and return_weights:
        weight_data = nest.GetStatus(wr, "events")[0]
    else:
        weight_data = None

    spikes_pre = nest.GetStatus(sd_pre, "events")[0]
    spikes_post = nest.GetStatus(sd_post, "events")[0]

    vm_post = nest.GetStatus(vm_post, "events")[0]

    if return_weights:
        return spikes_pre, spikes_post, vm_post, weights_init, weights_final, weight_data
    else:
        return spikes_pre, spikes_post, vm_post


def compute_SNR(weights, input_spikes, inter_times, exp_params, SNR_params, np_rng=None):
    """
    Take a list of synaptic weights and rerun simulation with a freely evolving
    neuron and static synapses to measure signal-to-noise ratio.

    Parameters
    ----------
    weights : list
        List of synaptic weights into post- from presynaptic neurons.
    input_spikes : list
        List of input spikes from presynaptic neurons
    inter_times : list
        List of lengths of time intervals in between pattern presentations
    exp_params : dict
        Dictionary of experimental parameters
    SNR_Params : dict
        Dictionary of parameters for the simulation.
    np_rng : int or numpy.random.RandomState
        Numpy random number generator or integer specifying the seed
        for an RNG. Defaults to None, in which case the seed takes a
        default value of 17893.
    """
    (spikes_pre, spikes_post, vm_post_free) = pattern_detection_experiment(
        "static_synapse",
        {},
        SNR_params,
        input_spikes,
        weight_array=np.array(weights),
        np_rng=np_rng,
    )

    cycle_lengths = np.array(inter_times) + exp_params["T_stim"]
    cycle_end_times = np.cumsum(cycle_lengths)
    stim_onset_times = cycle_end_times - 100.0

    stim = np.zeros_like(vm_post_free["V_m"])
    for i, (stim_onset, cycle_end) in enumerate(zip(stim_onset_times, cycle_end_times)):
        stim[
            np.logical_and(vm_post_free["times"] >= stim_onset, vm_post_free["times"] < cycle_end)
        ] = (i + 1)
    vm_post_free["stim"] = stim

    V_inter = vm_post_free["V_m"][vm_post_free["stim"] == 0]

    Vmax = np.array([])
    for s in range(1, int(stim.max()) + 1):
        V_stim = vm_post_free["V_m"][vm_post_free["stim"] == s]
        Vmax = np.append(Vmax, V_stim.max())
    SNR = (Vmax - V_inter.mean()) / V_inter.std()
    return np.mean(SNR), vm_post_free
