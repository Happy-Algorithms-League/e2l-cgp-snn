import copy
import json
import nest
import numpy as np
import os

from sim_utils import (
    compute_SNR,
    build_poisson_input,
    create_frozen_noise,
    pattern_detection_experiment,
)

import sys

sys.path.insert(0, "../includes/")
import utils  # noqa: E402

nest.Install("HomeostaticSTDPmodule")

compute = ["B", "C", "D"]
# compute = ['D']

data_directory = utils.get_data_directory_from_cmd_args()

with open(os.path.join(data_directory, "params.json"), "r") as f:
    params = json.load(f)
    synapse_params = params["synapse_params"]
    exp_params = params["exp_params"]

"""
Create data used in panels B, C, and D of Fig.5

B: Input spikes visualized in raster plot
"""
if "B" in compute:
    weight = 0.5
    exp_params_B = copy.deepcopy(exp_params)
    exp_params_B["T"] = 1100.0

    # Create input spikes for visualization
    seed = 123
    np_rng = np.random.RandomState(seed=exp_params_B["base_rng"] + 123)
    spike_pattern = create_frozen_noise(
        exp_params_B["Npre"], exp_params_B["T_stim"], exp_params_B["bg_rate"], np_rng
    )

    inter_spike_pattern = create_frozen_noise(
        exp_params_B["Npre"], exp_params_B["T"], exp_params_B["bg_rate"], np_rng
    )
    rng = np.random.RandomState(seed=123123)
    input_spikes, inter_times = build_poisson_input(
        inter_spike_pattern, spike_pattern, exp_params_B, rng
    )

    input_spikes = np.array(input_spikes)
    with open(os.path.join(data_directory, "Fig5B_params.json"), "w") as f:
        json.dump(exp_params_B, f)
    np.save(os.path.join(data_directory, "Fig5B_spikes.npy"), input_spikes)


"""
C: Membrane potential of post-synaptic neuron in example simulation
"""
if "C" in compute:
    exp_params_C = copy.deepcopy(exp_params)
    exp_params_C["T"] = 200000.0

    synapse_model = "stdp_homeostatic_synapse"
    synapse_params_homeo = {
        "lambda": 0.01,
        "mu_plus": 0.0,
        "tau_plus": 20.0,
        "Wout": -0.0015,
        "weight": 0.5,
        "Wmax": 1.0,
        "receptor_type": 0,
    }

    # Create frozen noise for simulation
    np_rng = np.random.RandomState(seed=exp_params_C["base_rng"] + 123)
    spike_pattern = create_frozen_noise(
        exp_params_C["Npre"], exp_params_C["T_stim"], exp_params_C["bg_rate"], np_rng
    )

    inter_spike_pattern = create_frozen_noise(
        exp_params_C["Npre"], exp_params_C["T"], exp_params_C["bg_rate"], np_rng
    )
    rng = np.random.RandomState(seed=123123)
    input_spikes, inter_times = build_poisson_input(
        inter_spike_pattern, spike_pattern, exp_params_C, rng
    )

    # Run simulation
    (
        spikes_pre,
        spikes_post,
        vm_post,
        weights0,
        weights1,
        weight_data,
    ) = pattern_detection_experiment(
        synapse_model, synapse_params_homeo, exp_params_C, input_spikes, return_weights=1000.0
    )
    # Store data
    with open(os.path.join(data_directory, "Fig5C_params.json"), "w") as f:
        json.dump(exp_params_C, f)

    np.save(os.path.join(data_directory, "Fig5C_vm_post.npy"), np.array(vm_post["V_m"]))


"""
D: Fitness value of stdp homeostasis synaptic rule (orange line) and
validation fitness of two selected learning rules (blue and red dots
with errorbars)
"""
if "D" in compute:
    print("Validation run, homeostatic synapse.")
    exp_params_D = {
        "base_rng": 123,
        "Npre": 1000,
        "Npost": 1,
        "T": 200000.0,
        "T_stim": 100.0,
        "T_inter": 400.0,
        "tbin": 1.0,
        "bg_rate": 3.0,
        "neuron_params": {"V_reset": 0.0, "E_L": 0.0, "V_th": 20.0, "tau_m": 18.0},
    }

    base_rng = 11214124
    n_trials = 20
    np_rng = np.random.RandomState(seed=base_rng + 123)

    SNR_params = copy.deepcopy(exp_params_D)
    SNR_params["T"] = 11000.0
    SNR_params["neuron_params"]["V_th"] = 20000.0

    synapse_model = "stdp_homeostatic_synapse"
    synapse_params_homeo = {
        "lambda": 0.01,
        "mu_plus": 0.0,
        "tau_plus": 20.0,
        "Wout": -0.0015,
        "weight": 0.5,
        "Wmax": 1.0,
        "receptor_type": 0,
    }

    SNR_list = []
    for i in range(n_trials):
        print(f"\t trial {i}")
        spike_pattern = create_frozen_noise(
            exp_params_D["Npre"], exp_params_D["T_stim"], exp_params_D["bg_rate"], np_rng
        )
        inter_spike_pattern = create_frozen_noise(
            exp_params_D["Npre"], exp_params_D["T"], exp_params_D["bg_rate"], np_rng
        )
        input_spikes, inter_times = build_poisson_input(
            inter_spike_pattern, spike_pattern, exp_params_D, np_rng
        )
        # Perform simulation
        (
            spikes_pre,
            spikes_post,
            vm_post,
            weights0,
            weights1,
            weight_data,
        ) = pattern_detection_experiment(
            synapse_model, synapse_params_homeo, exp_params_D, input_spikes, return_weights=True
        )
        # Compute SNR
        SNR, vm_post_free = compute_SNR(
            weights1, input_spikes, inter_times, exp_params_D, SNR_params, np_rng=i
        )
        SNR_list.append(SNR)

    np.save(os.path.join(data_directory, "Fig5D_SNR_homeostatic_synapse.npy"), np.array(SNR_list))

if "D" in compute:
    """
    Perform simulations with selected learning rules on test data samples
    """
    selected_runs = [9, 18]
    num_validation_trials = 20
    synapse_params["weight"] = 0.5
    synapse_model = "stdp_sympy_synapse"

    for run in selected_runs:
        print(f"Validating run {run}.")
        fn = os.path.join(data_directory, "champion_{}.txt".format(run))
        with open(fn, "r") as f:
            li = f.readlines()
            facilitate_expr = li[0].split(",")[-1]
            depress_expr = li[0].split(",")[-2]
        synapse_params.update({"expr_depress": depress_expr, "expr_facilitate": facilitate_expr})

        # Create frozen noise for stimulation
        np_rng_validation = np.random.RandomState(seed=exp_params["base_rng"] + 1999)

        fitness_validation_list = []
        # Run simulation for num_validation_trials
        for i in range(num_validation_trials):
            print(f"\t trial {i}")
            spike_pattern = create_frozen_noise(
                exp_params["Npre"], exp_params["T_stim"], exp_params["bg_rate"], np_rng_validation
            )

            inter_spike_pattern = create_frozen_noise(
                exp_params["Npre"], exp_params["T"], exp_params["bg_rate"], np_rng_validation
            )
            rng = np.random.RandomState(seed=123123)
            input_spikes, inter_times = build_poisson_input(
                inter_spike_pattern, spike_pattern, exp_params, np_rng_validation
            )
            """
            Perform simulation
            """
            (
                spikes_pre,
                spikes_post,
                vm_post,
                weights0,
                weights1,
                weight_data,
            ) = pattern_detection_experiment(
                synapse_model, synapse_params, exp_params, input_spikes, return_weights=True
            )
            SNR_params = copy.deepcopy(exp_params)
            SNR_params["T"] = 31000.0
            SNR_params["neuron_params"]["V_th"] = 20000.0

            SNR, vm_post_free = compute_SNR(
                weights1, input_spikes, inter_times, exp_params, SNR_params, np_rng=i
            )
            fitness_validation_list.append(SNR)

        np.save(
            os.path.join(data_directory, f"Fig5D_fitness_validation_run_{run}.npy"),
            np.array(fitness_validation_list),
        )
