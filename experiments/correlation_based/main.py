import copy
import functools
import json
import numpy as np
import sys
from sim_utils import pattern_detection_experiment, build_poisson_input, create_frozen_noise
from primitive_utils import (
    CustomConstantFloatPointOne,
    CustomConstantFloatOne,
    CustomConstantFloatTen,
)
import cgp


def simulate(sympy_expr, synapse_params, exp_params, input_spikes):
    """
    Perform simulation and compute Signal-to-noise ratio with the
    resulting weights from the full simulations on a freely evolving
    neuron.

    Parameters
    ----------
    sympy_expr : List[str]
        List containing a string specifying the sympy expressions for
        the depression and facilitation branch of the learning rule.
    synapse_params : dict
        Dictionary holding the synapse parameters.
    exp_params : dict
        Dictionary holding the experiment parameters.
    input_spikes : list
        List of input spikes for each input neurons.

    Returns
    -------
    float:
        Signal-to-noise ratio of the freely evolving neuron.
    """
    for expr, label in zip(sympy_expr, ["expr_depress", "expr_facilitate"]):
        if expr is not None:
            synapse_params[label] = expr

    (
        spikes_pre,
        spikes_post,
        vm_post,
        weights0,
        weights1,
        weight_data,
    ) = pattern_detection_experiment(
        "stdp_sympy_synapse", synapse_params, exp_params, input_spikes, return_weights=True
    )
    """
    Take final weights and rerun simulation with a freely evolving
    neuron to measure signal-to-noise ratio.
    """
    SNR_params = copy.deepcopy(exp_params)
    SNR_params["T"] = 11000.0
    SNR_params["neuron_params"]["V_th"] = 20000.0
    (spikes_pre, spikes_post, vm_post_free) = pattern_detection_experiment(
        "static_synapse", {}, SNR_params, input_spikes, weight_array=np.array(weights1)
    )

    vm_post_free["stim"] = 0
    t = 0.1 + exp_params["T_inter"]
    s = 1
    stim_intervals = []
    while t < exp_params["T"]:
        stim_intervals.append((t, t + exp_params["T_stim"]))
        vm_post_free.loc[
            (vm_post_free.times >= t) & (vm_post_free.times < t + exp_params["T_stim"]), "stim"
        ] = s
        t += exp_params["T_stim"] + exp_params["T_inter"]
        s += 1

    SNR = []
    V_inter = vm_post_free[vm_post_free.stim == 0]
    V_stim = vm_post_free[vm_post_free.stim > 0]

    Vmax = np.array([])
    for s in V_stim.stim.unique():
        Vmax = np.append(Vmax, V_stim[V_stim.stim == s].V_m.max())
    SNR.append((Vmax - V_inter.V_m.mean()) / V_inter.V_m.std())
    SNR = np.array(SNR)
    return np.mean(SNR)


def objective(individual, n_trials, simulate_func, exp_params, input_spikes, synapse_params):
    """Objective function of the evolution.

    Evaluates the fitness of a given individual in a number of trials
    and sets the fitness of the individual.

    Parameters
    ----------
    individual : cgp.IndividualMultiGenome
        Individual to be evaluated.
    n_trials : int
        Number of trials that the invidiual is evaluated on.
    simulate_func : Callable
        Function executing the NEST simulation.
    exp_params : dict
        Dictionary holding the experiment parameters.
    input_spikes : list
        List of input spikes for each trial. Each item is a list of
        numpy arrays defining the spikes of each input neurons.
    synapse_params : dict
        Dictionary holding the synapse parameters.

    Returns
    -------
    cgp.IndividualMultiGenome
        Individual with updated fitness.
    """

    if individual.fitness is not None:
        return individual

    graph = [cgp.CartesianGraph(genome) for genome in individual.genome]
    try:
        sympy_expr = [str(g.to_sympy(simplify=True)[0]) for g in graph]
    except cgp.InvalidSympyExpression:
        individual.fitness = -np.inf
        return individual

    @cgp.utils.disk_cache("cache.pkl")
    def inner_objective(exp_params, synapse_params, input_spikes, sympy_expr):
        SNR = []
        for i in range(n_trials):
            SNR.append(
                simulate_func(
                    sympy_expr,
                    exp_params=exp_params,
                    input_spikes=input_spikes[i],
                    synapse_params=synapse_params[i],
                )
            )
        fitness = np.min(SNR)
        return fitness

    # return the updated individual
    individual.fitness = inner_objective(exp_params, synapse_params, input_spikes, sympy_expr)
    return individual


if __name__ == "__main__":
    # For parallel execution of the evolution
    run_nr = int(sys.argv[-1])

    # Load parameters
    with open("params.json", "r") as f:
        params = json.load(f)

    # Update seeds
    params["exp_params"]["base_rng"]  # += run_nr
    params["gp_params"]["seed"] += run_nr

    # Determine initial weight that leads to approx. 20 Hz firing rate in the postsynaptic neuron
    weight = (
        params["exp_params"]["neuron_params"]["V_th"]
        / (
            params["exp_params"]["Npre"] * params["exp_params"]["bg_rate"]
            - np.sqrt(params["exp_params"]["Npre"] * params["exp_params"]["bg_rate"]) * 2
        )
        / (params["exp_params"]["neuron_params"]["tau_m"] * 1e-3)
    )
    synapse_params = []
    rng = np.random.RandomState(seed=params["exp_params"]["base_rng"] + 111)
    for i in range(params["gp_params"]["n_trials"]):
        d = copy.deepcopy(params["synapse_params"])
        d["weight"] = rng.normal(loc=weight, scale=0.1)
        synapse_params.append(d)

    """
    Create frozen noise for stimulation
    """
    print("Create frozen noise for experiments.")
    np_rng = np.random.RandomState(seed=params["exp_params"]["base_rng"] + 123)
    input_spikes = []
    for i in range(params["gp_params"]["n_trials"]):
        spike_pattern = create_frozen_noise(
            params["exp_params"]["Npre"],
            params["exp_params"]["T_stim"],
            params["exp_params"]["bg_rate"],
            np_rng,
        )
        inter_spike_pattern = create_frozen_noise(
            params["exp_params"]["Npre"],
            params["exp_params"]["T"],
            params["exp_params"]["bg_rate"],
            np_rng,
        )
        input_spikes.append(
            build_poisson_input(inter_spike_pattern, spike_pattern, params["exp_params"], np_rng)[
                0
            ]
        )
    # Wrap objective with frozen noise
    objective_wrapped = functools.partial(
        objective,
        n_trials=params["gp_params"]["n_trials"],
        simulate_func=simulate,
        exp_params=params["exp_params"],
        input_spikes=input_spikes,
        synapse_params=synapse_params,
    )

    np.random.seed(params["gp_params"]["seed"])

    # Prepare evolution.
    primitives = (
        cgp.Add,
        cgp.Sub,
        cgp.Mul,
        cgp.Div,
        cgp.Pow,
        CustomConstantFloatPointOne,
        CustomConstantFloatOne,
        CustomConstantFloatTen,
    )
    params["gp_params"]["genome_params"].update({"primitives": primitives})

    # create population object that will be evolved
    genome_params = [params["gp_params"]["genome_params"] for i in range(2)]
    pop = cgp.Population(
        **params["gp_params"]["population_params"],
        seed=params["gp_params"]["seed"],
        genome_params=genome_params,
    )

    # Initialize evolutionary algorithm
    ea = cgp.ea.MuPlusLambda(**params["gp_params"]["ea_params"])

    # Bookkeeping
    history = {"fitness": [], "fitness_champion": [], "expr_champion": []}

    def record_history(pop, run_nr):
        history["fitness"].append(pop.fitness_parents())
        history["fitness_champion"].append(pop.champion.fitness)
        history["expr_champion"].append(str(pop.champion.to_sympy(simplify=True)))
        if len(history["fitness"]) % 2 == 0:
            np.save("fitness_{}.npy".format(run_nr), history["fitness"])

    record_history_wrapped = functools.partial(record_history, run_nr=run_nr)

    print("Starting evolution.")
    cgp.evolve(
        pop,
        objective_wrapped,
        ea,
        params["gp_params"]["max_generations"],
        params["gp_params"]["min_fitness"],
        callback=record_history_wrapped,
        print_progress=True,
    )

    print("Evolution finished. Storing results.")
    # Evolution of fitness values over time
    np.save("fitness_{}.npy".format(run_nr), np.array(history["fitness"]))
    # Store champion of this evolutionary run
    str_tuple = [str(len(history["fitness_champion"])), str(pop.champion.fitness)]
    for i in range(2):
        str_tuple.append(str(pop.champion.to_sympy()[i][0]))
    s = ",".join(str_tuple)
    print(s)
    with open("champion_{}.txt".format(run_nr), "w") as f:
        f.write(s)
    # Evolution of champion expression over time
    with open("champion_evol_{}.txt".format(run_nr), "w") as f:
        for s in history["expr_champion"]:
            f.write(s)
            f.write("\n")
