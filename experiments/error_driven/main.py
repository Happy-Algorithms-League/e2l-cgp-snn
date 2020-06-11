import json
import numpy as np
import pickle
import sys

import cgp

import nest


from sim import error, sim


class CustomConstantFloatOne(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 1.0


class CustomConstantFloatZeroPointOne(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 0.1


class CustomConstantFloatTen(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 10.0


def objective(individual):

    if individual.fitness is not None:
        return individual

    try:
        expr_str = str(individual.to_sympy()[0])
    except cgp.exceptions.InvalidSympyExpression:
        individual.fitness = -np.inf
        return individual

    with open("sim_params.json", "r") as f:
        sim_params = json.load(f)

    @cgp.utils.disk_cache("cache.pkl")
    def inner_objective(sim_params, expr_str):

        fitness = 0
        for trial in range(sim_params["n_trials_per_individual"]):
            history_u_student, history_u_teacher, *_ = sim(
                sim_params, expr_str=expr_str, trial=trial
            )
            fitness += -error(
                history_u_student["V_m"], history_u_teacher["V_m"], 0.1 * sim_params["t_sim"]
            )  # skip first 10% (initial transients)

        return fitness * 1.0 / sim_params["n_trials_per_individual"]

    try:
        individual.fitness = inner_objective(sim_params, expr_str)
    except ZeroDivisionError:
        individual.fitness = -np.inf
        return individual

    return individual


if __name__ == "__main__":

    nest.Install("us_sympy_synapse_module")

    seed_offset = int(sys.argv[1])

    with open("params.json", "r") as f:
        params = json.load(f)
    params["population_params"]["seed"] += seed_offset

    np.random.seed(params["population_params"]["seed"])

    params["genome_params"]["primitives"] = cgp.utils.primitives_from_class_names(
        params["genome_params"]["primitives"]
    )

    pop = cgp.Population(**params["population_params"], genome_params=params["genome_params"])

    ea = cgp.ea.MuPlusLambda(**params["ea_params"])

    history = {}
    history["fitness"] = np.empty(
        (params["evolve_params"]["max_generations"], params["population_params"]["n_parents"])
    )
    history["fitness_champion"] = np.empty(params["evolve_params"]["max_generations"])
    history["expr_champion"] = []

    def recording_callback(pop):
        history["fitness"][pop.generation] = pop.fitness_parents()
        history["fitness_champion"][pop.generation] = pop.champion.fitness
        history["expr_champion"].append(str(pop.champion.to_sympy()[0]))

    cgp.evolve(
        pop,
        objective,
        ea,
        **params["evolve_params"],
        print_progress=True,
        callback=recording_callback,
    )
    print(pop.champion.fitness, pop.champion.to_sympy())
    print("[final champion]", pop.champion.fitness, pop.champion.to_sympy())

    with open(f"res_{seed_offset}.pkl", "wb") as f:
        pickle.dump(history, f)
