import json
import numpy as np
import sys
import os
import pickle

import nest

from sim import error, sim

sys.path.insert(0, "../includes/")
import utils  # noqa: E402


if __name__ == "__main__":

    nest.Install("us_sympy_synapse_module")

    data_directory = utils.get_data_directory_from_cmd_args()

    with open(f"{data_directory}/sim_params.json", "r") as f:
        sim_params = json.load(f)

    # use a different seed than evo runs to generate different spike
    # train realization and class labels
    sim_params["seed"] = 817818273

    n_runs = 6
    for i in range(n_runs):
        with open(os.path.join(data_directory, f"res_{i}.pkl"), "rb") as f:
            res = pickle.load(f)

        print("sim for", res["expr_champion"][-1])

        errors = np.empty(sim_params["n_trials_per_individual"])
        for trial in range(sim_params["n_trials_per_individual"]):
            print(f'  trial {trial + 1}/{sim_params["n_trials_per_individual"]}')
            data_u, data_u_target, *_ = sim(
                sim_params, expr_str=res["expr_champion"][-1], trial=trial
            )

            errors[trial] = -error(
                data_u["V_m"], data_u_target["V_m"], 0.1 * sim_params["t_sim"]
            )  # skip first 10% (initial transients)

        with open(os.path.join(data_directory, f"res_test_{i}.pkl"), "wb") as f:
            pickle.dump(errors, f)
