import json
import numpy as np
import os
import pickle
import sys

import nest
from sim import train


sys.path.insert(0, "../includes/")
import utils  # noqa: E402


def plasticity_rule(learning_rate, R, E):
    return learning_rate * (R - 1.0) * E


if __name__ == "__main__":

    nest.Install("usrl_synapse_module")

    data_directory = utils.get_data_directory_from_cmd_args()

    with open(os.path.join(data_directory, "sim_params.json"), "r") as f:
        sim_params = json.load(f)

    sim_params["generation"] = 0  # dummy

    sim_params["record_stimuli_to_file"] = False
    sim_params["record_connections_to_file"] = False
    sim_params["record_input_to_file"] = True
    sim_params["record_output_to_file"] = True
    sim_params["file_name_stimuli"] = None
    sim_params["file_name_template"] = None
    sim_params["file_name_input"] = os.path.join(data_directory, sim_params["file_name_input"])
    sim_params["file_name_output"] = os.path.join(data_directory, sim_params["file_name_output"])

    # use a different seed than evo runs to generate different spike
    # train realization and class labels
    sim_params["seed"] = 817818273

    cumulative_reward = np.empty(sim_params["n_trials_per_individual"])
    for trial in range(sim_params["n_trials_per_individual"]):
        print(f'trial {trial + 1}/{sim_params["n_trials_per_individual"]}')
        history_R, _, _, _, _ = train(sim_params, plasticity_rule, trial)

        cumulative_reward[trial] = sum(history_R)

    with open(os.path.join(data_directory, "res_test.pkl"), "wb") as f:
        pickle.dump(cumulative_reward, f)
