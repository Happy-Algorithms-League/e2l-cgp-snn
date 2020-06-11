import json
import os
import pickle
import sys

import nest

from sim import sim

sys.path.insert(0, "../includes/")
import utils  # noqa: E402


if __name__ == "__main__":

    nest.Install("us_sympy_synapse_module")

    data_directory = utils.get_data_directory_from_cmd_args()

    with open(f"{data_directory}/sim_params.json", "r") as f:
        sim_params = json.load(f)

    sim_params["t_sim"] *= 2.0

    (history_u_student, history_u_teacher, history_weights_student, weights_teacher) = sim(
        sim_params, expr_str="(x_0 - x_1) * x_2", trial=0
    )

    res = {
        "data_u": history_u_student,
        "data_u_teacher": history_u_teacher,
        "data_syn": history_weights_student,
        "data_syn_teacher": weights_teacher,
    }
    with open(os.path.join(data_directory, "traces.pkl"), "wb") as f:
        pickle.dump(res, f)
