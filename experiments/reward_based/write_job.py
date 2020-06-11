import dicthash
import json
import numpy as np
import os
import sys

sys.path.insert(0, "../includes/")
import write_job_utils as utils  # noqa: E402


if __name__ == "__main__":

    sim_params = {
        "seed": 11231,
        "n_threads": 1,
        "n_inputs": 50,  # 10,
        "n_stimuli": 30,  # 5,
        "input_rate": 6.0,  # (Hz)
        "input_connectivity": 0.8,
        "stimulus_duration": 500,  # (ms)
        "population_size": 1,
        "dt": 0.01,  # (ms)
        "n_training_trials": 500,
        "n_test_trials": 20,
        "n_trials_per_individual": 5,
        "learning_rate": 10.0,
        "initial_weight": 1e3,  # (pA)
        "rho": 0.01,  # (Hz)
        "delta": 0.2,  # (mV)
        "E_L": -70.0,  # (mV)
        "record_stimuli_to_file": False,
        "record_connections_to_file": False,
        "record_input_to_file": False,
        "record_output_to_file": False,
        "file_name_input": "input-{generation}-{label}.npy",
        "file_name_output": "output-{generation}-{label}.npy",
        "file_name_stimuli": "stimuli-{generation}.npy",
        "file_name_template": "connections-{generation}-{label}.npy",
    }

    sim_key = dicthash.generate_hash_from_dict(sim_params)

    params = {
        "sim_key": sim_key,  # consistency check
        # machine setup
        "submit_command": "sbatch",
        "jobfile_template": "slurm_template.jdf",
        "jobname": <jobname>,
        "wall_clock_limit": "12:00:00",
        "n_processes": 8,
        "n_threads": 8,
        "n_nodes": 1,
        "mail": <mail>,
        "account": <account>,
        "partition": <partition>,
        "sim_script": "main.py",
        "dependencies": ["sim.py"],
        "population_params": {"n_parents": 4, "mutation_rate": 0.045, "seed": 12,},
        "genome_params": {
            "n_inputs": 2,
            "n_outputs": 1,
            "n_columns": 5,
            "n_rows": 1,
            "levels_back": None,
            "primitives": (
                "Add",
                "Sub",
                "Mul",
                "Div",
                "CustomConstantFloatOne",
                "CustomConstantFloatZeroPointOne",
                "CustomConstantFloatTen",
            ),
        },
        "ea_params": {
            "n_offsprings": 4,
            "n_breeding": 4,
            "tournament_size": 1,
            "n_processes": 8,
        },
        "evolve_params": {"max_generations": 500, "min_fitness": 300,},
    }
    params["md5_hash_sim_script"] = utils.md5_file(
        params["sim_script"]
    )  # consistency check

    key = dicthash.generate_hash_from_dict(params, blacklist=["wall_clock_limit"])

    params["outputdir"] = os.path.join(os.getcwd(), key)
    params["workingdir"] = os.getcwd()

    submit_job = False

    print("preparing job")
    print(" ", params["outputdir"])

    utils.mkdirp(params["outputdir"])
    utils.write_json(sim_params, os.path.join(params["outputdir"], "sim_params.json"))
    utils.write_json(params, os.path.join(params["outputdir"], "params.json"))
    utils.create_jobfile(params)
    utils.copy_file(params["sim_script"], params["outputdir"])
    utils.copy_files(params["dependencies"], params["outputdir"])
    if submit_job:
        print("submitting job")
        utils.submit_job(params)
