import dicthash
import json
import numpy as np
import os
import sys

sys.path.insert(0, "../includes/")
import write_job_utils as utils  # noqa: E402


if __name__ == "__main__":

    sim_params = {
        "seed": 1235,
        "n_inputs": 5,
        "n_outputs": 1,
        "initial_weight": 5.0,
        "range_input_rates": [150.0, 850.0],
        "dt": 0.01,
        "t_sim_step": 5.0,
        "t_sim": 10000.0,
        "do_shift_weights": True,
        # 'do_shift_weights': False,
        "range_teacher_weights": [-20.0, 20.0],
        # 'range_teacher_weights': [0.0, 40.0],
        # 'range_teacher_weights': [-40.0, 0.0],
        "lr": 1.7,
        "rho": 0.2,
        "delta": 1.0,
        "n_trials_per_individual": 15,
    }

    sim_key = dicthash.generate_hash_from_dict(sim_params)

    params = {
        "label": "testest",
        "sim_key": sim_key,  # consistency check
        # machine setup
        "submit_command": "sbatch",
        "jobfile_template": "slurm_template.jdf",
        "jobname": <jobname>,
        "wall_clock_limit": "12:00:00",
        "n_processes": 6,
        "n_threads": 8,
        "n_nodes": 1,
        "mail": <mail>,
        "account": <account>,
        "partition": <partition>,
        "sim_script": "main.py",
        "dependencies": ["sim.py"],
        "population_params": {"n_parents": 4, "mutation_rate": 0.045, "seed": 12,},
        "genome_params": {
            "n_inputs": 3,
            "n_outputs": 1,
            "n_columns": 12,
            "n_rows": 1,
            "levels_back": None,
            "primitives": ("Add", "Sub", "Mul", "Div", "CustomConstantFloatOne"),
        },
        "ea_params": {
            "n_offsprings": 4,
            "n_breeding": 4,
            "tournament_size": 1,
            "n_processes": 8,
        },
        "evolve_params": {"max_generations": 1000, "min_fitness": 0.0,},
    }
    params["md5_hash_sim_script"] = utils.md5_file(
        params["sim_script"]
    )  # consistency check
    params["md5_hash_dependencies"] = [
        utils.md5_file(fn) for fn in params["dependencies"]
    ]  # consistency check

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
