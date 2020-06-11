import os

import dicthash

import sys

sys.path.insert(0, "../includes/")
import write_job_utils as utils  # noqa: E402

params = {
    "machine_params": {
        # machine setup
        "submit_command": "sbatch",
        "jobfile_template": "juwels_template.jdf",
        "jobname": <jobname>,
        "wall_clock_limit": "24:00:00",
        "n_processes": 20,
        "n_threads": 16,
        "n_nodes": 5,
        "mail": <mail>,
        "account": <account>,
        "partition": <partition>,
        "sim_script": "main.py",
        "submit_job": True,
        "python_binary": None,
        "nest_lib": None,
    },
    # misc
    "seed_seed": 123,
    "output_file_template": "res_{label}.pkl",
    "record_states_training": False,
    "record_weights_training": False,
    "exp_params": {
        "base_rng": 11214124,
        "Npre": 1000,
        "Npost": 1,
        "T": 1000.0,
        "T_stim": 100.0,
        "T_inter": 400.0,
        "tbin": 1.0,
        "bg_rate": 3.0,
        "neuron_params": {"V_reset": 0.0, "E_L": 0.0, "V_th": 20.0, "tau_m": 18.0},
        # 'gamma': 1.
    },
    "synapse_params": {
        "expr_facilitate": "0.0",
        "expr_depress": "0.0",
        "receptor_type": 0,
        "delay": 1.0,
        "Wmax": 1.0,
        "tau_plus": 20.0,
        "weight": "norm_dist",
    },
    "gp_params": {
        "seed": 123,
        "n_threads": 16,
        "min_fitness": 10.0,
        "max_generations": 1,
        # Type of fitness aggregation across trials
        "fitness": "min",
        # No. of trials per individual
        "n_trials": 8,
        # evo parameters
        "population_params": {"n_parents": 8, "mutation_rate": 0.05},
        "ea_params": {"n_offsprings": 8, "n_breeding": 8, "tournament_size": 1},
        "genome_params": {
            "n_inputs": 2,
            "n_outputs": 1,
            "n_columns": 5,
            "n_rows": 1,
            "levels_back": None,
        },
    },
}

params["md5_hash_sim_script"] = utils.md5_file(params["machine_params"]["sim_script"])

key = dicthash.generate_hash_from_dict(params, blacklist=[("gp_params", "max_generations")])

params["machine_params"]["workingdir"] = os.getcwd()
params["machine_params"]["outputdir"] = os.path.join(params["machine_params"]["workingdir"], key)

print("preparing job")
print(" ", params["machine_params"]["outputdir"])

utils.mkdirp(params["machine_params"]["outputdir"])
utils.write_json(params, f"{params['machine_params']['outputdir']}/params.json")
utils.create_jobfile(params["machine_params"])
utils.copy_file(params["machine_params"]["sim_script"], params["machine_params"]["outputdir"])
for fn in ["write_job.py", "sim_utils.py", "primitive_utils.py"]:
    utils.copy_file(fn, params["machine_params"]["outputdir"])

os.chdir(params["machine_params"]["outputdir"])
