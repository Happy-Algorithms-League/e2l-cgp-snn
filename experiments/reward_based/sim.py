import json
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

import nest


def generate_poisson_processes(rate, duration, size, resolution):
    counts = np.random.poisson(duration * 1e-3 * rate, size)
    times = np.array(
        [np.sort(np.random.uniform(0.0 + resolution, duration, size=count)) for count in counts]
    )
    return times - (times % resolution)


def generate_stimuli(params):
    input_spike_trains = []
    target_activity = []

    for i in range(params["n_stimuli"]):
        input_spike_trains.append(
            generate_poisson_processes(
                params["input_rate"], params["stimulus_duration"], params["n_inputs"], params["dt"]
            )
        )
        target_activity.append(np.random.choice([-1.0, 1.0]))

    return input_spike_trains, target_activity


def dump_stimuli_to_file(params, generation, input_spike_trains, target_activity):
    stimuli = []
    for st, a in zip(input_spike_trains, target_activity):
        stimuli.append([st, a])

    np.save(params["file_name_stimuli"].format(generation=generation), stimuli)


def load_stimuli_from_file(params):
    stimuli = np.load(params["file_name_stimuli"])
    input_spike_trains = []
    target_activity = []

    for s in stimuli:
        input_spike_trains.append(s[0])
        target_activity.append(s[1])

    return input_spike_trains, target_activity


def dump_input_to_file(params, spike_trains, generation, label):
    np.save(params["file_name_input"].format(generation=generation, label=label), spike_trains)


def dump_output_to_file(params, sd, target_activity, generation, label):
    events = nest.GetStatus(sd, "events")[0]
    assert len(np.unique(events["senders"])) <= 1
    np.save(
        params["file_name_output"].format(generation=generation, label=label),
        [target_activity, events["times"]],
    )


def set_input_spike_trains(params, inputs, input_spike_trains):
    for i in range(params["n_inputs"]):
        if len(input_spike_trains[i]) > 0:
            nest.SetStatus(inputs[i], {"spike_times": input_spike_trains[i]})


def init_kernel(params):
    nest.ResetKernel()
    nest.hl_api.set_verbosity("M_ERROR")
    nest.SetKernelStatus({"resolution": params["dt"]})


def build_nodes(params):
    # create input units
    inputs_sg = nest.Create("spike_generator", params["n_inputs"])
    inputs_pa = nest.Create("parrot_neuron", params["n_inputs"], {"time_driven": True})

    # create neurons and recording devices
    neurons = nest.Create(
        "iaf_psc_exp",
        params["population_size"],
        {
            "rho": params["rho"],
            "delta": params["delta"],
            "E_L": params["E_L"],
            "time_driven": True,
        },
    )
    sd = nest.Create("spike_detector")
    # m = nest.Create('multimeter', 1, {'interval': 0.1, 'record_from': ['V_m']})
    m = None

    return inputs_sg, inputs_pa, neurons, sd, m


def connect_nodes(params, inputs_sg, inputs_pa, neurons, sd, m):
    # connect inputs to outputs
    nest.Connect(inputs_sg, inputs_pa, {"rule": "one_to_one"})

    nest.Connect(
        inputs_pa,
        neurons,
        {"rule": "pairwise_bernoulli", "p": params["input_connectivity"]},
        {"synapse_model": "usrl_synapse", "weight": params["initial_weight"]},
    )

    conn = nest.GetConnections(source=inputs_pa, target=neurons)
    for c in conn:
        nest.SetStatus(c, {"weight": np.random.normal(0.0, params["initial_weight"])})

    # connect recording devices
    nest.Connect(neurons, sd)
    # nest.Connect(m, [neurons[0]])


def connect_nodes_from_file(params, label):
    connections = np.load(params["file_name_template"].format(label=label))

    for c in connections:
        nest.Connect(
            [int(c[0])],
            [int(c[1])],
            syn_spec={"weight": float(c[2]), "delay": float(c[3]), "model": c[4]},
        )


def connect_nodes_from_dict(params, connectivity):

    for c in connectivity:
        nest.Connect(
            np.array([c["source"]]),
            np.array([c["target"]]),
            syn_spec={
                "weight": np.array([c["weight"]]),
                "delay": np.array([c["delay"]]),
                "synapse_model": c["synapse_model"],
            },
        )


def dump_connections_to_file(params, generation, label):
    conn_status = nest.GetStatus(nest.GetConnections())

    connections = []
    for cs in conn_status:
        connections.append(
            [cs["source"], cs["target"], cs["weight"], cs["delay"], cs["synapse_model"].name]
        )

    np.save(params["file_name_template"].format(generation=generation, label=label), connections)


def compute_reward(params, neurons, sd, target_activity):
    spike_data = nest.GetStatus(sd, "events")[0]
    c = []
    for target in neurons:
        target_gid = target.tolist()[0]
        c.append(int(target_gid in spike_data["senders"]))
    s = -1 if np.sum(c) < params["population_size"] / 2 else 1
    R = -1 if s != target_activity else 1

    return c, R


def update_weight(params, inputs_pa, neurons, R, plasticity_rule):

    history_E = []
    history_w = []
    history_delta_w = []
    for target_gid in neurons:
        conn = nest.GetConnections(source=inputs_pa, target=target_gid)
        for conn_i in conn:
            w = nest.GetStatus(conn_i, "weight")[0]
            E = nest.GetStatus(conn_i, "E")[0]
            delta_w = plasticity_rule(params["learning_rate"], R, E)

            history_E.append(E)
            history_w.append(w)
            history_delta_w.append(delta_w)

            nest.SetStatus(conn_i, {"weight": w + delta_w})

    return history_E, history_w, history_delta_w


def train(params, plasticity_rule, seed_offset=0):

    np.random.seed(params["seed"] + seed_offset)

    input_spike_trains, target_activity = generate_stimuli(params)

    if params["record_stimuli_to_file"]:
        dump_stimuli_to_file(params, params["generation"], input_spike_trains, target_activity)

    # initial set up of network, will be restored to this state in
    # every training iteration
    init_kernel(params)

    inputs_sg, inputs_pa, neurons, sd, m = build_nodes(params)

    connect_nodes(params, inputs_sg, inputs_pa, neurons, sd, m)

    nest.Simulate(1)  # sorts connections

    if params["record_connections_to_file"]:
        dump_connections_to_file(params, params["generation"], "initial")

    connectivity = nest.GetStatus(nest.GetConnections())

    history_R = []  # TODO resize before sim and avoid appends
    history_c = []
    history_E = []
    history_w = []
    history_delta_w = []
    for trial in range(params["n_training_trials"]):

        # restore initial state
        init_kernel(params)
        inputs_sg, inputs_pa, neurons, sd, m = build_nodes(params)
        connect_nodes_from_dict(params, connectivity)

        # select random input pattern
        pattern = np.random.randint(0, params["n_stimuli"])
        set_input_spike_trains(params, inputs_sg, input_spike_trains[pattern])

        if params["record_input_to_file"]:
            dump_input_to_file(params, input_spike_trains[pattern], params["generation"], trial)

        nest.Simulate(params["stimulus_duration"])

        c, R = compute_reward(params, neurons, sd, target_activity[pattern])

        if params["record_output_to_file"]:
            dump_output_to_file(params, sd, target_activity[pattern], params["generation"], trial)

        history_E_tmp, history_w_tmp, history_delta_w_tmp = update_weight(
            params, inputs_pa, neurons, R, plasticity_rule
        )

        history_E.append(history_E_tmp)
        history_w.append(history_w_tmp)
        history_delta_w.append(history_delta_w_tmp)

        connectivity = nest.GetStatus(nest.GetConnections())

        if params["record_connections_to_file"]:
            dump_connections_to_file(params, params["generation"], trial)

        history_R.append(R)
        history_c.append(c)

    return (
        history_R,
        history_c,
        np.array(history_E),
        np.array(history_w),
        np.array(history_delta_w),
    )


if __name__ == "__main__":

    nest.Install("usrl_synapse_module")

    repo_root = os.path.abspath(__file__).split("experiments")[0]
    sys.path.insert(0, os.path.join(repo_root, "figures/reward_driven/"))
    from create_manuscript_figure import plot_raster

    def get_data_directory():

        try:
            directory = sys.argv[1]
        except IndexError:
            raise ValueError("please provide data directory as argument")

        if directory[-1] != "/":
            directory += "/"

        return directory

    data_directory = get_data_directory()

    with open(os.path.join(data_directory, "sim_params.json"), "r") as f:
        params = json.load(f)

    # params['n_training_trials'] = 50

    params["generation"] = 0  # dummy
    params["file_name_template"] = os.path.join(data_directory, params["file_name_template"])
    params["file_name_input"] = os.path.join(data_directory, params["file_name_input"])
    params["file_name_output"] = os.path.join(data_directory, params["file_name_output"])

    params["record_input_to_file"] = True
    params["record_output_to_file"] = True

    def plasticity_rule(learning_rate, R, E):
        """The learning rule given in Urbanczik & Senn, 2009"""
        return learning_rate * E * (R - 1.0)

    trial = 1  # dummy
    history_R, _, _, _, _ = train(params, plasticity_rule, trial)

    width = 6.4
    figsize = (width, width / 3 / 1.618)
    fig = plt.figure(figsize=figsize)

    ax_raster_initial = fig.add_axes([0.1, 0.32, 0.22, 0.52])
    ax_raster_final = fig.add_axes([0.34, 0.32, 0.22, 0.52])
    ax_raster_final.spines["left"].set_visible(False)
    ax_raster_frame = fig.add_axes([0.1, 0.32, 0.42, 0.52], frameon=False, xticks=[], yticks=[])

    ax_cumulative_reward = fig.add_axes([0.67, 0.32, 0.3, 0.52])
    ax_cumulative_reward.set_xlabel("Trial")
    ax_cumulative_reward.set_ylabel("Cum. reward")

    plot_raster("", params, ax_raster_initial, ax_raster_final, ax_raster_frame)

    ax_cumulative_reward.plot(np.cumsum(history_R))

    figname = "sim.pdf"
    print(f"creating {figname}")
    plt.savefig(figname, dpi=300)
