import numpy as np
import nest


def generate_poisson_processes(rate, duration, size, resolution):
    counts = np.random.poisson(duration * 1e-3 * rate, size)
    times = np.array(
        [np.sort(np.random.uniform(0.0 + resolution, duration, size=count)) for count in counts]
    )
    return times - (times % resolution)


def error(u, u_target, offset_time=0.0):
    offset = int(offset_time / 1.0)  # recording interval is 1ms by default
    return np.sqrt(np.sum((u[offset:] - u_target[offset:]) ** 2))


def _set_up_nest_kernel(sim_params, trial):
    nest.ResetKernel()
    nest.hl_api.set_verbosity("M_ERROR")
    nest.SetKernelStatus(
        {
            "grng_seed": sim_params["seed"] + trial,
            "rng_seeds": [sim_params["seed"] + trial],
            "resolution": sim_params["dt"],
        }
    )


def _create_nodes(sim_params):
    inputs_sg = nest.Create("spike_generator", sim_params["n_inputs"])
    inputs_pa = nest.Create("parrot_neuron", sim_params["n_inputs"], {"time_driven": True})
    student = nest.Create(
        "iaf_psc_exp",
        sim_params["n_outputs"],
        {
            "time_driven": True,
            "rho": sim_params["rho"],
            "delta": sim_params["delta"],
            "reset_after_spike": False,
        },
    )
    teacher = nest.Create(
        "iaf_psc_exp",
        sim_params["n_outputs"],
        {
            "time_driven": True,
            "rho": sim_params["rho"],
            "delta": sim_params["delta"],
            "reset_after_spike": False,
        },
    )
    m_student = nest.Create("multimeter", 1, {"record_from": ["V_m"]})
    m_teacher = nest.Create("multimeter", 1, {"record_from": ["V_m"]})

    return inputs_sg, inputs_pa, student, teacher, m_student, m_teacher


def _connect_nodes(
    sim_params, inputs_sg, inputs_pa, student, teacher, m_student, m_teacher, expr_str
):
    nest.Connect(inputs_sg, inputs_pa, {"rule": "one_to_one"})

    nest.SetDefaults(
        "us_sympy_synapse",
        {
            "rho": sim_params["rho"],
            "delta": sim_params["delta"],
            "eta": sim_params["lr"],
            "tau_I": 100.0,
            "expr": expr_str,
        },
    )

    nest.Connect(
        inputs_pa,
        student,
        {"rule": "all_to_all"},
        {"synapse_model": "us_sympy_synapse", "weight": sim_params["initial_weight"]},
    )

    nest.Connect(inputs_pa, teacher, {"rule": "all_to_all"}, {"synapse_model": "static_synapse"})
    nest.Connect(m_student, student)
    nest.Connect(m_teacher, teacher)


def _set_input_patterns(sim_params, inputs_sg):
    input_rates = np.random.uniform(*sim_params["range_input_rates"], size=sim_params["n_inputs"])
    for i in range(sim_params["n_inputs"]):
        input_pattern = generate_poisson_processes(
            input_rates[i], sim_params["t_sim"], 1, sim_params["dt"]
        )
        nest.SetStatus(inputs_sg[i], {"spike_times": input_pattern[0]})


def _set_random_teacher_weights(sim_params, inputs_pa, teacher):
    if sim_params["do_shift_weights"]:
        if (
            np.random.rand() < 0.0
        ):  # apply a random shift to avoid bias towards u_target > u or u_target < u
            weights_shift = -15.0
        else:
            weights_shift = 15.0
    else:
        weights_shift = 0.0
    for conn in nest.GetConnections(source=inputs_pa, target=teacher):
        nest.SetStatus(
            conn,
            {"weight": np.random.uniform(*sim_params["range_teacher_weights"]) + weights_shift},
        )


def sim(sim_params, expr_str, trial):

    assert sim_params["n_outputs"] == 1

    np.random.seed(sim_params["seed"] + trial)

    _set_up_nest_kernel(sim_params, trial)

    inputs_sg, inputs_pa, student, teacher, m_student, m_teacher = _create_nodes(sim_params)

    _connect_nodes(
        sim_params, inputs_sg, inputs_pa, student, teacher, m_student, m_teacher, expr_str
    )

    _set_input_patterns(sim_params, inputs_sg)

    # simulate a single step to sort connections such that they remain
    # in constant order for the rest of the script
    nest.Simulate(sim_params["dt"])

    _set_random_teacher_weights(sim_params, inputs_pa, teacher)
    weights_teacher = nest.GetStatus(
        nest.GetConnections(source=inputs_pa, target=teacher), "weight"
    )

    # perform training
    history_weights_student = []
    history_weights_student.append(
        nest.GetStatus(nest.GetConnections(source=inputs_pa, target=student), "weight")
    )
    for i in range(int(sim_params["t_sim"] / sim_params["t_sim_step"])):
        nest.SetStatus(
            student, {"u_target": nest.GetStatus(teacher, "V_m")[0]}
        )  # set target potential
        nest.Simulate(sim_params["t_sim_step"])
        history_weights_student.append(
            nest.GetStatus(nest.GetConnections(source=inputs_pa, target=student), "weight")
        )

    history_u_student = nest.GetStatus(m_student, "events")[0]
    history_u_teacher = nest.GetStatus(m_teacher, "events")[0]

    return (history_u_student, history_u_teacher, history_weights_student, weights_teacher)
