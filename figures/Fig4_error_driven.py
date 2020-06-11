import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pyx
import sys

from colors import blue, fitness_colors, synaptic_weight_colors

sys.path.insert(0, "../experiments/includes/")
import utils  # noqa: E402

matplotlib.rc_file("plotstyle.rc")


def low_pass_filter(x, tau, dt):
    """Filter a signal x of resolution dt with an exponential kernel with
    time constant tau"""

    assert tau > dt
    P = np.exp(-dt / tau)
    x_filtered = np.empty_like(x)
    x_filtered[0] = x[0]
    for i in range(1, len(x)):
        x_filtered[i] = P * x_filtered[i - 1] + (1.0 - P) * x[i]

    return x_filtered


def plot_error(data_directory, ax_error):
    """Plot error traces"""

    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel(r"$\|v(t) - u(t)\|$")
    ax_error.set_xlim(params["xlim_error"])
    ax_error.set_ylim(params["ylim_error"])
    ax_error.set_yscale("log")

    with open(os.path.join(data_directory, "traces.pkl"), "rb") as f:
        res = pickle.load(f)

    error = np.sqrt((res["data_u"]["V_m"] - res["data_u_teacher"]["V_m"]) ** 2)
    dt = res["data_u"]["times"][1] - res["data_u"]["times"][0]
    error = low_pass_filter(error, 10.0, dt)

    n_cut = int(
        0.0005 * len(res["data_u"]["times"])
    )  # cut of initial times before the first input spikes arrive

    ax_error.plot(res["data_u"]["times"][n_cut:] * 1e-3, error[n_cut:], color=blue)


def plot_synaptic_weights(data_directory, sim_params, ax):
    """Plot synaptic weight traces"""

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Synaptic weight")
    ax.set_xlim(params["xlim_weights"])
    ax.set_ylim(params["ylim_weights"])

    with open(os.path.join(data_directory, "traces.pkl"), "rb") as f:
        res = pickle.load(f)

    n_cut = int(
        0.0005 * len(res["data_u"]["times"])
    )  # cut of initial times before the first input spikes arrive

    times_weights = (
        np.arange(0.0, int(sim_params["t_sim"] / sim_params["t_sim_step"]))
        * sim_params["t_sim_step"]
        * 1e-3
    )
    for i in range(len(res["data_syn"][0])):
        ax.plot(
            times_weights[n_cut:],
            [h[i] for h in res["data_syn"]][n_cut:-1],
            color=synaptic_weight_colors[i],
            zorder=1,
        )
        ax.axhline(res["data_syn_teacher"][i], color=synaptic_weight_colors[i], zorder=-1, ls="--")


def plot_fitness(data_directory, ax, ax_test):
    """Plot fitness of best individual over generation index"""

    n_runs = 6  # manual setting, check number of jobs in juwels/juwels_template.jdf

    ax.set_xlabel("Generation index")
    ax.set_ylabel("Fitness")
    ax.set_xlim(params["xlim_fitness"])
    ax.set_ylim(params["ylim_fitness"])
    ax.set_yscale("symlog")

    ax_test.set_xlim([-0.65, 5.65])
    ax_test.set_ylim(params["ylim_fitness_test"])
    ax_test.set_xticks([])

    for i in range(n_runs):

        with open(os.path.join(data_directory, f"res_{i}.pkl"), "rb") as f:
            res = pickle.load(f)

        expr_final = (r"$" + res["expr_champion"][-1] + "$").replace("**", "^").replace("*", "")
        expr_final = expr_final.replace("x_2(x_0 - x_1)", "(x_0 - x_1)x_2")
        for var in params["variable_mapping"]:
            expr_final = expr_final.replace(var, params["variable_mapping"][var])
        ax.plot(res["fitness_champion"], label=expr_final, color=fitness_colors[i])

        # validation data
        with open(os.path.join(data_directory, f"res_test_{i}.pkl"), "rb") as f:
            res_test = pickle.load(f)
        ax_test.errorbar(
            i,
            np.mean(res_test),
            yerr=np.std(res_test),
            ls="",
            marker="o",
            elinewidth=1.0,
            capsize=1.2,
            color=fitness_colors[i],
            markersize=4,
        )

    ax.legend(loc=(1.35, 0.1))
    ax_fitness.set_yticks(-(10 ** np.array([5, 3, 1])))


params = {
    "variable_mapping": {"x_0": r"v", "x_1": r"u", "x_2": r"\bar s_j"},
    "xlim_mem": (0.0, 20.0),
    "ylim_mem": (-75.0, -59.0),
    "xlim_error": (-0.25, 20.0),
    "ylim_error": (1e-3, 1e1),
    "xlim_weights": (-0.25, 20.0),
    "ylim_weights": (0, 35),
    "xlim_fitness": (0, 1e3),
    "ylim_fitness": (-1e5, -5.0),
    "xlim_fitness_inset": (500, 1e3),
    "ylim_fitness_inset": (-15, -13),
    "ylim_fitness_test": (-35, -5.0),
}

if __name__ == "__main__":

    data_directory = utils.get_data_directory_from_cmd_args()

    with open(os.path.join(data_directory, "sim_params.json"), "r") as f:
        sim_params = json.load(f)

    sim_params["t_sim"] *= 2.0

    figsize = (6.4, 3.5)
    fig = plt.figure(figsize=figsize)

    ax_sketch = fig.add_axes([0.1, 0.65, 0.24, 0.28], frameon=False)  # place holder for sketch
    ax_sketch.set_xticks([])
    ax_sketch.set_yticks([])
    ax_error = fig.add_axes([0.45, 0.65, 0.21, 0.28])
    ax_weights = fig.add_axes([0.75, 0.65, 0.21, 0.28])
    ax_fitness = fig.add_axes([0.1, 0.12, 0.4, 0.37])
    ax_fitness_test = fig.add_axes([0.56, 0.12, 0.07, 0.37])

    # Panel labels
    fd = {
        "fontsize": 10,
        "weight": "bold",
        "horizontalalignment": "left",
        "verticalalignment": "bottom",
    }
    for ax_label, ax in zip(["A", "B", "C", "D"], [ax_sketch, ax_error, ax_weights, ax_fitness]):
        pos = ax.get_position()

        plt.text(
            pos.x0 - 0.03,
            pos.y1 + 0.02,
            r"\bfseries{}" + ax_label,
            fontdict=fd,
            transform=fig.transFigure,
        )

    plot_error(data_directory, ax_error)
    plot_synaptic_weights(data_directory, sim_params, ax_weights)
    plot_fitness(data_directory, ax_fitness, ax_fitness_test)

    figname = "Fig4_error_driven_results.eps"
    print(f"creating {figname}")
    plt.savefig(figname, dpi=1200)

    # Merge sketch and results
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0.0, 0.0, figname, width=figsize[0], height=figsize[1]))
    c.insert(
        pyx.epsfile.epsfile(0.3, 2.1, "Fig4_error_driven_sketch.eps", width=0.28 * figsize[0])
    )

    print("creating Fig4_error_driven.eps")
    c.writeEPSfile("Fig4_error_driven.eps")
