import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pyx
import sys

from colors import fitness_colors, red

sys.path.insert(0, "../experiments/includes/")
import utils  # noqa: E402

matplotlib.rc_file("plotstyle.rc")


def plot_raster(data_directory, sim_params, ax_initial, ax_final, ax_frame):
    """Create raster plot of initial and final phases of learning"""

    ax_initial.set_ylabel("Neuron")
    ax_initial.set_xlim(
        0.0, (params["n_trials_raster"] * sim_params["stimulus_duration"] + 50.0) * 1e-3
    )
    ax_initial.set_ylim(params["ylim_raster"])
    ax_initial.set_title("Early training", pad=5.0)

    ax_final.set_yticks([])
    ax_final.set_xlim(
        (
            (sim_params["n_training_trials"] - params["n_trials_raster"])
            * sim_params["stimulus_duration"]
            - 50.0
        )
        * 1e-3,
        (sim_params["n_training_trials"] * sim_params["stimulus_duration"]) * 1e-3,
    )
    ax_final.set_ylim(params["ylim_raster"])
    ax_final.set_title("Late training", pad=5.0)

    ax_frame.set_xlim([0, 1])
    ax_frame.set_ylim([0, 1])

    Delta_t = (sim_params["n_training_trials"] - 2 * params["n_trials_raster"]) * sim_params[
        "stimulus_duration"
    ]
    ax_frame.text(0.525, 0.45, f"{Delta_t / 1000.:.0f}s", rotation=90)
    ax_frame.set_xlabel("Time (s)", labelpad=20)

    def is_correct_classification(target_activity, outputs):
        """Determine whether output activity matches target_activity"""
        assert (abs(target_activity + 1.0) < 1e-9) or (
            abs(target_activity - 1.0) < 1e-9
        ), target_activity

        if abs(target_activity + 1.0) < 1e-9 and len(outputs) == 0:
            return True
        elif abs(target_activity - 1.0) < 1e-9 and len(outputs) > 0:
            return True
        else:
            return False

    for trial in np.concatenate(
        [
            np.arange(params["n_trials_raster"]),
            np.arange(
                sim_params["n_training_trials"] - params["n_trials_raster"],
                sim_params["n_training_trials"],
            ),
        ]
    ):

        inputs = np.load(
            os.path.join(data_directory, sim_params["file_name_input"]).format(
                generation=0, label=trial
            ),
            allow_pickle=True,
        )

        target_activity, outputs = np.load(
            os.path.join(data_directory, sim_params["file_name_output"]).format(
                generation=0, label=trial
            ),
            allow_pickle=True,
        )

        if trial < params["n_trials_raster"]:
            ax = ax_initial
        else:
            ax = ax_final

        time_offset = trial * sim_params["stimulus_duration"]
        for i in range(sim_params["n_inputs"]):
            ax.plot(
                (inputs[i] + time_offset) * 1e-3,
                np.ones_like(inputs[i]) * i,
                ".",
                color="k",
                ms=2.0,
            )
        ax.plot(
            (outputs + time_offset) * 1e-3,
            np.ones_like(outputs) * sim_params["n_inputs"] + 3,
            ".",
            color=red,
            markersize=4,
            ls="",
        )

        if target_activity < 0:
            rect_color = params["raster_pattern_colors"][0]
        else:
            rect_color = params["raster_pattern_colors"][1]
        rect = matplotlib.patches.Rectangle(
            (time_offset * 1e-3, -1),
            sim_params["stimulus_duration"] * 1e-3,
            sim_params["n_inputs"] + 4,
            linewidth=0,
            facecolor=rect_color,
            zorder=-2,
        )
        ax.add_patch(rect)
        ax.vlines(
            1.0e-3 * (time_offset + sim_params["stimulus_duration"]),
            -1,
            sim_params["n_inputs"] + 3,
            lw=0.5,
            color="k",
        )

        x_pos = time_offset + sim_params["stimulus_duration"] / 2.0
        if is_correct_classification(target_activity, outputs):
            ax.plot(
                x_pos * 1e-3,
                sim_params["n_inputs"] + 10,
                marker=r"$\checkmark$",
                color="k",
                markersize=6,
            )

    # Broken axis symbol
    d = 0.025  # how big to make the diagonal lines in axes coordinates
    d0 = 1.0
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(color="k", clip_on=False, lw=0.8)
    ax_initial.plot((d0 - d, d0 + d), (-d, +d), transform=ax_initial.transAxes, **kwargs)
    ax_final.plot((-d, +d), (-d, +d), transform=ax_final.transAxes, **kwargs)


def plot_fitness(ax, ax_test, ax_single):
    """Plot fitness over generation index"""

    def format_label(expr):
        """Turn a sympy expression into something more readable"""
        expr = (r"$" + expr + "$").replace("**", "^").replace("*", "")
        for var in params["variable_name_mapping"]:
            expr = expr.replace(var, params["variable_name_mapping"][var])
        return r"" + expr

    n_runs = 8  # manual selection, check number of jobs in juwels/juwels_template.jdf

    ax.set_xlabel("Generation index")
    ax.set_ylabel("Fitness")
    ax.set_xticks(range(0, 501, 100))
    ax.set_xlim(params["xlim_fitness"])
    ax.set_ylim(params["ylim_fitness"])

    ax_test.set_xlim([-1, 1])
    ax_test.set_ylim(params["ylim_fitness"])
    ax_test.spines["left"].set_visible(False)
    ax_test.set_xticks([])
    ax_test.set_xticklabels([""])
    ax_test.set_yticks([])

    ax_single.set_xlabel("Generation index")
    ax_single.set_xlim(params["xlim_fitness"])
    ax_single.set_ylim(params["ylim_fitness"])
    ax_single.set_xticks(range(0, 501, 100))
    ax_single.set_yticklabels([])

    # evolution of all runs
    ax_fitness.axhline(210.2, color="0.85", lw=3)

    labeled_runs = (2, 3, 5, 6)  # manually selected highest fitness runs

    for i in range(n_runs):
        with open(os.path.join(data_directory, f"res_{i}.pkl"), "rb") as f:
            res = pickle.load(f)

            if i in labeled_runs:
                ax_fitness.plot(
                    res["fitness_champion"],
                    label=format_label(res["expr_champion"][-1]),
                    color=fitness_colors[i],
                )
            else:
                ax_fitness.plot(res["fitness_champion"], color="0.5", zorder=-1)

    ax_fitness.legend(fontsize=4.0, loc="lower right")

    idx = 6  # manually selected to be most interesting

    # validation data
    with open(os.path.join(data_directory, "res_test.pkl"), "rb") as f:
        res_test = pickle.load(f)
    ax_test.errorbar(
        0.0,
        np.mean(res_test),
        yerr=np.std(res_test),
        ls="",
        marker="o",
        elinewidth=1.0,
        capsize=1.2,
        color=fitness_colors[idx],
        markersize=4,
    )

    # evolution of single run
    ax_fitness_single.axhline(210.2, color="0.85", lw=3)

    with open(os.path.join(data_directory, f"res_{idx}.pkl"), "rb") as f:
        res = pickle.load(f)
        ax_fitness_single.plot(res["fitness_champion"], label=i, color=fitness_colors[idx])

        fitness_prev = -np.inf
        n_steps = 0
        for i, (fitness, expr) in enumerate(zip(res["fitness_champion"], res["expr_champion"])):
            if abs(fitness_prev - fitness) > 1e-8:
                if n_steps == 3:  # manual fix for label position
                    offset = 10
                else:
                    offset = 0
                ax_fitness_single.text(i + 5 + offset, fitness + 8, format_label(expr), fontsize=9)

                fitness_prev = fitness
                n_steps += 1


params = {
    "variable_name_mapping": {"x_0": r"R", "x_1": r"E_j^\text{r}"},
    "n_trials_raster": 8,
    "raster_pattern_colors": ["1.0", "0.7"],
    "ylim_raster": (0, 65),
    "xlim_fitness": (-10, 510),
    "ylim_fitness": (-65, 260),
    "expr_yoffset_fitness_single": 37,
}

if __name__ == "__main__":

    data_directory = utils.get_data_directory_from_cmd_args()

    with open(os.path.join(data_directory, "sim_params.json"), "r") as f:
        sim_params = json.load(f)

    figsize = (6.4, 3.5)
    fig = plt.figure(figsize=figsize)

    ax_sketch = fig.add_axes([0.1, 0.65, 0.24, 0.28], frameon=False)  # place holder for sketch
    ax_sketch.set_xticks([])
    ax_sketch.set_yticks([])
    ax_raster_initial = fig.add_axes([0.45, 0.65, 0.24, 0.28])
    ax_raster_final = fig.add_axes([0.72, 0.65, 0.24, 0.28])
    ax_raster_final.spines["left"].set_visible(False)
    ax_raster_frame = fig.add_axes([0.45, 0.65, 0.47, 0.28], frameon=False, xticks=[], yticks=[])
    ax_fitness = fig.add_axes([0.1, 0.12, 0.39, 0.37])
    ax_fitness_test = fig.add_axes([0.505, 0.12, 0.01, 0.37])
    ax_fitness_single = fig.add_axes([0.57, 0.12, 0.39, 0.37])

    # Panel labels
    fd = {
        "fontsize": 10,
        "weight": "bold",
        "horizontalalignment": "left",
        "verticalalignment": "bottom",
    }

    plot_raster(data_directory, sim_params, ax_raster_initial, ax_raster_final, ax_raster_frame)
    plot_fitness(ax_fitness, ax_fitness_test, ax_fitness_single)

    for ax_label, ax in zip(
        ["A", "B", "C", "D"], [ax_sketch, ax_raster_initial, ax_fitness, ax_fitness_single]
    ):
        pos = ax.get_position()

        plt.text(
            pos.x0 - 0.03,
            pos.y1 + 0.02,
            r"\bfseries{}" + ax_label,
            fontdict=fd,
            transform=fig.transFigure,
        )

    figname = "Fig3_reward_driven_results.eps"
    print(f"creating {figname}")
    plt.savefig(figname, dpi=1200)

    # Merge sketch and results
    c = pyx.canvas.canvas()
    c.insert(pyx.epsfile.epsfile(0.0, 0.0, figname, width=figsize[0], height=figsize[1]))
    c.insert(
        pyx.epsfile.epsfile(0.3, 2.1, "Fig3_reward_driven_sketch.eps", width=0.3 * figsize[0])
    )

    print("creating Fig3_reward_driven.eps")
    c.writeEPSfile("Fig3_reward_driven.eps")
