import copy
import nest
import numpy as np
import json
import os
import pylab as pl
import sys

from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib import rc_file

import pyx
from colors import orange, lightblue, red
from colors import selected_colors, selected_color_shades

from Fig5_corr_learning_utils import deltaW, analytical_learning_rule

sys.path.insert(0, "../experiments/includes/")
import utils  # noqa: E402

rc_file("plotstyle.rc")

nest.Install("HomeostaticSTDPmodule")
nest.Install("stdp_sympy_synapse_module")

data_directory = utils.get_data_directory_from_cmd_args()

"""
Figure layout
"""
fig = pl.figure(figsize=(6.4, 5.94))
axes = {}
gs0 = gridspec.GridSpec(3, 3)
gs0.update(left=0.1, right=0.97, top=0.95, wspace=0.5, hspace=0.5, bottom=0.1)
axes["sketch"] = pl.subplot(gs0[0, 0], frameon=False)
axes["task"] = pl.subplot(gs0[0, 1])

axes_vm = pl.subplot(gs0[0, 2], frameon=False)
axes_vm.set_xticks([])
axes_vm.set_yticks([])
pos = axes_vm.get_position()

gs1 = gridspec.GridSpec(1, 3, wspace=0.45, hspace=0.4)
gs1.update(left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1)
axes_vm = [pl.subplot(gs1[0, j]) for j in range(3)]
axes["vm"] = axes_vm[0]

ax = pl.subplot(gs0[1, 0], frameon=False)
ax.set_xticks([])
ax.set_yticks([])
ax_pos = ax.get_position()
width = ax_pos.x1 - ax_pos.x0
axes["gp_fitness"] = pl.axes((ax_pos.x0, ax_pos.y0, 0.85 * width, ax_pos.y1 - ax_pos.y0))
ax_valid = pl.axes((ax_pos.x0 + 0.9 * width, ax_pos.y0, 0.15 * width, ax_pos.y1 - ax_pos.y0))

axes["gp_lr1"] = pl.subplot(gs0[1, 1])
axes["gp_lr2"] = pl.subplot(gs0[1, 2])

axes["stdp"] = pl.subplot(gs0[2, 0])
axes["gp_lr1_eff"] = pl.subplot(gs0[2, 1])
axes["gp_lr2_eff"] = pl.subplot(gs0[2, 2])

axes_gp_lr_list = [axes["gp_lr1"], axes["gp_lr2"]]
axes_gp_lr_eff_list = [axes["gp_lr1_eff"], axes["gp_lr2_eff"]]

# Panel labels
fd = {
    "fontsize": 10,
    "weight": "bold",
    "horizontalalignment": "left",
    "verticalalignment": "bottom",
}

axes_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
for (key, ax), label in zip(axes.items(), axes_labels):
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    pos = ax.get_position()
    pl.text(
        pos.x0 - 0.03,
        pos.y1 + 0.02,
        r"\bfseries{}" + label,
        fontdict=fd,
        transform=fig.transFigure,
    )


axes["sketch"].set_xticks([])
axes["sketch"].set_yticks([])

"""
Explain task in Panels A and B
"""

ax = axes["task"]

with open(os.path.join(data_directory, "Fig5B_params.json"), "r") as f:
    exp_params_B = json.load(f)
input_spikes = np.load(os.path.join(data_directory, "Fig5B_spikes.npy"), allow_pickle=True)

t_max = exp_params_B["T"]
for i, spikes in enumerate(input_spikes):
    sp = spikes[spikes <= 1100.0]
    ax.plot(sp, np.ones_like(sp) * i, ".", color="k", ms=1.5)

ax.set_xlim((0.0, 1100.0))
ax.set_ylim((0, exp_params_B["Npre"]))

ax.fill_between(
    [500.0, 900.0], exp_params_B["Npre"] + 10, exp_params_B["Npre"] + 60, color=red, clip_on=False
)
ax.fill_between(
    [900.0, 1000.0],
    exp_params_B["Npre"] + 10,
    exp_params_B["Npre"] + 60,
    color=lightblue,
    clip_on=False,
)
ax.text(550.0, exp_params_B["Npre"] + 85, r"$T_{\mathrm{inter}}$", rotation=0, color=red)
ax.text(910.0, exp_params_B["Npre"] + 85, r"$T_{\mathrm{pattern}}$", rotation=0, color=lightblue)


t = 0.1 + exp_params_B["T_inter"]
while t < 2000.0:
    ax.fill_betweenx([0, exp_params_B["Npre"]], t, t + exp_params_B["T_stim"], color="0.7")
    t += exp_params_B["T_stim"] + exp_params_B["T_inter"]

ax.set_xticks([0.0, 500.0, 1000.0])
ax.set_xticklabels([r"$0.$", r"$0.5$", r"$1.0$"])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Input neuron")
ax.set_yticks([])

"""
Panel C: Evolution of membrane potential of the post-synaptic neuron
"""

# Load data
with open(os.path.join(data_directory, "Fig5C_params.json"), "r") as f:
    exp_params_C = json.load(f)
vm_post = np.load(os.path.join(data_directory, "Fig5C_vm_post.npy"))
vm_times = np.arange(0.1, exp_params_C["T"] - 0.9, 0.1)

intervals = [(0.0, 1000.0), (49000.0, 50000.0), (199000.0, 200000.0)]

for i, (time_interval, ax) in enumerate(zip(intervals, axes_vm)):
    vm = vm_post[(vm_times >= time_interval[0]) & (vm_times < time_interval[1])]
    vm_t = vm_times[(vm_times >= time_interval[0]) & (vm_times < time_interval[1])]
    t = 0.1 + exp_params_C["T_inter"]
    while t < exp_params_C["T"]:
        if t >= time_interval[0] and t < time_interval[1]:
            ax.fill_between(
                [t - exp_params_C["T_inter"], t],
                vm.max() + 0.5,
                vm.max() + 1.5,
                color=red,
                clip_on=False,
            )
            ax.fill_between(
                [t, t + exp_params_C["T_stim"]],
                vm.max() + 0.5,
                vm.max() + 1.5,
                color=lightblue,
                clip_on=False,
            )
            ax.fill_betweenx([vm.min(), vm.max()], t, t + exp_params_C["T_stim"], color="0.7")
        t += exp_params_C["T_stim"] + exp_params_C["T_inter"]
    ax.plot(vm_t, vm, color=orange, lw=0.8)

    if i == 0:
        ax.set_ylabel("$u\, (\mathrm{mV})$")  # noqa: W605
    if i == 1:
        ax.set_xlabel("Time (s)")
    if i > 0:
        ax.set_yticks([])
        ax.spines["left"].set_color("none")
    ax.set_xticks(time_interval)
    ax.set_xticklabels([r"${}$".format(int(t / 1000)) for t in time_interval])

    d = 0.025
    kwargs = dict(color="k", clip_on=False, lw=0.8)
    if i < 2:
        ax.plot((1 - d, 1 + d), (-d, +d), transform=ax.transAxes, **kwargs)
    if i > 0:
        ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, **kwargs)


"""
Panel G: STDP rule of Masquelier (2017)
"""
synapse_params_homeo = {
    "lambda": 0.01,
    "mu_plus": 0.0,
    "tau_plus": 20.0,
    "Wout": -0.0015,
    "weight": 0.5,
    "Wmax": 1.0,
    "receptor_type": 0,
}
synapse_params_homeo[
    "receptor_type"
] = 1  # Needs to be set because of the parrot neuron used in deltaW

ax = axes["stdp"]

weight = 0.5
delay = 5.0
dt = 10.0
width = 200.0
delta_t_range = np.arange(-delay - width, delay + width, 5.0)


deltat, dw = deltaW(
    "stdp_homeostatic_synapse", synapse_params_homeo, delta_t_range, np.array([weight]), delay
)
ax.hlines(0.0, deltat[0], deltat[-1], linestyles="dashed", lw=1.0, color="0.5")

ax.plot(deltat, dw[0], color=orange)

ax.set_xlabel(r"$t_{\mathrm{post}} - t_{\mathrm{pre}}$")
ax.set_ylabel(r"$\Delta w_j$")
ax.yaxis.set_label_coords(-0.25, 0.5, transform=ax.transAxes)
ax.set_ylim((-0.0025, 0.01))
ax.set_yticks([synapse_params_homeo["Wout"], 0.0, 0.01])
ax.set_yticklabels([r"$w_{\mathrm{out}}$", r"$0.$", r"$0.01$"])
ax.set_xlim((-150.0, 150.0))


"""
Panels D, E, F: Results from evolution
"""
with open(os.path.join(data_directory, "params.json"), "r") as f:
    params = json.load(f)
    synapse_params = params["synapse_params"]


fitness_list = []
# Selected learning rules for further analysis
selected_runs = [9, 18]

selected_lr = []
runs = list(range(20))
# To ensure that the selected runs are plotted last
runs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19]
runs += selected_runs
for run in runs:
    fn = os.path.join(data_directory, "champion_{}.txt".format(run))
    fn2 = os.path.join(data_directory, "fitness_{}.npy".format(run))

    fitness = np.load(fn2)

    fitness_curve = np.max(fitness, axis=1)
    # Remove artifacts from fitness
    fitness_curve = fitness_curve[fitness_curve > 0]
    fitness_curve = np.append(
        fitness_curve,
        np.ones(params["gp_params"]["max_generations"] - len(fitness_curve)) * fitness_curve[-1],
    )
    if run in selected_runs:
        color = selected_colors[selected_runs.index(run)]
        color_shades = selected_color_shades[selected_runs.index(run)]
    else:
        color = "0.7"
    axes["gp_fitness"].plot(fitness_curve, color=color)

    if run in selected_runs:  # Plot stdp curves in panels E, F
        # Load symbolic expressions
        with open(fn, "r") as f:
            li = f.readlines()
            facilitate_expr = li[0].split(",")[-1]
            depress_expr = li[0].split(",")[-2]
            fitness = li[0].split(",")[1]
            fitness_list.append(float(fitness))
        if run in selected_runs:
            selected_lr.append((depress_expr, facilitate_expr))
            print(run, depress_expr, "|", facilitate_expr, fitness)
            synapse_params.update(
                {
                    "expr_depress": depress_expr,
                    "expr_facilitate": facilitate_expr,
                    "weight": weight,
                }
            )

        r = selected_runs.index(run)
        ax = axes_gp_lr_list[r]
        ax_eff = axes_gp_lr_eff_list[r]

        ax.set_xlabel(r"$t_{\mathrm{post}} - t_{\mathrm{pre}}$")
        ax_eff.set_xlabel(r"$t_{\mathrm{post}} - t_{\mathrm{pre}}$")
        ax.set_ylabel(r"$\Delta w_j$")
        ax_eff.set_ylabel(r"$\Delta \tilde w_j$")
        ax.yaxis.set_label_coords(-0.3, 0.5, transform=ax.transAxes)
        ax_eff.yaxis.set_label_coords(-0.3, 0.5, transform=ax_eff.transAxes)

        ax.hlines(0.0, deltat[0], deltat[-1], linestyles="dashed", lw=1.0, color="0.5")
        ax_eff.hlines(0.0, deltat[0], deltat[-1], linestyles="dashed", lw=1.0, color="0.5")
        w_range = [0.25, 0.5, 0.75]
        for i, w in enumerate(w_range):
            synapse_params["weight"] = w
            deltaw_ana = np.array(
                [
                    analytical_learning_rule(dt - synapse_params["delay"], synapse_params)[0]
                    for dt in delta_t_range
                ]
            )
            ax.plot(deltat, deltaw_ana, "-", ms=5, label=facilitate_expr, color=color_shades[i])

            deltaw_ana_eff = copy.copy(deltaw_ana)
            deltaw_ana_eff[delta_t_range < 0] += deltaw_ana[-1]
            deltaw_ana_eff[delta_t_range >= 0] += deltaw_ana[0]
            deltaw_ana_eff *= 0.5
            ax_eff.plot(delta_t_range, deltaw_ana_eff, ms=5, color=color_shades[i])

        ax.set_ylim((-0.03, 0.015))
        ax.set_yticks([-0.02, 0.0, 0.01])

        ax.set_xlim((-150.0, 150.0))
        ax_eff.set_xlim((-150.0, 150.0))

        ax_pos = ax_eff.get_position()
        if r == 0:
            y_offset = 0.35
            ax_eff.set_yticks([0.0, 0.005])
            ax_eff.set_ylim((-0.00125, 0.005))
        else:
            y_offset = 0.1
            ax_eff.set_yticks([-0.02, -0.01, 0.0])
        ax_cb = pl.axes(
            (
                ax_pos.x0 + 0.65 * (ax_pos.x1 - ax_pos.x0),
                ax_pos.y0 + y_offset * (ax_pos.y1 - ax_pos.y0),
                0.02,
                0.13,
            ),
            frame_on=False,
        )
        ax_cb.set_xticks([])
        ax_cb.set_yticks([])
        cmap2 = ListedColormap(color_shades)
        sm = pl.cm.ScalarMappable(cmap=cmap2, norm=pl.Normalize(vmin=0.1, vmax=0.9))
        sm.set_array([])
        cbticks = [0.25, 0.5, 0.75]
        cbar = pl.colorbar(sm, cax=ax_cb, ticks=cbticks, fraction=1.0)
        cbar.set_label(r"$w$")
        cbar.solids.set_edgecolor("none")
        cbar.ax.tick_params(length=0)

        # Load SNR validation values from file and plot into Panel D
        fitness_validation_list = np.load(
            os.path.join(data_directory, f"Fig5D_fitness_validation_run_{run}.npy")
        )
        ax_valid.errorbar(
            selected_runs.index(run),
            np.mean(fitness_validation_list),
            yerr=np.std(fitness_validation_list),
            ls="",
            marker="o",
            elinewidth=1.0,
            capsize=1.2,
            color=selected_colors[selected_runs.index(run)],
            markersize=4,
        )
ax_valid.yaxis.set_ticks_position("none")
ax_valid.xaxis.set_ticks_position("none")
ax_valid.spines["left"].set_color("none")
ax_valid.spines["right"].set_color("none")
ax_valid.spines["top"].set_color("none")
ymax = max(axes["gp_fitness"].get_ylim()[1], ax_valid.get_ylim()[1])
ax_valid.set_ylim((0.0, ymax))
ax_valid.set_xlim((-0.5, 2.5))
ax_valid.set_xticks([])
ax_valid.set_yticks([])

axes["gp_fitness"].set_ylim((0.0, ymax))

# Load SNR values for homeostatic synapse
SNR_homeostasis_list = np.load(os.path.join(data_directory, "Fig5D_SNR_homeostatic_synapse.npy"))
ax_valid.errorbar(
    2,
    np.mean(SNR_homeostasis_list),
    yerr=np.std(SNR_homeostasis_list),
    ls="",
    marker="o",
    elinewidth=1.0,
    capsize=1.2,
    color=orange,
    markersize=4,
)

axes["gp_fitness"].set_xlabel("Generation index")
axes["gp_fitness"].set_ylabel("Fitness")


"""
Save figure
"""
pl.savefig("Fig5_corr_learning_mpl.eps")


"""
Merge in sketch figure
"""
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0.5, 0.5, "Fig5_corr_learning_mpl.eps", width=17.6))
c.insert(pyx.epsfile.epsfile(1.4, 12.5, "Fig5_corr_learning_sketch.eps", width=5.7))

c.writeEPSfile("Fig5_corr_learning.eps")
