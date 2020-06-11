import pylab as pl
import seaborn as sns

from matplotlib import cm


sns.set()
color_palette = sns.color_palette()

blue, orange, green, red, purple, brown, pink, grey, gold, lightblue = color_palette

orange_light = tuple([min(1.0, 1.3 * o) for o in orange])

blue_dark = tuple([0.7 * b for b in blue])
blue_light = tuple([1.3 * b for b in blue])

red_dark = tuple([0.7 * r for r in red])
red_light = tuple([1.3 * r for r in red])

# colors for fitness plot in Fig. 3 and 4 (reward learning and error learning
fitness_colors = [pink, orange, green, red, purple, brown, blue, grey, gold, lightblue]

# synaptic_weight in Fig. 4
synaptic_weight_colors = [blue, orange, green, red, purple]
# colors for learning rules of correlation-based task
selected_colors = [blue, red]
selected_color_shades = [[blue_light, blue, blue_dark], [red_light, red, red_dark]]

# colormap for Figure 7
cmap = cm.get_cmap("viridis_r")

if __name__ == "__main__":
    sns.palplot(color_palette)
    pl.show()
