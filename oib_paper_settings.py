# oib_paper_settings
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# colors
my_gray = "#9b9b9b"
my_red = "#d54e00"
my_blue = "#5670e9"

my_palette = [my_red, my_blue]
my_cmap = LinearSegmentedColormap.from_list("my_cmap", my_palette)

# settings
font_settings = {
    "family": "serif",
    "size": 12}
fig_settings = {
    "figsize": (8.27,11.69/2.5)}
tick_settings = {
    "labelsize": 12}
axes_settings = {
    "grid": True}
grid_settings = {
    "alpha": 0.66}

# parse all
plt.rc("xtick", **tick_settings)
plt.rc("ytick", **tick_settings)
plt.rc("figure", **fig_settings)
plt.rc("font", **font_settings)
plt.rc("axes", **axes_settings)
plt.rc("grid", **grid_settings)
