import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_style():
    # namespace for plotting styles:
    class Ns:
        pass

    Ns.default_color = 'black'

    # Line Styles
    Ns.lsSignal = dict(color='black', lw=2)
    Ns.lsStimulusTrace = dict(color='tab:blue', lw=2)
    Ns.lsReg = dict(color='tab:green', lw=2)

    Ns.lsDotted = dict(color='tab:blue', lw=0.5, linestyle='dotted')

    # Face Style
    Ns.fsDefault = dict(alpha=0.25, facecolor='tab:green')

    # Text Styles
    Ns.txtScaleBar = dict(color='black', fontsize=8)
    Ns.txtScaleBarRef = dict(color='white', fontsize=8)
    Ns.txtAnnotationRef = dict(color='white', fontsize=14)
    Ns.txtAnnotation = dict(color='black', fontsize=8)
    Ns.txtHeader = dict(fontsize=11)

    # Color Bar Styles
    Ns.ColorBarText = dict(color='Black', fontsize=8)
    Ns.ColorBar_text_size = 8
    Ns.ColorBar_label_pad = 10
    Ns.ColorBar_label_rotation = 270

    # global settings:
    plt.rcParams['image.cmap'] = 'jet'

    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    # Ticks:
    plt.rcParams['xtick.major.pad'] = '2'
    plt.rcParams['ytick.major.pad'] = '2'
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    # plt.rcParams['xtick.labelsize'] = 'small'
    # plt.rcParams['ytick.labelsize'] = 'small'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    # Title Size:
    plt.rcParams['axes.titlesize'] = 10

    # Axes Label Size:
    plt.rcParams['axes.labelsize'] = 10
    Ns.xlabel = dict(fontsize=10)  # Manually positioned labels (e.g. common labels) via ax.text()

    # Axes Line Width:
    plt.rcParams['axes.linewidth'] = 1

    # Line Width:
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.color'] = 'k'

    # Marker Size:
    plt.rcParams['lines.markersize'] = 2

    # Error Bars:
    plt.rcParams['errorbar.capsize'] = 0

    # Legend Font Size:
    plt.rcParams['legend.fontsize'] = 8
    Ns.txtLegend = dict(fontsize=8)

    # Set pcolor shading
    plt.rcParams['pcolor.shading'] = 'auto'

    return Ns
