from matplotlib import pyplot
import seaborn as sns
import matplotlib.pyplot as plt
from BackEnd.analyze_dimples import pd, np, cv2, findIntervals


def from_fig_to_array(fig: plt.Figure) -> np.ndarray:
    """

    :param fig: a matplotlib figure that we want to convert to an array
    :return: 3D array of the figure
    """
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = cv2.cvtColor(image_from_plot, cv2.COLOR_BGR2RGB)
    return image


def create_count_plot_and_array(ax: plt.Figure, data: pd.DataFrame, x: str, palette, x_label: str) -> np.ndarray:
    ax = sns.countplot(ax=ax, data=data, x=x, palette=palette)
    ax.set_ylabel('Count',
                  fontsize='xx-large')
    ax.set_xlabel(x_label,
                  fontsize='xx-large')
    interval_names = data['intervals'].unique()
    ax.set_xticks(range(len(interval_names)))
    ax.set_xticklabels(labels=interval_names, rotation=45, size=15)

    fig = ax.figure
    image_as_array = from_fig_to_array(fig)
    plt.close(fig)
    return image_as_array


def createBinPlot(image_analysis: dict, measurement_name: str, color_palette: str, x_label: str,
                  num_of_bins: int = 10) -> np.ndarray:
    """
    The function takes a dictionary of an image analysis that contains properties of a single image,
    and makes a histogram plot based on the depth intervals as the x axis and the frequency as the y-axis.


    :param image_analysis: a dictionary containing the properties we wanted to analyze in a prediction.
    :param measurement_name: string, could be ratio,local,global.
    :param color_palette: string, name of color palette from seaborn
    :param x_label: string, the label under the the x axis.
    :param num_of_bins: the number of intervals.

    :return: 3D numpy array, the plot is converted into a numpy array
    """
    df_depth = pd.DataFrame({
        measurement_name: image_analysis[measurement_name],
    }
    )
    df_depth['intervals'] = pd.cut(x=df_depth[measurement_name], bins=num_of_bins, right=False)

    df_depth = df_depth.sort_values('intervals')

    ax_dims = (15, 15)
    fig, ax = pyplot.subplots(figsize=ax_dims)
    pal = sns.color_palette(color_palette)
    return create_count_plot_and_array(ax, df_depth, "intervals", pal, x_label)


def createAreaHistPlot(image_analysis: dict, num_of_bins: int = 15) -> np.ndarray:
    """
    The function takes a dictionary of an image analysis that contains properties of a single image,
    and makes a histogram plot based on the area intervals as the x axis and the frequency as the y-axis.


    :param image_analysis: a dictionary containing the properties we wanted to analyze in a prediction.
    :param num_of_bins: the number of intervals.
    :return: 3D numpy array, the plot is converted into a numpy array

    """
    intervals = findIntervals(image_analysis["area"], num_of_bins=num_of_bins)
    # left_intervals  = [interval.left for interval in intervals]

    df = pd.DataFrame({
        "area": image_analysis["area"],
        "intervals": image_analysis["interval_range"]
    })
    ax_dims = (15, 15)
    fig, ax = pyplot.subplots(figsize=ax_dims)
    ax = sns.histplot(ax=ax, data=df, x="intervals", multiple="dodge",
                      shrink=0.5, stat="frequency", color="skyblue")
    ax.set_xlabel('Intervals of area',
                  fontsize='xx-large')
    ax.set_ylabel('Frequency',
                  fontsize='x-large')
    x = range(0, len(intervals))
    ax.set_xticks(x)
    ax.set_xticklabels(labels=intervals, rotation=45, size=15)
    ax.margins(0.2, 0.2)
    ax.autoscale(enable=True, axis="x", tight=True)
    fig.tight_layout()

    image_as_array = from_fig_to_array(fig)
    plt.close(fig)
    return image_as_array
