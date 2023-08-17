import matplotlib.pyplot as plt
import matplotlib
import addcopyfighandler
import numpy as np

matplotlib.use('TkAgg')


class Plotter:
    """Create plots for different use cases"""

    # Todo: add a limit function for adjusting the limits
    def __init__(self, **kwargs):
        fig_args = {'figsize': (8, 5)}
        fig_args.update(kwargs)
        self.dim = 2
        self.fig = plt.figure(**fig_args)
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.draw()

    # General actions
    def set_axis(self, x_label="$x$", y_label="$y$", plot_title=None):
        self.ax.axis('equal')
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(plot_title)

    def plot_data(self, data: tuple, scatter: bool = False, **kwargs) -> None:
        """Plot 2D data, choose between plot and scatter.
            Args:
                data(tuple): A tuple containing the x,y  vectors for the plot.
                scatter(bool): If true use scatter otherwise use plot"""
        plot_kwargs = dict(marker='o', color='b')
        plot_kwargs.update(kwargs)
        if scatter:
            if isinstance(data, set):
                x_points, y_points = zip(*data)
                self.ax.scatter(x_points, y_points, **kwargs)
            elif isinstance(data, tuple):
                self.ax.scatter(*data, **kwargs)
        else:
            self.ax.plot(*data, **kwargs)

    def add_labels(self, labels: list = None) -> None:
        """Add or show labels in a plot
            Args:
                labels(list): All labels to present, if None will call
                 the leaned function to present labels attached to the plots"""
        if labels:
            self.ax.legend(labels=labels)
        else:
            self.ax.legend()

    def add_grid(self, activate=True) -> None:
        """Show or hide the grid, default is to show"""
        self.ax.grid(activate)

    # Specific plots
    def plot_true_values(self, true_values: tuple, **kwargs) -> None:
        """Plot a all true_values in a 2D plot
            Args:
                true_values(tuple): Containing the x,y vectors to plot"""
        true_values_kwargs = {'marker': 'o',
                              'color': 'b',
                              'label': 'Measurements'}
        true_values_kwargs.update(kwargs)
        self.plot_data(true_values, **true_values_kwargs)

    def plot_measurements(self, measurements: tuple, **kwargs) -> None:
        """Plot a all measurements in a 2D plot
            Args:
                measurements(tuple): Containing the x,y vectors to plot"""
        measurements_kwargs = {'marker': 'o',
                               'color': 'r',
                               'label': 'Measurements'}
        measurements_kwargs.update(kwargs)
        self.plot_data(measurements, scatter=True, **measurements_kwargs)

    def plot_clutter(self, clutter: tuple|set, **kwargs) -> None:
        """Plot the clutter as a set of markers in a 2D plot
            Args:
                clutter(tuple): Containing the x,y vector."""
        clutter_kwargs = {'marker': 'v',
                          'color': 'y',
                          'label': 'Clutter'}
        clutter_kwargs.update(kwargs)
        self.plot_data(clutter, scatter=True, **clutter_kwargs)

    def plot_covariance(self, covariance: tuple, **kwargs) -> None:
        """Plot the covariance as a dashed line
            Args:
                covariance(tuple): Containing the x,y vector."""
        covariance_kwargs = {'linestyle': '--',
                             'color': 'r',
                             'label': '$\sigma$'}
        covariance_kwargs.update(kwargs)
        self.plot_data(covariance, **covariance_kwargs)

    @staticmethod
    def show_plot():
        plt.show()


if __name__ == "__main__":
    # Example
    plotter = Plotter()
    plotter.set_axis(plot_title='Target Tracking')
    # Data Creation
    x = np.arange(50)
    y_true = x
    plotter.plot_true_values((x, y_true))
    y_measurements = x + np.random.standard_normal(size=x.size)
    plotter.plot_measurements((x, y_measurements))
    y_clutter = x + np.random.randint(-8, 8, size=x.size)
    plotter.plot_clutter((x, y_clutter))
    y_cov = x + 2
    y_cov_minus = x - 2
    plotter.plot_covariance((x, y_cov))
    plotter.plot_covariance((x, y_cov_minus), **{'label': '$-\sigma$'})
    plotter.add_grid()
    plotter.add_labels()
    plt.show()
