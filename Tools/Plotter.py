import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import os
import addcopyfighandler
import numpy as np

matplotlib.use('TkAgg')


class Plotter:
    EARTH_RADIUS = 6378

    """Create plots for different use cases"""

    # Todo: add a limit function for adjusting the limits
    def __init__(self, **kwargs):
        fig_args = {'figsize': (8, 5)}
        fig_args.update(kwargs)
        self.dim = 2
        self.subplot_num = 1
        self.fig = plt.figure(**fig_args)
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.draw()

    def add_subplot(self, layout: list):
        if self.subplot_num == 1:
            self.ax.remove()
        self.ax = self.fig.add_subplot(*layout, self.subplot_num)
        self.subplot_num += 1
        plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.2)

    # General actions
    def set_global_axis(self, title: str):
        self.fig.suptitle(title)

    def set_axis(self, x_label="$x$", y_label="$y$", plot_title=None):
        # self.ax.axis('equal')
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
            if isinstance(data, (set, list)):
                x_points, y_points = zip(*data)
                self.ax.scatter(x_points, y_points, **kwargs)
            elif isinstance(data, tuple):
                self.ax.scatter(*data, **kwargs)
        else:
            self.ax.plot(*data, **kwargs)

    def add_labels(self, labels: list = None, **kwargs) -> None:
        """Add or show labels in a plot
            Args:
                labels(list): All labels to present, if None will call
                 the leaned function to present labels attached to the plots"""
        if labels:
            self.ax.legend(labels=labels, **kwargs)
        else:
            self.ax.legend(**kwargs)

    def add_grid(self, activate=True) -> None:
        """Show or hide the grid, default is to show"""
        self.ax.grid(activate)

    def set_limits(self, x_lim: list = None, y_lim: list = None):
        if x_lim:
            self.ax.set_xlim(x_lim)
        if y_lim:
            self.ax.set_ylim(y_lim)
        if not y_lim and not x_lim:
            self.ax.autoscale()

    # Specific plots
    def plot_true_values(self, true_values: tuple, **kwargs) -> None:
        """Plot a all true_values in a 2D plot
            Args:
                true_values(tuple): Containing the x,y vectors to plot"""
        true_values_kwargs = {'marker': 'o',
                              'color': 'b',
                              'label': 'Target',
                              'markersize': 1}
        true_values_kwargs.update(kwargs)
        self.plot_data(true_values, **true_values_kwargs)

    def plot_measurements(self, measurements: tuple | set | list, **kwargs) -> None:
        """Plot all measurements in a 2D plot
            Args:
                measurements: Containing the x,y vectors to plot"""
        measurements_kwargs = {'marker': 'o',
                               'color': 'c',
                               'label': 'Valid Measurements',
                               's': 5}
        measurements_kwargs.update(kwargs)
        self.plot_data(measurements, scatter=True, **measurements_kwargs)

    def plot_clutter(self, clutter: tuple | set, **kwargs) -> None:
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
        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        # Plot the ellipse
        plt.plot(eigenvectors[0] * eigenvalues[0], eigenvectors[1] * eigenvalues[1])

    def animate(self, frame_length: int, data_dict: dict, save: bool = False):
        anim = animation.FuncAnimation(self.fig, self.animate_plot, fargs=(data_dict,),
                                       frames=frame_length, interval=0.1,
                                       )
        # Save the animation as a GIF
        if save:
            anim.save('..\\Tools\\animation.gif', writer='pillow', fps=20)
        plt.show()

    def animate_satellite(self, frame_length: int, data_dict: dict):
        anim = animation.FuncAnimation(self.fig, self.animate_satellite_plot, fargs=(data_dict,),
                                       frames=frame_length, interval=0.1,
                                       )
        plt.show()

    def animate_plot(self, i: int, data_dict: dict):
        plt.cla()
        true_value, validated_measurement, estimated, clutter, updated_estimate = False, False, False, False, False
        self.set_axis(plot_title='Target Tracking')
        length = {}
        for key, value in data_dict.items():
            match key.lower():
                case 'true values':
                    true_value = True
                    x_vec, y_vec = value
                    length['true values'] = len(x_vec)
                case 'measurements':
                    validated_measurement = True
                    validated_measurements = value
                    length['measurements'] = len(validated_measurements)
                case 'estimated value':
                    estimated = True
                    estimated_state_x, estimated_state_y = value
                    length['estimated value'] = len(estimated_state_x)
                case 'clutter':
                    clutter = True
                    clutter_data = value
                    length['clutter'] = len(clutter_data)
                case 'updated estimate':
                    updated_estimate = True
                    updated_pdaf = value
                    length['updated estimate'] = len(updated_pdaf)
        max_length_val = max(length.values())
        if true_value:
            if i % (max_length_val // length['true values']) == 0:
                idx = i // (max_length_val // length['true values'])
                min_idx = 0 if idx < 20 else idx - 20
                self.plot_true_values((x_vec[min_idx:idx], y_vec[min_idx:idx]), **{'label': ''})
                self.ax.set_xlim(x_vec[idx] - 10, x_vec[idx] + 10)  # Set your desired x-axis limits
                self.ax.set_ylim(y_vec[idx] - 10, y_vec[idx] + 10)
        if clutter:
            # if i % (max_length_val // length['clutter']) == 0:
            idx = i // (max_length_val // length['clutter'])
            self.plot_clutter(clutter_data[idx], **{'label': ''})
        if validated_measurement:
            # if i % (max_length_val // length['measurements']) == 0:
            idx = i // (max_length_val // length['measurements'])
            self.plot_measurements(validated_measurements[idx], **{'label': ''})
        if estimated:
            if i % (max_length_val // length['estimated value']) == 0:
                idx = i // (max_length_val // length['estimated value'])
                min_idx = 0 if idx < 20 else idx - 20
                self.plot_data((estimated_state_x[min_idx:idx], estimated_state_y[min_idx:idx]),
                               **{'color': 'm', 'markersize': 1, 'label': ''})
        if updated_estimate:
            if i % (max_length_val // length['updated estimate']) == 0:
                idx = i // (max_length_val // length['updated estimate'])
                self.plot_measurements(updated_pdaf[idx], **{'color': 'orange',
                                                             's': 20,
                                                             'label': '',
                                                             'marker': '+'})

        self.add_grid()
        labels = [keys for keys in data_dict.keys()]
        idx = labels.index('measurements')
        if idx:
            estimated_string = labels.pop(idx)
            labels.append(estimated_string)
        idx = labels.index('estimated value')
        if idx:
            estimated_string = labels.pop(idx)
            labels.append(estimated_string)
        self.add_labels(labels)

    def animate_satellite_plot(self, i: int, data_dict: dict):
        true_value, validated_measurement, estimated, clutter, updated_estimate = False, False, False, False, False
        radial_estimate = False
        self.set_axis(plot_title='Satellite Tracking')
        self.ax.set_xlim(-10000, 10000)  # Set your desired x-axis limits
        self.ax.set_ylim(-10000, 10000)
        length = {}
        for key, value in data_dict.items():
            match key.lower():
                case 'true values':
                    true_value = True
                    x_vec, y_vec = value
                    length['true values'] = len(x_vec)
                case 'measurements':
                    validated_measurement = True
                    validated_measurements = value
                    length['measurements'] = len(validated_measurements)
                case 'estimated value':
                    estimated = True
                    estimated_state_x, estimated_state_y = value
                    length['estimated value'] = len(estimated_state_x)
                case 'clutter':
                    clutter = True
                    clutter_data = value
                    length['clutter'] = len(clutter_data)
                case 'updated estimate':
                    updated_estimate = True
                    updated_pdaf = value
                    length['updated estimate'] = len(updated_pdaf)
                case 'radial':
                    radial_estimate = True
                    radial = value
                    length['radial'] = len(radial)
        max_length_val = max(length.values())
        if true_value:
            if i % (max_length_val // length['true values']) == 0:
                idx = i // (max_length_val // length['true values'])
                self.plot_true_values((x_vec[0:idx], y_vec[0:idx]), **{'label': ''})
        if clutter:
            if i % (max_length_val // length['clutter']) == 0:
                idx = i // (max_length_val // length['clutter'])
                self.plot_clutter(clutter_data[idx], **{'label': ''})
        if validated_measurement:
            if i % (max_length_val // length['measurements']) == 0:
                idx = i // (max_length_val // length['measurements'])
                self.plot_measurements(validated_measurements[idx], **{'label': ''})
        if estimated:
            if i % (max_length_val // length['estimated value']) == 0:
                idx = i // (max_length_val // length['estimated value'])
                self.plot_data((estimated_state_x[0:idx], estimated_state_y[0:idx]),
                               **{'color': 'm', 'markersize': 1, 'label': ''})
        if updated_estimate:
            if i % (max_length_val // length['updated estimate']) == 0:
                idx = i // (max_length_val // length['updated estimate'])
                self.plot_measurements(updated_pdaf[idx], **{'color': 'orange',
                                                             's': 20,
                                                             'label': '',
                                                             'marker': '+'})

        if radial_estimate:
            idx = i // (max_length_val // length['radial'])
            if np.abs(radial[idx] - np.deg2rad(90)) < 0.1:
                plt.cla()
        self.add_grid()
        self.add_labels([keys for keys in data_dict.keys()])

    def plot_earth(self):
        image = plt.imread(
            '..\\Tools\\earth_from_space.png')
        image[image[:, :, 3] == 0] = [0, 0, 0, 0]  # set transparent
        # Get the current scatter plot
        extent = (-self.EARTH_RADIUS, self.EARTH_RADIUS, -self.EARTH_RADIUS, self.EARTH_RADIUS)
        self.ax.imshow(image, extent=extent)

    @staticmethod
    def show_plot():
        plt.show()


if __name__ == "__main__":
    from matplotlib.patches import Ellipse

    # Define the center and radii of the ellipse
    center = (0, 0)
    radii = (2, 1)
    # Create the ellipse patch
    ellipse = Ellipse(xy=center, width=radii[0], height=radii[1], angle=0)
    ellipse.set_edgecolor('black')
    ellipse.set_fill(False)

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
    plotter.ax.add_patch(ellipse)
    plt.show()
