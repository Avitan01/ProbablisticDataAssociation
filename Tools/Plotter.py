import matplotlib.pyplot as plt
import matplotlib
import addcopyfighandler
import numpy as np

matplotlib.use('TkAgg')


class Plotter:

    def __init__(self, **kwargs):
        fig_args = {'figsize': (8, 5)}
        fig_args.update(kwargs)
        self.dim = 2
        self.fig = plt.figure(**fig_args)
        self.set_axis()
        plt.draw()

    def set_axis(self, x_label="$x$", y_label="$y$"):
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.axis('equal')
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title('hey')

    def plot_data(self, data: tuple, scatter: bool = False, **kwargs):
        plot_kwargs = dict(marker='o', color='b')
        plot_kwargs.update(kwargs)
        if scatter:
            self.ax.scatter(*data, **kwargs)
        else:
            self.ax.plot(*data, **kwargs)

    def plot_measurements(self, measurements):
        self.plot_data(measurements, **{'marker':'-o', 'color':'r'})

    def add_labels(self, labels: list):
        self.ax.legend(labels=labels)


if __name__ == "__main__":
    plotter = Plotter()
    x = np.arange(20)
    y = x + np.random.randint(1, 5, size=x.size)
    y2 = x + np.random.randint(1, 5, size=x.size)
    plotter.plot_measurements((x, y))
    plotter.plot_measurements((x, y2))

    plotter.add_labels(['line', 'scatter'])
    plt.show()
