from DataGeneration.Target import Target
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt


target = Target(initial_x=0.0, initial_y=0.0, steps=100, simulation_duration=10,
                velocity_x=2, velocity_y=3)
target.initiate()
x_vec, y_vec = target.entire_x, target.entire_y
print(x_vec)
print(y_vec)

plotter = Plotter()
plotter.set_axis(plot_title='Target Tracking')

plotter.plot_true_values((x_vec, y_vec))
plotter.add_grid()
plotter.add_labels()
plt.show()


