from DataGeneration.Target import Target
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt


target = Target(initial_x=0.0, initial_y=0.0, dt=0.1, simulation_duration=10,
                velocity_x=2, velocity_y=3)
x_vec, y_vec = target.x_trajectory, target.y_trajectory
# print(x_vec)
# print(y_vec)
print(target.get_state(5.33))
plotter = Plotter()
plotter.set_axis(plot_title='Target Tracking')

plotter.plot_true_values((x_vec, y_vec))
plotter.add_grid()
plotter.add_labels()
plt.show()


