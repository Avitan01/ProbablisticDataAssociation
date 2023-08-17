import numpy as np
from scipy import stats
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt

from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter

plotter = Plotter()
target = Target(initial_x=0.0, initial_y=0.0, dt=0.1, simulation_duration=10,
                velocity_x=2, velocity_y=3)
clutter = Clutter(dist_type='Uniform', std=6)
pdaf = ProbabilisticDataAssociationFilter(
                initial_x=0.0, initial_y=0.0, initial_v_x=2, initial_v_y=3,
                dt=0.1, Pd=0.8)

plotter.set_axis(plot_title='Target Tracking')
log_state = []
saved_clutter = []
validated_measurements = []

# Generate random noise
noise = stats.norm.rvs(1, 100, size=(len(target.time_vector), len(target.time_vector)))
for i, time in enumerate(target.time_vector):
    [x_true, y_true, curr_time] = target.get_state(time)
    log_state.append(pdaf.state[0:2])
    cluster = clutter.generate_clutter((x_true, y_true))
    saved_clutter.append(cluster)
    cluster.add((x_true + noise[i][0], y_true + noise[i][1]))  # Add noise to true measurements
    pdaf.predict()
    if i % 10 == 0:
        print('Updating')
        validated_measurements.append(pdaf.update(cluster))
print(log_state)
x_vec, y_vec = target.x_trajectory, target.y_trajectory
plotter.plot_true_values((x_vec, y_vec))
plotter.plot_measurements(log_state, **{'color': 'm', 's': 10, 'label': 'PDAF'})
if False:
    for i, clutter_to_plot in enumerate(saved_clutter):
        if i != 1:
            plotter.plot_clutter(clutter_to_plot, **{'label': ''})
        else:
            plotter.plot_clutter(clutter_to_plot)
    for i, measurements_to_plot in enumerate(validated_measurements):
        if measurements_to_plot and i != 1:
            plotter.plot_measurements(measurements_to_plot, **{'label': ''})
        elif measurements_to_plot:
            plotter.plot_measurements(measurements_to_plot)
# pdaf.run_filter(target.time_vector)
# plotter.plot_true_values((x_vec, y_vec))
# plotter.plot_measurements(pdaf.state_log)
# print(pdaf.state_log)
# data = {(1, 2), (2, 3), (4, 4)}
# plotter.plot_clutter(data)


plotter.add_grid()
plotter.add_labels()
plt.show()
