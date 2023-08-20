from scipy import stats
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt
from matplotlib import animation

from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter

plotter = Plotter()
target = Target(initial_x=0.0, initial_y=0.0, dt=0.1, simulation_duration=10,
                initial_vx=2, initial_vy=3, system_variance=5)
clutter = Clutter(dist_type='Normal', std=0.5)
pdaf = ProbabilisticDataAssociationFilter(
    initial_x=0.0, initial_y=0.0, initial_v_x=2, initial_v_y=3,
    dt=0.1, Pd=0.8)

plotter.set_axis(plot_title='Target Tracking')
log_state = []
saved_clutter = []
validated_measurements = []

# Generate random noise
noise = stats.norm.rvs(1, 0.1, size=(len(target.time_vector), len(target.time_vector)))
for i, time in enumerate(target.time_vector):
    [x_true, y_true, curr_time] = target.get_state(time)
    cluster = clutter.generate_clutter((x_true, y_true))
    cluster.add((x_true + noise[i][0], y_true + noise[i][1]))  # Add noise to true measurements
    pdaf.predict()
    if i % 10 == 0:
        log_state.append(pdaf.state[0:2])
        saved_clutter.append(cluster)
        validated_measurements.append(pdaf.update(cluster))

x_vec, y_vec = target.x_trajectory, target.y_trajectory


# plotter.plot_true_values((x_vec, y_vec))
# if True:
#     for i, clutter_to_plot in enumerate(saved_clutter):
#         if clutter_to_plot not in validated_measurements:
#             if i != 1:
#                 plotter.plot_clutter(clutter_to_plot, **{'label': ''})
#             else:
#                 plotter.plot_clutter(clutter_to_plot)
#     for i, measurements_to_plot in enumerate(validated_measurements):
#         if measurements_to_plot and i != 1:
#             plotter.plot_measurements(measurements_to_plot, **{'label': ''})
#         elif measurements_to_plot:
#             plotter.plot_measurements(measurements_to_plot)
#     plotter.plot_measurements(log_state, **{'color': 'm', 's': 15, 'label': 'PDAF'})
#     plotter.add_grid()
#     plotter.add_labels()
def animate(j, idx=0):

    # plotter = Plotter()
    plotter.set_axis(plot_title='Target Tracking')
    plotter.ax.set_xlim(-20, 50)  # Set your desired x-axis limits
    plotter.ax.set_ylim(-20, 50)
    if j % 10 == 0:
        # plt.cla()
        idx = j//10
        # plotter.plot_clutter(saved_clutter[idx],  **{'label': ''})
        # plotter.plot_measurements(validated_measurements[idx],  **{'label': ''})
        plotter.plot_measurements(log_state[idx], **{'color': 'm', 's': 15, 'label': ''})
    plotter.plot_true_values((x_vec[idx:j], y_vec[idx:j]), **{'label': ''})
    plotter.add_grid()
    plotter.add_labels(False)


anim = animation.FuncAnimation(plotter.fig, animate,
                               frames=len(target.time_vector), interval=100,
                               )
#pause
anim.event_source.stop()

#unpause
anim.event_source.start()
plt.show()
