from scipy import stats
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Ellipse
import numpy as np

from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter

if __name__ == '__main__':
    # Initiate parameters
    plotter = Plotter()
    dt = 0.1
    target = Target(
        initial_x=0.0, initial_y=0.0, dt=dt, simulation_duration=10,
        initial_vx=2, initial_vy=3, system_variance=10
    )
    clutter = Clutter(
        dist_type='uniform', std=0.5
    )
    # Define PDAF parameters
    state_size = 4
    #                 x    y   vx   vy
    initial_state = (0.0, 0.0, 2.0, 3.0)
    initial_covariance_magnitude = 10
    transition_matrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    Pd = 0.8  # Probability for detection
    Pg = 0.97  # Factor for probability
    observation_size = 2
    observation_matrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    )

    process_noise_gain = 0.01 ** 2
    measurement_noise_gain = 7 ** 2
    validation_size = 16  # AKA gamma

    pdaf = ProbabilisticDataAssociationFilter(
        state_size, initial_state, initial_covariance_magnitude,
        transition_matrix, Pd, Pg, observation_matrix, observation_size,
        process_noise_gain, measurement_noise_gain, validation_size
    )

    plotter.set_axis(plot_title='Target Tracking')
    log_state = []
    log_cov = []
    saved_clutter = []
    validated_measurements = []
    updated_pdaf = []

    # Generate random noise
    noise = stats.norm.rvs(1, 0.5, size=(len(target.time_vector), len(target.time_vector)))
    # Start simulation
    for i, time in enumerate(target.time_vector):
        [x_true, y_true, curr_time] = target.get_state(time)
        cluster = clutter.generate_clutter((x_true, y_true))
        cluster.add((x_true + noise[i][0], y_true + noise[i][1]))  # Add noise to true measurements
        # Predict
        pdaf.predict()
        log_state.append(pdaf.state[0:2])
        log_cov.append(np.linalg.eig(pdaf.covariance))
        # Measure every 2 seconds
        if i % 20 == 0:
            # Update
            validated = pdaf.update(cluster)
            updated_pdaf.append(pdaf.state[0:2])
            saved_clutter.append(cluster)
            validated_measurements.append(validated)
    # End simulation
    # Plotting
    x_vec, y_vec = target.x_trajectory, target.y_trajectory

    # plotme = Plotter()
    # + (x_vec[i] - log_state[i][0])
    # state_tilde = [y_vec[i] - log_state[i][1] for i in range(len(y_vec))]
    # plotme.plot_data((target.time_vector, log_cov), scatter=False)

    plot_type = {'static': True,
                 'animate': False
                 }
    # Static plotting
    to_plot_or_not_to_plot = {
        'true values': True,
        'estimated values': True,
        'validated measurements': True,
        'clutter': True,
        'updated estimate': True,
        'covariance': False
    }
    # Animate simulation
    data_dict = {
        'true values': (x_vec, y_vec),
        'measurements': validated_measurements,
        'estimated value': ([x for x, y in log_state], [y for x, y in log_state]),
        'clutter': saved_clutter,
    }  # 'updated estimate': updated_pdaf
    if plot_type['static']:
        if to_plot_or_not_to_plot['true values']:
            plotter.plot_true_values((x_vec, y_vec))
        if to_plot_or_not_to_plot['clutter']:
            for i, clutter_to_plot in enumerate(saved_clutter):
                if i != 1:
                    plotter.plot_clutter(clutter_to_plot, **{'label': ''})
                else:
                    plotter.plot_clutter(clutter_to_plot)
        if to_plot_or_not_to_plot['validated measurements']:
            for i, measurements_to_plot in enumerate(validated_measurements):
                if i != 1:
                    plotter.plot_measurements(measurements_to_plot, **{'label': ''})
                else:
                    plotter.plot_measurements(measurements_to_plot)
        if to_plot_or_not_to_plot['estimated values']:
            x_points, y_points = zip(*log_state)
            plotter.plot_data((x_points, y_points), **{'color': 'm', 'markersize': 1, 'label': 'PDAF'})
        if to_plot_or_not_to_plot['updated estimate']:
            plotter.plot_measurements(updated_pdaf, **{'color': 'orange',
                                                       's': 20,
                                                       'label': 'Updating PDAF',
                                                       'marker': '+'})
        if to_plot_or_not_to_plot['covariance']:
            # Choose a scale factor
            scale_factor = 1
            # for eigvalues in log_cov:
                # Plot the ellipse
            eigvals = log_cov[50][0]
            ell = Ellipse(xy=(x_points[50], y_points[50]), width=np.sqrt(eigvals[0]) , height=np.sqrt(eigvals[1]), angle=0,
                          color='r')
            ell.set_alpha(0.1)
            plt.gca().add_artist(ell)
            # Todo: understand the proper way to plot the cov

            # x_points, y_points = zip(*log_state)
            # x_cov = [x for x, y in log_cov]
            # y_cov = [y for x, y in log_cov]
            # x_point_cov = [point + cov for point, cov in zip(x_points, x_cov)]
            # y_point_cov = [point - cov for point, cov in zip(y_points, y_cov)]
            # x_point_cov_minus = [point - cov for point, cov in zip(x_points, x_cov)]
            # y_point_cov_minus = [point + cov for point, cov in zip(y_points, y_cov)]
            # plotter.plot_covariance((x_point_cov, y_point_cov))
            # plotter.plot_covariance((x_point_cov_minus, y_point_cov_minus))

        plotter.add_grid()
        plotter.add_labels()

    if plot_type['animate']:
        plotter.animate(len(target.time_vector), data_dict)
    plt.show()
