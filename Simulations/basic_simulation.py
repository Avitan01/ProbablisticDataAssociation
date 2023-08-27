from scipy import stats
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter

if __name__ == '__main__':
    # Initiate parameters
    plotter = Plotter()
    plotter.set_axis(plot_title='Target Tracking basic simulation',
                     x_label='x [m]', y_label='y [m]')
    plot_type = {'static': True,
                 'animate': False
                 }
    # Static plotting
    to_plot_or_not_to_plot = {
        'true values': True,
        'estimated values': True,
        'validated measurements': True,
        'clutter': True,
        'updated estimate': False,
        'covariance': False
    }

    dt = 0.1
    target = Target(
        initial_x=0.0, initial_y=0.0, dt=dt, simulation_duration=100,
        initial_vx=1, initial_vy=2, system_variance=0.01 ** 2
    )
    clutter = Clutter(
        dist_type='log normal', std=1, clutter_size=2
    )
    # Define PDAF parameters
    state_size = 4
    #                 x    y   vx   vy
    initial_state = (0.0, 0.0, 1.0, 2.0)
    initial_covariance_magnitude = 10
    transition_matrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    Pd = 0.1  # Probability for detection
    Pg = 0.66  # Factor for probability
    observation_size = 2
    observation_matrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    )

    process_noise_gain = 0.01 ** 2
    measurement_noise_gain = 7 ** 2

    pdaf = ProbabilisticDataAssociationFilter(
        state_size, initial_state, initial_covariance_magnitude,
        transition_matrix, Pd, Pg, observation_matrix, observation_size,
        process_noise_gain, measurement_noise_gain
    )

    log_state = []
    log_cov = []
    saved_clutter = []
    validated_measurements = []
    updated_pdaf = []

    # Generate random noise
    noise = stats.norm.rvs(1, 0.5, size=(len(target.time_vector), len(target.time_vector)))
    # Start simulation
    for i, time in enumerate(target.time_vector):
        [x_true, y_true, _, _, curr_time] = target.get_state(time)
        cluster = clutter.generate_clutter((x_true, y_true))
        cluster.add((x_true + noise[i][0], y_true + noise[i][1]))  # Add noise to true measurements
        log_state.append(pdaf.state[0:2])
        log_cov.append(pdaf.covariance)
        # Predict
        pdaf.predict()
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
            # Calculate the eigenvalues and eigenvectors
            factor = len(log_state)//len(log_cov)
            for i, cov in enumerate(log_cov):
                # if i + 1 == len(log_state):
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # print(eigenvalues[0])
                ell = Ellipse(xy=(log_state[i*factor][0], log_state[i*factor][1]), width=np.sqrt(eigenvalues[0]), height=np.sqrt(eigenvalues[1]), angle=0,
                              color='r')
                ell.set_alpha(0.1)
                plt.gca().add_artist(ell)


            # Plot the ellipse
            # plt.plot(eigenvectors[0] * eigenvalues[0] + log_state[0][0],
            #          eigenvectors[1] * eigenvalues[1] + log_state[0][1])

            # Add labels and title
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.title("Ellipse Boundary")
            # plt.show()

            # plotter.plot_covariance(log_cov[0])
            # Choose a scale factor
            # scale_factor = 1
            # # for eigvalues in log_cov:
            #     # Plot the ellipse
            # eigvals = log_cov[50][0]
            # ell = Ellipse(xy=(x_points[50], y_points[50]), width=np.sqrt(eigvals[0]) , height=np.sqrt(eigvals[1]), angle=0,
            #               color='r')
            # ell.set_alpha(0.1)
            # plt.gca().add_artist(ell)
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
