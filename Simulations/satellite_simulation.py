from scipy import stats

from DataGeneration.Satellite import Satellite
from DataGeneration.SpaceClutter import SpaceClutter
from Tools.Plotter import Plotter
import matplotlib.pyplot as plt
import numpy as np

from ProbabilisticDataAssociation.ProbabilisticDataAssociationFilter import ProbabilisticDataAssociationFilter

if __name__ == '__main__':
    plotter = Plotter()
    plot_type = {'static': True,
                 'animate': False
                 }
    # Static plotting
    to_plot_or_not_to_plot = {
        'true values': False,
        'estimated values': True,
        'validated measurements': False,
        'clutter': False,
        'updated estimate': False,
        'covariance': False,
        'earth': True
    }

    plotter.set_axis(plot_title='Satellite Tracking simulation',
                     x_label='x [km]', y_label='y [km]')
    dt = 1  # [s]
    satellite = Satellite(
        initial_r=1500,
        initial_theta=90,
        dt=dt,
        orbits=1,
        system_variance=1
    )

    clutter = SpaceClutter(
        view_angle=[60, 120],
        LEO_mean=2000,
        GEO_mean=36000
    )

    # # Define PDAF parameters
    state_size = 4
    #                 r  , theta, r dot, theta dot
    initial_state = ((1500 + 6378) / 1000, np.deg2rad(90), 0.0, (2 * np.pi) / (128 * 60))
    initial_covariance_magnitude = 100
    transition_matrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    Pd = 0.999  # Probability for detection
    Pg = 0.001  # Factor for probability
    observation_size = 2
    observation_matrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]]
    )

    process_noise_gain = 1 ** 2
    measurement_noise_gain = 10 ** 2

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
    MSE = []
    # Generate random noise
    noise = stats.norm.rvs(1, 5, size=(len(satellite.time_vector), len(satellite.time_vector)))
    # Start simulation
    for i, time in enumerate(satellite.time_vector):
        [r_true, theta_true, _, _, curr_time] = satellite.get_state_radial(time)
        # Predict
        temp_log = np.zeros((2, 1))
        temp_log[0] = 1000 * pdaf.state[0]
        temp_log[1] = pdaf.state[1]
        log_state.append(temp_log)
        log_cov.append(pdaf.covariance)
        pdaf.predict()
        # if i % 50 == 0:

        if np.deg2rad(1200000) > theta_true > np.deg2rad(1000):
            cluster_x_y, cluster_radial = clutter.generate_clutter()
            cluster_radial.add((r_true + noise[0][i], theta_true))  # Add noise to true measurements
            # Update
            cluster_radial_normilaize = {(r /1000, theta) for r, theta in cluster_radial}
            validated = pdaf.update(cluster_radial_normilaize)
            temp_log = np.zeros((2, 1))
            temp_log[0] = 1000 * pdaf.state[0]
            temp_log[1] = pdaf.state[1]
            updated_pdaf.append(temp_log)
            saved_clutter.append(cluster_x_y)
            validated_measurements.append({(1000*radius * np.cos(angle), 1000*radius * np.sin(angle)) for radius, angle in validated})
            print(f'update time: {time}')
        MSE.append(np.sqrt((r_true / 1000 - pdaf.state[0]) ** 2 + (theta_true - pdaf.state[0]) ** 2))
    # # End simulation

    if plot_type['static']:
        if to_plot_or_not_to_plot['true values']:
            plotter.plot_true_values((satellite.x_trajectory, satellite.y_trajectory), **{'markersize': 2})
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
            r_points, theta_points = zip(*log_state)
            x_points = r_points * (np.cos(theta_points))
            y_points = r_points * (np.sin(theta_points))
            plotter.plot_measurements((x_points, y_points), **{'color': 'm'})
            # plotter.plot_data((x_points, y_points), **{'color': 'm', 'markersize': 1, 'label': 'PDAF'})
        if to_plot_or_not_to_plot['updated estimate']:
            r_points, theta_points = zip(*updated_pdaf)
            x_points = r_points * (np.cos(theta_points))
            y_points = r_points * (np.sin(theta_points))

            plotter.plot_measurements((x_points, y_points), **{'color': 'orange',
                                                       's': 20,
                                                       'label': 'Updating PDAF',
                                                       'marker': '+'})
        if to_plot_or_not_to_plot['earth']:
            plotter.plot_earth()

    plotter.set_limits()
    plotter.add_grid()
    plotter.add_labels()

    # # Plotting
    x_vec, y_vec = satellite.x_trajectory, satellite.y_trajectory
    # # Animate simulation
    r_points, theta_points = zip(*log_state)
    x_points = r_points * (np.cos(theta_points))
    y_points = r_points * (np.sin(theta_points))
    data_dict = {
        'true values': (x_vec, y_vec),
        # 'measurements': validated_measurements,
        'estimated value': (x_points, y_points),
        # 'clutter': saved_clutter,
        'radial': theta_points,
    }

    if plot_type['animate']:
        plotter.animate_satellite(len(satellite.time_vector), data_dict)
    ploting = Plotter()
    ploting.plot_data((satellite.time_vector, MSE))
    plt.show()
