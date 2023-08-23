# Todo: Changing initial conditions.
#       Getting state estimation error
#       Covariance size(diagonal)
#       Pg and Pd change separately

# Parameters to return in monte carlo:
#   - state estimation
#   - covariance
#   - STD
#   - Estimation of each variable with +- std(sqrt(P))
from scipy import stats
import numpy as np
import pandas as pd
from DataGeneration.Target import Target
from DataGeneration.Clutter import Clutter
from ProbabilisticDataAssociation.ProbablisticDataAssociationFilter import ProbabilisticDataAssociationFilter
from Tools.Plotter import Plotter
from matplotlib import pyplot as plt


def monte_carlo(N: int, kwargs_dict: dict) -> dict:
    """Run N simulations and return the results.
        Args:
            N(int): Number of simulations.
            kwargs_dict(dict): A dictionary with keys as function names and values as kwargs for the functions.
        Return:
             dict: A key-value dictionary with results."""
    logs = {}
    size_of_col = int(np.ceil(kwargs_dict['target']['simulation_duration'] / kwargs_dict['target']['dt']))
    x_state = np.zeros((N, size_of_col))
    y_state = np.zeros((N, size_of_col))
    x_variance = np.zeros((N, size_of_col))
    y_variance = np.zeros((N, size_of_col))
    for num in range(N):
        target = Target(**kwargs_dict['target'])
        clutter = Clutter(**kwargs_dict['clutter'])
        pdaf = ProbabilisticDataAssociationFilter(**kwargs_dict['pdaf'])
        noise = stats.norm.rvs(1, 0.5, size=(len(target.time_vector), len(target.time_vector)))
        for i, time in enumerate(target.time_vector):
            [x_true, y_true, _] = target.get_state(time)
            cluster = clutter.generate_clutter((x_true, y_true))
            cluster.add((x_true + noise[i][0], y_true + noise[i][1]))  # Add noise to true measurements
            # Predict
            pdaf.predict()
            # Measure every 2 seconds
            if i % 10 == 0:
                # Update
                pdaf.update(cluster)
            # Log
            x_state[num][i] = pdaf.state[0] - x_true
            y_state[num][i] = pdaf.state[1] - y_true
            x_variance[num][i] = pdaf.covariance[0][0]
            y_variance[num][i] = pdaf.covariance[1][1]

    # logs['state estimation error'][0] = np.mean(np.array([logs['state estimation error'][0], x_state]), axis=0))
    # logs['variance']
    logs['x state'] = x_state
    logs['y state'] = y_state
    logs['x var'] = x_variance
    logs['y var'] = y_variance
    logs['time'] = target.time_vector
    return logs


if __name__ == '__main__':
    kwargs_dict = {
        'target': {
            'initial_x': 0.0,
            'initial_y': 0.0,
            'dt': 0.1,
            'simulation_duration': 20,
            'initial_vx': 1,
            'initial_vy': 2,
            'system_variance': 0.1
        },
        'clutter': {
            'dist_type': 'uniform',
            'std': 20
        },
        'pdaf': {
            'number_of_state_variables': 4,
            'initial_state': (0.0, 0.0, 1.0, 2.0),
            'initial_covariance_magnitude': 10,
            'transition_matrix': np.array(
                [[1, 0, 0.1, 0],
                 [0, 1, 0, 0.1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            ),
            'Pd': 0.8,
            'Pg': 0.66,
            'observation_matrix': np.array(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0]]
            ),
            'number_of_measurement_variables': 2,
            'process_noise_gain': 0.01 ** 2,
            'measurement_noise_gain': 7 ** 2
        }
    }
    N = 2
    results = monte_carlo(N, kwargs_dict)
    mean_x = np.mean(results['x state'], axis=0)
    mean_y = np.mean(results['y state'], axis=0)
    std_x = np.sqrt(np.mean(results['x var'], axis=0))
    std_y = np.sqrt(np.mean(results['y var'], axis=0))

    plotter = Plotter()
    plotter.plot_data((results['time'], mean_x), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_x), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_x), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{x}$',
                     plot_title=f'Monte carlo simulation N={N} in x direction')
    plotter.add_grid()
    plotter.add_labels()
    plotter.show_plot()

    plotter = Plotter()
    plotter.plot_data((results['time'], mean_y), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_y), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_y), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{y}$',
                     plot_title=f'Monte carlo simulation N={N} in y direction')
    plotter.add_grid()
    plotter.add_labels()
    plotter.show_plot()
