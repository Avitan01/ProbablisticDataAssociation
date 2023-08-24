import numpy as np
from Simulations.monte_carlo_simulation import monte_carlo
from Tools.Plotter import Plotter

sim_results =[]
if __name__ == '__main__':
    param_dict = {
        'target': {
            'initial_x': 0.0,
            'initial_y': 0.0,
            'dt': 0.1,
            'simulation_duration': 50,
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
            'initial_state': (0.0, 0.0, 0.1, 5.0),
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
    N = 100
    results = monte_carlo(N, param_dict)
    mean_x = np.mean(results['x state'], axis=0)
    mean_y = np.mean(results['y state'], axis=0)
    std_x = np.sqrt(np.mean(results['x var'], axis=0))
    std_y = np.sqrt(np.mean(results['y var'], axis=0))

    plotter = Plotter()
    plotter.add_subplot([2, 1])
    plotter.plot_data((results['time'], mean_x), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_x), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_x), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{x}$',
                     plot_title=f'Monte carlo simulation N={N} with Missmatch at x Location')
    plotter.add_grid()
    plotter.add_labels()
    # plotter.show_plot()
    plotter.add_subplot([2, 1])
    # plotter = Plotter()
    plotter.plot_data((results['time'], mean_y), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_y), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_y), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{y}$',
                     plot_title=f'Monte carlo simulation N={N} Missmatch at y Location')
    plotter.add_grid()
    plotter.add_labels()
    plotter.show_plot()
