import numpy as np
from Simulations.monte_carlo_simulation import monte_carlo
from Tools.Plotter import Plotter

sim_results = []
if __name__ == '__main__':
    param_dict = {
        'target': {
            'initial_x': 0.0,
            'initial_y': 0.0,
            'dt': 0.1,
            'simulation_duration': 100,
            'initial_vx': 0.1,
            'initial_vy': 6,
            'system_variance': 0.1
        },
        'clutter': {
            'dist_type': 'uniform',
            'std': 20,
            'clutter_size': 20,
        },
        'pdaf': {
            'number_of_state_variables': 4,
            #                          vx=0.1 vy=5
            'initial_state': (0.0, 0.0, 1.0, 2.0),
            # Increase cov - reduce state error
            'initial_covariance_magnitude': 10,
            'transition_matrix': np.array(
                [[1, 0, 0.1, 0],
                 [0, 1, 0, 0.1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            ),
            'Pd': 0.1,
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

    mean_vx = np.mean(results['vx state'], axis=0)
    mean_vy = np.mean(results['vy state'], axis=0)
    std_vx = np.sqrt(np.mean(results['vx var'], axis=0))
    std_vy = np.sqrt(np.mean(results['vy var'], axis=0))

    plotter = Plotter()
    plotter.add_subplot([2, 2])
    plotter.plot_data((results['time'], mean_x), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_x), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_x), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{x}$',
                     plot_title=f'Monte carlo simulation N={N} with Missmatch at X velocity')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})
    # plotter.show_plot()
    plotter.add_subplot([2, 2])
    # plotter = Plotter()
    plotter.plot_data((results['time'], mean_y), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_y), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_y), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{y}$',
                     plot_title=f'Monte carlo simulation N={N} with Missmatch at Y velocity')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 2])
    plotter.plot_data((results['time'], mean_vx), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_vx), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_vx), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vx}$',
                     plot_title=f'Monte carlo simulation N={N} with Missmatch at X velocity')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 2])
    plotter.plot_data((results['time'], mean_vy), **{'label': '$\mu$'})
    plotter.plot_data((results['time'], std_vy), **{'color': 'r', 'label': '$\mu$ + $\sigma$'})
    plotter.plot_data((results['time'], -std_vy), **{'color': 'r', 'label': '$\mu$ - $\sigma$'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vy}$',
                     plot_title=f'Monte carlo simulation N={N} with Missmatch at Y velocity')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.show_plot()
