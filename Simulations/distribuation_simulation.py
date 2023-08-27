import numpy as np
from Tools.Plotter import Plotter
from Simulations.monte_carlo_simulation import monte_carlo


if __name__ == '__main__':
    distributions = {'uniform': 0.1,
                     'normal': 0.1,
                     'log normal': 0.1,
                     'rayleigh': 0.1
                     }
    dist_info = {key: [] for key in distributions.keys()}
    for dist in distributions.keys():
        arguments_dict = {
            'target': {
                'initial_x': 0.0,
                'initial_y': 0.0,
                'dt': 0.1,
                'simulation_duration': 100,
                'initial_vx': 1,
                'initial_vy': 2,
                'system_variance': 0.01 ** 2
            },
            'clutter': {
                'dist_type': dist,
                'std': distributions[dist],
                'clutter_size': 20,
            },
            'pdaf': {
                'number_of_state_variables': 4,
                'initial_state': (0.0, 0.0, 1.0, 2.0),
                'initial_covariance_magnitude': 100,
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
        results = monte_carlo(N, arguments_dict)
        mean_x = np.mean(results['x state'], axis=0)
        mean_y = np.mean(results['y state'], axis=0)
        std_x = np.sqrt(np.mean(results['x var'], axis=0))
        std_y = np.sqrt(np.mean(results['y var'], axis=0))
        mean_vx = np.mean(results['vx state'] ** 2, axis=0)
        mean_vy = np.mean(results['vy state'] ** 2, axis=0)
        std_vx = np.sqrt(np.mean(results['vx var'], axis=0))
        std_vy = np.sqrt(np.mean(results['vy var'], axis=0))
        p_det = np.mean(results['p det'], axis=0)
        mse = np.mean(results['mse'], axis=0)
        dist_info[dist] = {
            'x state': mean_x,
            'y state': mean_y,
            'vx state': mean_vx,
            'vy state': std_x,#p_det,
            'mse': mse,
            'p det': p_det,
        }
    plotter = Plotter()
    plotter.add_subplot([2, 1])
    for dist in distributions.keys():
        plotter.plot_data((results['time'], dist_info[dist]['mse']), **{'label': f'{dist}'})
    plotter.set_axis(x_label='Time [s]', y_label='MSE',
                     plot_title=f'Monte carlo simulation N={N} MSE')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 1])
    for dist in distributions.keys():
        plotter.plot_data((results['time'], dist_info[dist]['p det']), **{'label': f'{dist}'})
    plotter.set_axis(x_label='Time [s]', y_label='det(P)',
                     plot_title=f'Monte carlo simulation N={N} det(P)')
    plotter.add_grid()
    plotter.set_limits(y_lim=[0, 0.5])
    plotter.add_labels(**{'loc': 'upper right'})
    # plotter.add_subplot([2, 2])
    # for dist in distributions.keys():
    #     plotter.plot_data((results['time'], dist_info[dist]['x state']), **{'label': f'{dist}'})
    # plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{x}$',
    #                  plot_title=f'Monte carlo simulation N={N} in x direction')
    # plotter.add_grid()
    # plotter.add_labels(**{'loc': 'upper right'})
    #
    # plotter.add_subplot([2, 2])
    # for dist in distributions.keys():
    #     plotter.plot_data((results['time'], dist_info[dist]['y state']), **{'label': dist})
    # plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{y}$',
    #                  plot_title=f'Monte carlo simulation N={N} in y direction')
    # plotter.add_grid()
    # plotter.add_labels(**{'loc': 'upper right'})
    #
    # plotter.add_subplot([2, 2])
    # for dist in distributions.keys():
    #     plotter.plot_data((results['time'], dist_info[dist]['vx state']), **{'label': dist})
    # plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vx}$',
    #                  plot_title=f'Monte carlo simulation N={N} in x direction')
    # plotter.add_grid()
    # plotter.add_labels(**{'loc': 'upper right'})
    #
    # plotter.add_subplot([2, 2])
    # for dist in distributions.keys():
    #     plotter.plot_data((results['time'], dist_info[dist]['vy state']), **{'label': dist})
    # plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vy}$',
    #                  plot_title=f'Monte carlo simulation N={N} in y direction')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})
    plotter.set_global_axis('Effect of clutter distribution on the state estimation error')
    plotter.show_plot()

