import numpy as np
from Tools.Plotter import Plotter
from Simulations.monte_carlo_simulation import monte_carlo


if __name__ == '__main__':
    Pg_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    Pg_info = {str(Pg): [] for Pg in Pg_range}
    for Pg in Pg_range:
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
                'dist_type': 'uniform',
                'std': 20,
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
                'Pg': Pg,
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
        Pg_info[str(Pg)] = {
            'x state': mean_x,
            'y state': mean_y,
            'vx state': mean_vx,
            'vy state': mean_vy
        }
    plotter = Plotter()
    plotter.add_subplot([2, 2])
    for Pg in Pg_range:
        plotter.plot_data((results['time'], Pg_info[str(Pg)]['x state']), **{'label': f'Pg={Pg}'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{x}$',
                     plot_title=f'Monte carlo simulation N={N} in x direction')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 2])
    for Pg in Pg_range:
        plotter.plot_data((results['time'], Pg_info[str(Pg)]['y state']), **{'label': f'Pg={Pg}'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{y}$',
                     plot_title=f'Monte carlo simulation N={N} in y direction')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 2])
    for Pg in Pg_range:
        plotter.plot_data((results['time'], Pg_info[str(Pg)]['vx state']), **{'label': f'Pg={Pg}'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vx}$',
                     plot_title=f'Monte carlo simulation N={N} in x direction')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})

    plotter.add_subplot([2, 2])
    for Pg in Pg_range:
        plotter.plot_data((results['time'], Pg_info[str(Pg)]['vy state']), **{'label': f'Pg={Pg}'})
    plotter.set_axis(x_label='Time [s]', y_label='$\\tilde{vy}$',
                     plot_title=f'Monte carlo simulation N={N} in y direction')
    plotter.add_grid()
    plotter.add_labels(**{'loc': 'upper right'})
    plotter.set_global_axis('Pg effect on the state estimation error')
    plotter.show_plot()

