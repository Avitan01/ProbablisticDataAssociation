import numpy as np
from Simulations.monte_carlo_simulation import monte_carlo


if __name__ == '__main__':
    Pd_vals = np.linspace(0.1,0.9,10)
    Pd_results = []
    for Pd in Pd_vals:
        param_dict = {
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
                'Pd': Pd,
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
        N = 1
        results = monte_carlo(N, param_dict)
        Pd_results.append(results)

