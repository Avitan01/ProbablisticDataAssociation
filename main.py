import subprocess


if __name__ == '__main__':
    simulation = input('Chose a simulation to run:\n'
                       'basic\nlocation initial\nvelocity initial\n'
                       'Pd\nPg\nmonte carlo\nmeasurement noise\n'
                       'distribution\n')
    match simulation.lower().strip():
        case 'basic':
            simulation_name = 'basic'
        case 'location initial':
            simulation_name = 'location_initial_condition'
        case 'velocity initial':
            simulation_name = 'velocity_initial_condition'
        case 'pd':
            simulation_name = 'Pd'
        case 'pg':
            simulation_name = 'Pg'
        case 'monte carlo':
            simulation_name = 'monte_carlo'
        case 'measurement noise':
            simulation_name = 'measurement_noise'
        case 'distribution':
            simulation_name = 'distribution'
        case _:
            simulation_name = 'basic'
    print(f'{simulation_name} was chosen to run\n')
    subprocess.call(['python', f'Simulations\\{simulation_name}_simulation.py'])
