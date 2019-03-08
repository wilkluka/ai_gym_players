from simulation.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation(expected_rounds=1, history_limit=1)
    simulation.run()
