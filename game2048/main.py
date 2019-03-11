from simulation.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation(expected_rounds=3, history_limit=4)
    simulation.run()
