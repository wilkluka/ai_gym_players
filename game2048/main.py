from simulation.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation(expected_rounds=2, history_limit=3)
    simulation.run()
