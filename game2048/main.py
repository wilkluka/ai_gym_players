import shutil

from simulation.simulation import Simulation
import subprocess


if __name__ == "__main__":
    subprocess.call('paplay /usr/share/sounds/freedesktop/stereo/trash-empty.oga'.split())
    try:
        shutil.rmtree('./logs/')
        print("previous logs removed")
    except FileNotFoundError:
        print("nothing to remove")
    simulation = Simulation(expected_rounds=5, history_limit=10)
    simulation.run()
