import traci
import sumolib
import time

CONFIG_PATH = r"C:\Traffic_project\SUMO_RL\Hello_SUMO\hello_1.sumocfg"

sumo_binary = sumolib.checkBinary("sumo-gui")

sumo_cmd = [sumo_binary, "-c", CONFIG_PATH, "--start"]

traci.start(sumo_cmd)

while traci.simulation.getMinExpectedNumber() > 0: 
    traci.simulationStep()
    #time.sleep(0.01)
traci.close()