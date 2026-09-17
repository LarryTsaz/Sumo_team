import time
import traci
import sumolib

# =========================
# 1. SUMO 設定檔位置
# =========================
CONFIG_PATH = (
    r"C:\Traffic_project\SUMO_RL"
    r"\sumo-rl\sumo_rl\nets"
    r"\2way-single-intersection"
    r"\single-intersection.sumocfg"
)


# =========================
# 2. 啟動 SUMO
# =========================
sumo_binary = sumolib.checkBinary("sumo-gui")

sumo_cmd = [sumo_binary, "-c", CONFIG_PATH, "--start"]

traci.start(sumo_cmd)


# =========================
# 3. 讀取目前 state
# =========================
def get_state():

    # 每個方向的 incoming lanes
    incoming_lanes = {
        "north": ["n_t_0", "n_t_1"],
        "south": ["s_t_0", "s_t_1"],
        "east": ["e_t_0", "e_t_1"],
        "west": ["w_t_0", "w_t_1"],
    }

    state = {}

    # 計算每個方向的 queue
    for direction, lanes in incoming_lanes.items():

        total_halting = 0 # 計算排隊數

        for lane_id in lanes:

            halting_num = (traci.lane.getLastStepHaltingNumber(lane_id))

            total_halting += halting_num

        state[direction] = total_halting

    # 取得 traffic light phase
    tls_id = traci.trafficlight.getIDList()[0]

    phase = traci.trafficlight.getPhase(tls_id)

    state["phase"] = phase

    return state


# =========================
# 4. 印出 state
# =========================
def print_state(step, state):

    sim_time = traci.simulation.getTime()

    total_vehicles = len(traci.vehicle.getIDList())

    print(
        f"\nstep={step}, "
        f"time={sim_time}, "
        f"vehicles={total_vehicles}"
    )
    print(f"traffic light phase = {state['phase']}")

    print(f"north queue = {state['north']}")

    print(f"south queue = {state['south']}")

    print(f"east queue = {state['east']}")

    print(f"west queue = {state['west']}")

    print(f"n-s vehicles = {state['north']+state['south']}")

    print(f"e-w vehicles = {state['east']+state['west']}")
# =========================
# 5. 主程式
# =========================
def main():

    try:

        for step in range(1000):

            # SUMO 往前走一步
            traci.simulationStep()

            # 取得 state
            state = get_state()

            # 印出 state
            print_state(step, state)

            # 只是方便觀察 GUI
            time.sleep(0.05)

    finally:

        traci.close()

        print("\nSUMO / TraCI 已關閉")


# =========================
# 6. 執行
# =========================
if __name__ == "__main__":
    main()