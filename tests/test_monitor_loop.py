import time

from gpu_use import node_monitor

if __name__ == "__main__":
    while True:
        node_monitor()
        time.sleep(30)
