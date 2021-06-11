import sys
import time
from os import path as osp

sys.path = [osp.dirname(osp.dirname(osp.abspath(__file__)))] + sys.path
from gpu_use.monitor import node_monitor

if __name__ == "__main__":
    while True:
        node_monitor()
        time.sleep(30)
