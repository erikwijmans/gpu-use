import time

from gpu_use import gpu_use_cli, node_monitor
from gpu_use.cli.cli import _show_dense
from gpu_use.db.schema import Node
from gpu_use.db.session import SessionMaker

if __name__ == "__main__":
    while True:
        node_monitor()
        session = SessionMaker()
        nodes = session.query(Node).order_by(Node.name).all()
        _show_dense(nodes, None, False)

        time.sleep(30)
