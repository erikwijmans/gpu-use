import os
import time

from daemon.runner import DaemonRunner

from gpu_use.monitor.monitor import node_monitor


class DaemonRunnerPy3(DaemonRunner):
    def _open_streams_from_app_stream_paths(self, app):
        """ Open the `daemon_context` streams from the paths specified.

            :param app: The application instance.

            Open the `daemon_context` standard streams (`stdin`,
            `stdout`, `stderr`) as stream objects of the appropriate
            types, from each of the corresponding filesystem paths
            from the `app`.
            """
        self.daemon_context.stdin = open(app.stdin_path, "rt")
        self.daemon_context.stdout = open(app.stdout_path, "w+t")
        self.daemon_context.stderr = open(app.stderr_path, "w+t")


class MonitorDaemon:
    pidfile_path = "/var/run/gpu_used.pid"
    pidfile_timeout = 5

    stdin_path = "/dev/null"
    stdout_path = "/var/log/gpu-used/gpu-used.log"
    stderr_path = "/var/log/gpu-used/gpu-used.log"

    def __init__(self):
        pass
        #  os.makedirs(os.path.dirname(self.stdout_path), exist_ok=True)

    def run(self):
        while True:
            node_monitor()
            time.sleep(30)


def run_daemon():
    app = MonitorDaemon()
    runner = DaemonRunnerPy3(app)
    runner.do_action()


if __name__ == "__main__":
    run_daemon()
