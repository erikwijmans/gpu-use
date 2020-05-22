#!/usr/bin/python

import os
import os.path as osp
import shlex
import subprocess
import sys
import xml.etree.ElementTree as etree

from gpu_use.db.db_schema import GPU, Node, Process, SLURMJob
from gpu_use.db.session import SessionMaker


def getLineage(pid):
    ancestors = [pid]
    ppid = pid
    while not ppid == 1:
        ppid = int(
            subprocess.check_output(shlex.split(f"ps -p {ppid} -oppid="))
            .decode("utf-8")
            .strip()
        )
        ancestors.append(ppid)
    return ancestors


gpu_command = "nvidia-smi -q -x"
pid_command = "ps -o %p,:,%u,:,%t,:,%a,:,%P --no-header -p "
listpids_command = "scontrol listpids"
environ_file = "/proc/{}/environ"
print_environ_file_command = "xargs --null --max-args=1 echo < {}"


def node_monitor():
    # Init container variables
    gpu_info = {}

    # Collect information about system health overall
    hostname = os.uname()[1]

    # Process info containers
    gpu2pid_info = {}
    pid2job_info = {}
    pid2user_info = {}

    pids = []
    smi_out = subprocess.check_output(shlex.split(gpu_command)).decode("utf-8")
    gpu_xml = etree.fromstring(smi_out)  # nvidia-smi
    for gpu in gpu_xml.findall("gpu"):
        gpu_id = int(gpu.find("minor_number").text)
        gpu2pid_info[gpu_id] = [
            int(y.find("pid").text)
            for y in gpu.find("processes").findall("process_info")
        ]
        pids.extend(gpu2pid_info[gpu_id])

    # Get jobid to pid mappings
    slurm_pids = (
        subprocess.check_output(shlex.split(listpids_command)).decode("utf-8").strip()
    )  # job -> pid mappings
    slurm_pids = [info.split() for info in slurm_pids.split("\n")[1:]]
    slurm_pids = [dict(pid=int(info[0]), jid=int(info[1])) for info in slurm_pids]
    slurm_pids = list(
        filter(lambda info: osp.exists(environ_file.format(info["pid"])), slurm_pids)
    )

    # Get process info including who is running it
    pid_list = ",".join(
        [str(y) for y in pids] + [str(info["pid"]) for info in slurm_pids]
    )
    ps_info = (
        subprocess.check_output(
            shlex.split(pid_command + pid_list), stderr=subprocess.STDOUT
        )
        .decode("utf-8")
        .strip()
    )

    for info_line in ps_info.split("\n"):
        info = info_line.split(",:,")
        pid2user_info[int(info[0])] = info[1:]

    gpu2job_info = dict()
    for info in slurm_pids:
        pid = info["pid"]
        jid = info["jid"]
        pid2job_info[pid] = jid

        if osp.exists(environ_file.format(pid)):
            p_environ = (
                subprocess.check_output(
                    shlex.split(
                        print_environ_file_command.format(environ_file.format_map(pid))
                    )
                )
                .decode("utf-8")
                .split("\n")
            )
            cuda_devices = list(
                filter(lambda var: var.startswith("CUDA_VISIBLE_DEVICES="), p_environ)
            )
            if len(cuda_devices) == 0:
                continue

            cuda_devices = cuda_devices[0].split("=")[1]
            if cuda_devices == "NoDevFiles":
                continue

            cuda_devices = cuda_devices[0].split(",")
            for gpu_id in cuda_devices:
                gpu_id = int(gpu_id)
                gpu2job_info[gpu_id] = dict(jid=jid, user=pid2user_info[pid][0])

    node = Node(name=hostname)
    node.load = "{} / {} / {}".format(*os.getloadavg())

    processes = []
    gpus = []
    jobs = []

    for gpu_id in sorted(gpu2pid_info.keys()):
        gpu = GPU(id=gpu_id, node=node)
        gpus.append(gpu)

        if gpu_id in gpu2job_info:
            job = SLURMJob(
                job_id=gpu2job_info[gpu_id]["jid"],
                node=node,
                user=gpu2job_info[gpu_id]["user"],
            )
            jobs.append(job)
            gpu.slurm_job = job

        for pid in gpu2pid_info[gpu_id]:
            ancestors = getLineage(pid)
            job_ids = list(
                set([pid2job_info[i] for i in ancestors if i in pid2job_info.keys()])
            )

            if len(job_ids) >= 1:
                slurm_job = gpu.slurm_job
            else:
                slurm_job = None

            user = pid2user_info[pid][0]
            cmnd = pid2user_info[pid][2][:75]

            processes.append(
                Process(
                    id=pid,
                    gpu=gpu,
                    node=node,
                    user=user,
                    command=cmnd,
                    slurm_job=slurm_job,
                )
            )

    session = SessionMaker()
    existing_node = session.query(Node).filter_by(name=hostname).first()
    if existing_node is not None:
        for gpu in existing_node.gpus:
            session.delete(gpu)
        for proc in existing_node.processes:
            session.delete(proc)
        for slurm_job in existing_node.slurm_jobs:
            session.delete(slurm_job)

        session.delete(existing_node)

    session.add_all([node] + processes + gpus + jobs)
    session.commit()


if __name__ == "__main__":
    node_monitor()
    node_monitor()
