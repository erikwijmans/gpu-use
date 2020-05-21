#!/usr/bin/python

import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as etree

from sqlalchemy.orm import sessionmaker

from gpu_use.db.db_schema import GPU, Node, Process
from gpu_use.db.engine import engine


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
    smi_out = subprocess.Popen(
        [gpu_command], shell=True, stdout=subprocess.PIPE
    ).stdout.read()
    gpu_xml = etree.fromstring(smi_out)  # nvidia-smi
    for gpu in gpu_xml.findall("gpu"):
        gpu_id = int(gpu.find("minor_number").text)
        gpu2pid_info[gpu_id] = [
            int(y.find("pid").text)
            for y in gpu.find("processes").findall("process_info")
        ]
        pids.extend(gpu2pid_info[gpu_id])

    # Get process info including who is running it
    pid_list = ",".join([str(y) for y in pids])
    ps_info = subprocess.check_output(
        shlex.split(pid_command + pid_list), stderr=subprocess.STDOUT
    ).decode("utf-8")

    for info_line in ps_info.readlines():
        info = info_line.split(",:,")
        pid2user_info[int(info[0])] = info[1:]

    # Get jobid to pid mappings
    slurm_pids = subprocess.check_output(shlex.split(listpids_command)).decode(
        "utf-8"
    )  # job -> pid mappings
    for info_line in slurm_pids.readlines()[1:]:
        info = info_line.split()
        pid2job_info[int(info[0])] = int(info[1])

    session = sessionmaker(bind=engine)

    node = Node(name=hostname)
    node.load = "{} / {} / {}".format(*os.getloadavg())

    processes = []
    gpus = []

    for gpu_id in gpu2pid_info.keys():
        gpu = GPU(id=gpu_id, node=node)

        for pid in gpu2pid_info[gpu_id]:
            ancestors = getLineage(pid)
            job = list(
                set([pid2job_info[i] for i in ancestors if i in pid2job_info.keys()])
            )
            if len(job) >= 1:
                job = job[0]
            else:
                job = None

            user = pid2user_info[pid][0]
            cmnd = pid2user_info[pid][2][:75]

            processes.append(
                Process(
                    id=pid, gpu=gpu, node=node, user=user, command=cmd, slurm_job_id=job
                )
            )

    existing_node = session.query(Node).filter_by(name=hostname).first()
    if existing_node is not None:
        session.delete(existing_node)

    session.add_all([node] + [processes] + [gpus])
    session.commit()


if __name__ == "__main__":
    node_monitor()
    node_monitor()
