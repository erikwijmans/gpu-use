#!/usr/bin/python

import datetime
import os
import os.path as osp
import shlex
import subprocess
import sys
import xml.etree.ElementTree as etree

import sqlalchemy as sa

from gpu_use.db.schema import GPU, GPUProcess, Node, SLURMJob
from gpu_use.db.session import SessionMaker

DEBUG_CHECK = "scontrol show job {} 2>/dev/null | grep \"Partition\" | awk -F'[ =]+' '{{print $3 == \"debug\"}}'"


def _is_debug_job(jid):
    try:
        return bool(
            int(
                subprocess.check_output(DEBUG_CHECK.format(jid), shell=True).decode(
                    "utf-8"
                )
            )
        )
    except ValueError:
        return False


def get_lineage(pid):
    ancestors = [pid]
    ppid = pid
    while not ppid == 1:
        try:
            ppid = int(
                subprocess.check_output(shlex.split(f"ps -p {ppid} -oppid="))
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            return None

        ancestors.append(ppid)
    return ancestors


gpu_command = "nvidia-smi -q -x"
pid_command = "ps -o %p,:,%u,:,%t,:,%a,:,%P --no-header -p "
listpids_command = "scontrol listpids"
environ_file = "/proc/{}/environ"
print_environ_file_command = "cat {} | xargs --null --max-args=1 echo"


def is_gpu_state_same_and_all_in_use(gpu2pid_info):
    hostname = os.uname()[1]
    session = SessionMaker()

    gpus = session.query(GPU).filter_by(node_name=hostname).order_by(GPU.id).all()
    for gpu_id in sorted(gpu2pid_info.keys()):
        if gpu_id >= len(gpus):
            return False

        gpu = gpus[gpu_id]
        known_pids = {proc.id for proc in gpu.processes}
        new_pids = set(gpu2pid_info[gpu_id])

        if known_pids != new_pids:
            return False

        if len(known_pids) == 0:
            return False

    return True


def node_monitor():
    # Collect information about system health overall
    hostname = os.uname()[1]

    session = SessionMaker()
    node = session.query(Node).filter_by(name=hostname).first()
    if node is None:
        node = Node(name=hostname)
        session.add(node)

    node.load = "{:.2f} / {:.2f} / {:.2f}".format(*os.getloadavg())
    node.update_time = datetime.datetime.now()
    session.commit()

    # Init container variables
    gpu_info = {}

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

    if is_gpu_state_same_and_all_in_use(gpu2pid_info):
        return

    # Get jobid to pid mappings
    try:
        slurm_pids = (
            subprocess.check_output(shlex.split(listpids_command))
            .decode("utf-8")
            .strip()
            .split("\n")[1:]
        )  # job -> pid mappings
    except subprocess.CalledProcessError:
        slurm_pids = []

    slurm_pids = [info.split() for info in slurm_pids]
    slurm_pids = [dict(pid=int(info[0]), jid=int(info[1])) for info in slurm_pids]
    slurm_pids = list(
        filter(lambda info: osp.exists(environ_file.format(info["pid"])), slurm_pids)
    )

    # Get process info including who is running it
    pid_list = ",".join(
        [str(y) for y in pids] + [str(info["pid"]) for info in slurm_pids]
    )
    if len(pid_list) > 0:
        ps_info = (
            subprocess.check_output(
                shlex.split(pid_command + pid_list), stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
    else:
        ps_info = []

    for info_line in ps_info:
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
                    print_environ_file_command.format(environ_file.format(pid)),
                    shell=True,
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

            cuda_devices = cuda_devices.split(",")
            for gpu_id in cuda_devices:
                gpu_id = int(gpu_id)
                gpu2job_info[gpu_id] = dict(
                    jid=jid, user=pid2user_info[pid][0].strip()[0:32]
                )

    existing_processes = {
        (proc.id, proc.node_name, proc.gpu_id): proc
        for gpu in node.gpus
        for proc in gpu.processes
    }
    existing_jobs = {job.job_id: job for job in node.slurm_jobs}

    def _add_job(gpu, jid, user):
        if jid not in existing_jobs:
            job = SLURMJob(
                job_id=jid, node=node, user=user, is_debug_job=_is_debug_job(jid)
            )

            session.add(job)
            existing_jobs[jid] = job

        gpu.slurm_job = existing_jobs[jid]
        return existing_jobs[jid]

    new_processes = []
    all_pids = set()

    for gpu_id in sorted(gpu2pid_info.keys()):
        if gpu_id >= len(node.gpus):
            gpu = GPU(id=gpu_id, node=node)
            session.add(gpu)
        else:
            gpu = node.gpus[gpu_id]

        if gpu_id in gpu2job_info:
            job_info = gpu2job_info[gpu_id]
            _add_job(gpu, job_info["jid"], job_info["user"])

        for pid in gpu2pid_info[gpu_id]:
            ancestors = get_lineage(pid)
            if ancestors is None:
                continue

            all_pids.add(pid)

            if (pid, hostname, gpu_id) in existing_processes:
                proc = existing_processes[(pid, hostname, gpu_id)]
            else:
                proc = GPUProcess(id=pid, gpu=gpu)
                new_processes.append(proc)

            job_ids = list(
                set([pid2job_info[i] for i in ancestors if i in pid2job_info.keys()])
            )

            user = pid2user_info[pid][0].strip()[0:32]

            if len(job_ids) >= 1:
                slurm_job = _add_job(gpu, job_ids[0], user)
            else:
                slurm_job = None

            cmnd = pid2user_info[pid][2][0:128]

            proc.user = user
            proc.command = cmnd
            proc.slurm_job = slurm_job

    session.add_all(new_processes)
    for proc in (
        session.query(GPUProcess)
        .filter(
            (GPUProcess.node_name == hostname) & sa.not_(GPUProcess.id.in_(all_pids))
        )
        .all()
    ):
        session.delete(proc)

    for job in (
        session.query(SLURMJob)
        .filter(
            (SLURMJob.node_name == hostname)
            & sa.not_(SLURMJob.job_id.in_(list(set(pid2job_info.values()))))
        )
        .all()
    ):
        session.delete(job)

    session.commit()
