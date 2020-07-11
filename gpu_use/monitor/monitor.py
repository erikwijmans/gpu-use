#!/usr/bin/python

import datetime
import logging
import os
import shlex
import subprocess
import sys
from os import path as osp
from xml.etree import ElementTree as etree

import sqlalchemy as sa

from gpu_use.db.schema import GPU, GPUProcess, Node, SLURMJob
from gpu_use.db.session import SessionMaker

DEBUG_CHECK = "scontrol show job {} 2>/dev/null | grep \"Partition\" | awk -F'[ =]+' '{{print $3 == \"debug\"}}'"

NODE_GPU_ORDER = {
    "ripl-s1": {
        smi_id: cuda_id for cuda_id, smi_id in enumerate([0, 1, 2, 4, 5, 6, 3, 7])
    },
    "vicki": {
        smi_id: cuda_id for cuda_id, smi_id in enumerate([0, 2, 3, 5, 6, 7, 1, 4])
    },
}


formatter = logging.Formatter(
    "[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    "%m-%d %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)

logger = logging.getLogger("gpu-used")
logger.addHandler(ch)
logger.setLevel(logging.INFO)


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


gpu_command = "timeout 5m nvidia-smi -q -x"
pid_command = "ps -o %p,:,%t,:,%a,:,%P --no-header -p "
user_command = "ps -o pid,user:32 --no-header -p "
listpids_command = "scontrol listpids"
environ_file = "/proc/{}/environ"
print_environ_file_command = "cat {} | xargs --null --max-args=1 echo"


def is_gpu_state_same_and_all_in_use(node: Node, gpu2pid_info):
    gpus = node.gpus
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
    logger.info("Monitor Start")

    session = SessionMaker()
    try:
        do_node_monitor(session)
    except UnicodeDecodeError as e:
        logger.error(str(e))
    except OSError as e:
        logger.error(str(e))
    except subprocess.CalledProcessError as e:
        logger.error(str(e))
    finally:
        session.close()

    logger.info("Monitor End")


# Put this in a seperate function,
# that way we can always do session.close()
def do_node_monitor(session):
    # Collect information about system health overall
    hostname = os.uname()[1]

    # Init container variables
    gpu_info = {}

    # Process info containers
    gpu2pid_info = {}
    pid2job_info = {}
    pid2user_info = {}

    logger.info("Querying nvidia-smi")
    pids = []
    smi_out = subprocess.check_output(shlex.split(gpu_command)).decode("utf-8")
    logger.info("Done query nvidia-smi")

    # ripl-s1 has a weird GPU order according to CUDA, so
    # we need to re-order nvidia-smi
    gpu_order_mapping = NODE_GPU_ORDER.get(
        hostname, {smi_id: cuda_id for cuda_id, smi_id in enumerate(range(8))}
    )
    gpu_xml = etree.fromstring(smi_out)  # nvidia-smi
    for gpu in gpu_xml.findall("gpu"):
        gpu_id = int(gpu.find("minor_number").text)
        gpu_id = gpu_order_mapping[gpu_id]
        gpu2pid_info[gpu_id] = [
            int(y.find("pid").text)
            for y in gpu.find("processes").findall("process_info")
        ]
        pids.extend(gpu2pid_info[gpu_id])

    node = (
        session.query(Node)
        .filter_by(name=hostname)
        .options(
            sa.orm.joinedload(Node.gpus),
            sa.orm.joinedload(Node.slurm_jobs),
            sa.orm.joinedload(Node.gpus).joinedload("processes"),
            sa.orm.joinedload(Node.slurm_jobs).joinedload("processes"),
        )
        .first()
    )
    if node is None:
        node = Node(name=hostname)
        session.add(node)

    node.load = "{:.2f} / {:.2f} / {:.2f}".format(*os.getloadavg())
    node.update_time = datetime.datetime.now()
    session.commit()

    if is_gpu_state_same_and_all_in_use(node, gpu2pid_info):
        logger.info("State same, exiting")
        return

    logger.info("Querying slurm PIDs")
    # Get jobid to pid mappings
    try:
        slurm_pids = (
            subprocess.check_output(shlex.split(listpids_command))
            .decode("utf-8")
            .strip()
            .split("\n")[1:]
        )  # job -> pid mappings
    except subprocess.CalledProcessError as e:
        logger.error(str(e))
        slurm_pids = []

    logger.info("Done quering slurm PIDs")

    slurm_pids = [info.split() for info in slurm_pids]
    slurm_pids = [dict(pid=int(info[0]), jid=int(info[1])) for info in slurm_pids]
    slurm_pids = list(
        filter(lambda info: osp.exists(environ_file.format(info["pid"])), slurm_pids)
    )

    # Get process info including who is running it
    pid_list = ",".join(
        [str(y) for y in pids] + [str(info["pid"]) for info in slurm_pids]
    )
    pid2user = {}
    if len(pid_list) > 0:
        ps_info = (
            subprocess.check_output(
                shlex.split(pid_command + pid_list), stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )

        user_names_long = (
            subprocess.check_output(
                shlex.split(user_command + pid_list), stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )

        for line in user_names_long:
            line = line.strip()
            if len(line) == 0:
                continue

            pid, user = line.split(" ")
            pid2user[int(pid)] = user
    else:
        ps_info = []

    all_pids = set()

    for info_line in ps_info:
        info = info_line.split(",:,")
        pid = int(info[0])
        # There is a race between getting the long user-name
        # and a process dying
        if pid in pid2user:
            all_pids.add(pid)
            pid2user_info[pid] = [pid2user[pid]] + info[1:]

    gpu2job_info = dict()
    for info in slurm_pids:
        pid = info["pid"]
        jid = info["jid"]
        if pid not in all_pids:
            continue

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
        else:
            all_pids.remove(pid)

    existing_processes = {
        (proc.id, proc.node_name, proc.gpu_id): proc
        for gpu in node.gpus
        for proc in gpu.processes
    }

    # Jobs can migrate between nodes, so we need to query all jobs!
    existing_jobs = {job.job_id: job for job in session.query(SLURMJob).all()}

    def _add_job(jid, user):
        if jid not in existing_jobs:
            job = SLURMJob(
                job_id=jid, node=node, user=user, is_debug_job=_is_debug_job(jid)
            )

            session.add(job)
            existing_jobs[jid] = job

        job = existing_jobs[jid]
        job.node = node
        job.user = user

        return job

    new_processes = []

    for gpu_id in sorted(gpu2pid_info.keys()):
        if gpu_id >= len(node.gpus):
            gpu = GPU(id=gpu_id, node=node)
            session.add(gpu)
        else:
            gpu = node.gpus[gpu_id]

        if gpu_id in gpu2job_info:
            job_info = gpu2job_info[gpu_id]
            gpu.slurm_job = _add_job(job_info["jid"], job_info["user"])
        else:
            gpu.slurm_job = None

        for pid in gpu2pid_info[gpu_id]:
            if pid not in all_pids:
                continue

            ancestors = get_lineage(pid)
            if ancestors is None:
                logger.error("{} has no ancestors".format(pid))
                continue

            if (pid, hostname, gpu_id) in existing_processes:
                proc = existing_processes[(pid, hostname, gpu_id)]
            else:
                proc = GPUProcess(id=pid, gpu=gpu)
                new_processes.append(proc)

            job_ids = list(
                set([pid2job_info[i] for i in ancestors if i in pid2job_info.keys()])
            )
            if len(job_ids) > 1:
                raise RuntimeError(
                    "More than 1 job ID for a process: {}".format(job_ids)
                )

            user = pid2user_info[pid][0].strip()[0:32]

            if len(job_ids) == 1:
                jid = job_ids[0]
                slurm_job = _add_job(jid, user)
            else:
                slurm_job = None

            cmnd = pid2user_info[pid][2][0:128]

            proc.user = user
            proc.command = cmnd
            proc.slurm_job = slurm_job

    session.add_all(new_processes)
    session.commit()

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
