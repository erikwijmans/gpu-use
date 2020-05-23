import argparse
import datetime
import os
import platform
import re
import shlex
import subprocess
import sys
from typing import List, Set

import click

from gpu_use.db.schema import GPU, GPUProcess, Node, SLURMJob
from gpu_use.db.session import SessionMaker


def _is_valid_use(gpu: GPU) -> bool:
    return all(proc.slurm_job is not None for proc in gpu.processes)


def _is_user_on_gpu(gpu: GPU, user_names: Set[str]) -> bool:
    return (
        len(user_names) == 0
        or any(proc.user in user_names for proc in gpu.processes)
        or (gpu.slurm_job is not None and gpu.slurm_job.user in user_names)
    )


NODE_NAME_WITH_TIME = "{}\t\tUpdated: {}"


def _show_non_dense(
    nodes: List[Node], user_names: Set[str], only_errors, display_time, display_load
):
    for node in nodes:
        if not only_errors:
            name_str = click.style(node.name, bold=True)
            if display_load:
                name_str = "{}\tLoad: {}".format(name_str, node.load)
            if display_time:
                name_str = NODE_NAME_WITH_TIME.format(
                    name_str, node.update_time.strftime("%Y-%m-%d %H:%M:%S")
                )

            click.secho(
                "-------------------------------------------------------------------\n"
                + name_str
                + "\n-------------------------------------------------------------------",
                fg="bright_white",
                color=True,
            )

        for gpu in node.gpus:
            if not _is_user_on_gpu(gpu, user_names):
                continue

            reserved = False
            in_use = False
            valid_use = _is_valid_use(gpu)
            error = False
            err_msg = ""

            if gpu.slurm_job is not None:
                reserved = True

            if len(gpu.processes) > 0:
                in_use = True

            res_char = u"\u25A1"
            use_char = u"\u25A1"
            res_record = "-"
            color = "bright_white"

            if reserved:
                res_char = u"\u25A0"
                res_record = "{} ({})".format(gpu.slurm_job.user, gpu.slurm_job.job_id)

            if in_use:
                use_char = u"\u25A0"

            if reserved and in_use and not valid_use:
                color = "red"
                err_msg = ""
                error = True

            if reserved and not in_use:
                color = "red"
                err_msg = "[Idle reservation]"
                error = True
                if gpu.slurm_job.is_debug_job:
                    err_msg = "[Idle reservation - DEBUG]"
                    color = "magenta"

            if in_use and not reserved:
                color = "red"
                err_msg = "[Use without reservation]"
                error = True

            if not only_errors or (only_errors and error):
                gpu_record = "{}{}[{}]".format(res_char, use_char, gpu.id)
                if only_errors:
                    click.secho(
                        "{} {} {} {}".format(
                            node.name, gpu_record, res_record, err_msg
                        ),
                        fg=color,
                        nl=True,
                        color=True,
                    )
                else:
                    click.secho(
                        "{} {} {}".format(gpu_record, res_record, err_msg),
                        fg=color,
                        nl=False,
                        color=True,
                    )

                for proc in gpu.processes:
                    color = None
                    error = False
                    err_msg = ""

                    if (
                        proc.slurm_job is not None
                        and proc.slurm_job_id != gpu.slurm_job_id
                    ):
                        color = "red"
                        err_msg = "[Wrong Job (" + str(proc.slurm_job_id) + ")]"
                        error = True

                    if proc.slurm_job is None:
                        color = "red"
                        err_msg = "[No Job]"
                        error = True

                    if only_errors or (not only_errors and error):
                        click.secho(
                            "       "
                            + "{} {} {} {}".format(
                                proc.id, proc.command, proc.user, err_msg
                            ),
                            fg=color,
                            nl=only_errors,
                            color=True,
                        )

        if not only_errors:
            click.echo("")


def _show_dense(nodes: List[Node], user_names: Set[str], display_time, display_load):
    longest_name_length = max(len(node.name) for node in nodes)
    max_gpus = max(
        len([gpu for gpu in node.gpus if _is_user_on_gpu(gpu, user_names)])
        for node in nodes
    )

    for node in nodes:
        gpu_tot = 0
        gpu_res = 0
        gpu_used = 0

        name_str = ""
        if display_load:
            name_str += "{:5.2f} ".format(float(node.load.split("/")[0]))

        name_str = name_str + click.style(
            "{:{width}}".format(node.name, width=longest_name_length), bold=True
        )
        name_str = click.style(name_str, fg="bright_white")

        gpus_str = ""
        num_valid = 0
        for gpu in node.gpus:
            if not _is_user_on_gpu(gpu, user_names):
                continue

            num_valid += 1

            gpu_tot = gpu_tot + 1

            reserved = False
            in_use = False
            error = False
            err_msg = ""

            if gpu.slurm_job is not None:
                reserved = True
                gpu_res = gpu_res + 1

            if len(gpu.processes) > 0:
                in_use = True
                gpu_used = gpu_used + 1

            valid_use = _is_valid_use(gpu)

            res_char = u"\u25A1"
            use_char = u"\u25A1"
            color = "bright_white"

            if reserved:
                res_char = u"\u25A0"

            if in_use:
                use_char = u"\u25A0"

            if reserved and in_use and not valid_use:
                color = "red"

            if reserved and not in_use:
                color = "red"
                if gpu.slurm_job.is_debug_job:
                    color = "magenta"

            if in_use and not reserved:
                color = "red"

            gpus_str += click.style(
                "\t{}{}[{}]".format(res_char, use_char, gpu.id), fg=color
            )

        for _ in range(max_gpus - num_valid):
            gpus_str += "\t     "

        name_str += gpus_str
        name_str += "\t{} / {} / {}".format(gpu_used, gpu_res, gpu_tot)
        if display_time:
            name_str = NODE_NAME_WITH_TIME.format(
                name_str, node.update_time.strftime("%Y-%m-%d %H:%M:%S")
            )

        click.echo(name_str, color=True)


@click.command(name="gpu-use")
@click.option(
    "-n",
    "--node",
    type=str,
    default=None,
    show_default=True,
    help="Specify a specific node -- regex enabled",
)
@click.option(
    "-u",
    "--user",
    type=str,
    default=None,
    show_default=True,
    help="Specify a specific user -- regex enabled",
)
@click.option(
    "-d", "--dense", help="Use dense output format", default=False, is_flag=True
)
@click.option(
    "-e",
    "--error",
    "only_errors",
    help="Only display errors. Cannot be used with --dense",
    default=False,
    is_flag=True,
)
@click.option(
    "-t",
    "--display-time",
    "display_time",
    help="Display the last time the node was updated.",
    default=False,
    is_flag=True,
)
@click.option(
    "-l",
    "--display-load",
    "display_load",
    help="Display the average load for each load.  Displayed as <1min> / <5min> / <15min> normally."
    "  Only <1min> is shown in dense mode.",
    default=False,
    is_flag=True,
)
def gpu_use_cli(node, user, dense, only_errors, display_time, display_load):
    r"""Display real-time information about the GPUs on skynet


Notes:

    - When using gpu-use with `watch`, add `--color` for the output to be displayed
correctly, i.e. use `watch --color gpu-use -d`.
    """
    if only_errors and dense:
        raise click.BadArgumentUsage("--dense and --errors are mutually exclusive")

    node_re = re.compile(node) if node is not None else None
    user_re = re.compile(user) if user is not None else None

    session = SessionMaker()

    user_names = []
    nodes = session.query(Node)
    if node_re is not None or user_re is not None:
        node_filter = None
        if node_re is not None:
            node_names = [
                name[0]
                for name in session.query(Node.name).all()
                if node_re.match(name[0]) is not None
            ]
            if len(node_names) == 0:
                raise click.BadArgumentUsage("No nodes matched {}".format(node))

            node_filter = Node.name.in_(node_names)

        if user_re is not None:
            user_names = [
                name[0]
                for name in session.query(GPUProcess.user).distinct().all()
                if user_re.match(name[0]) is not None
            ] + [
                name[0]
                for name in session.query(SLURMJob.user).distinct().all()
                if user_re.match(name[0]) is not None
            ]

            if len(user_names) == 0:
                raise click.BadArgumentUsage("No users matched {}".format(user))

            user_filter = Node.gpus.any(
                GPU.processes.any(GPUProcess.user.in_(user_names))
            ) | Node.slurm_jobs.any(SLURMJob.user.in_(user_names))

            if node_filter is None:
                node_filter = user_filter
            else:
                node_filter = node_filter & user_filter

        nodes = nodes.filter(node_filter)

    nodes = nodes.order_by(Node.name).all()

    user_names = set(user_names)
    if dense:
        _show_dense(nodes, user_names, display_time, display_load)
    else:
        _show_non_dense(nodes, user_names, only_errors, display_time, display_load)


if __name__ == "__main__":
    gpu_use_cli()
