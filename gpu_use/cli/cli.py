import argparse
import os
import platform
import re
import shlex
import subprocess
import sys
from time import gmtime, strftime

import click

from gpu_use.db.db_schema import GPU, Node, Process
from gpu_use.db.session import SessionMaker
from gpu_use.monitor import node_monitor

DEBUG_CHECK = "scontrol show job {} 2>/dev/null | grep \"Partition\" | awk -F'[ =]+' '{{print $3 == \"debug\"}}'"


def _is_debug_job(jid):
    try:
        return bool(
            int(
                subprocess.check_output(
                    DEBUG_CHECK.format(gpu.slurm_job.job_id)
                ).decode("utf-8")
            )
        )
    except ValueError:
        return False


def _is_valid_use(gpu: GPU):
    return all(proc.slurm_job is not None for proc in gpu.processes)


def _show_dense(nodes, user: re.Pattern, only_errors):
    for node in nodes:
        if not only_errors:
            click.echo(
                click.style(
                    "-------------------------------------------------------------------\n"
                    + node.name
                    + "\n-------------------------------------------------------------------",
                    fg="white",
                    bold=True,
                )
            )

        for gpu in node.gpus:
            if not (
                any(user.match(proc.user) for proc in gpu.processes)
                or user.match(gpu.slurm_job_user) is not None
            ):
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

            res_char = u"\u25A1".encode("utf_8")
            use_char = u"\u25A1".encode("utf_8")
            res_record = "-"
            color = "white"

            if reserved:
                res_char = u"\u25A0".encode("utf_8")
                res_record = "{} ({})".format(gpu.user, gpu.slurm_job.job_id)

            if in_use:
                use_char = u"\u25A0".encode("utf_8")

            if reserved and in_use and not valid_use:
                color = "red"
                err_msg = ""
                error = True

            if reserved and not in_use:
                color = "red"
                err_msg = "[Idle reservation]"
                error = True
                if _is_debug_job(gpu.slurm_job_id):
                    err_msg = "[Idle reservation - DEBUG]"
                    color = "magenta"

            if in_use and not reserved:
                color = "red"
                err_msg = "[Use without reservation]"
                error = True

            if not only_errors or (only_errors and error):
                if only_errors:
                    click.echo(
                        click.style(
                            str((node, gpu_record, res_record, err_msg)), fg=color
                        )
                    )
                else:
                    click.echo(
                        click.style(str((gpu_record, res_record, err_msg)), fg=color)
                    )

                for proc in gpu.processes:
                    color = "white"
                    error = False
                    err_msg = ""

                    if proc.slurm_job is not None:
                        color = "red"
                        err_msg = "[Wrong Job (" + str(proc["jid"]) + ")]"
                        error = True

                    else:
                        color = "red"
                        err_msg = "[No Job]"
                        error = True

                    if only_errors or (not only_errors and error):
                        click.echo(
                            click.style(
                                "       "
                                + "{} {} {} {}".format(
                                    proc.id, proc.command, proc.user, err_msg
                                ),
                                fg=color,
                            )
                        )

        if not only_errors:
            click.echo("\n")


def _show_non_dense(nodes, user: re.Pattern, only_errors):
    for node in nodes:
        gpu_tot = 0
        gpu_res = 0
        gpu_used = 0
        click.echo(click.style(node.name, fg="white", bold=True))

        for gpu in node.gpus:
            if not (
                any(user.match(proc.user) for proc in gpu.processes)
                or user.match(gpu.slurm_job_user) is not None
            ):
                continue

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

            res_char = u"\u25A1".encode("utf_8")
            use_char = u"\u25A1".encode("utf_8")
            color = "white"

            if reserved:
                res_char = u"\u25A0".encode("utf_8")

            if in_use:
                use_char = u"\u25A0".encode("utf_8")

            if reserved and in_use and not valid_use:
                color = "red"

            if reserved and not in_use:
                color = "red"
                if _is_debug_job(gpu.slurm_job_id):
                    color = "magenta"

            if in_use and not reserved:
                color = "red"

            click.echo(
                click.style("\t{}{}[{}]".format(res_char, use_char, gpu.id), fg=color)
            )

        click.echo("\t{} / {} / {}".format(gpu_used, gpu_res, gpu_tot))


@click.command(name="gpu-use")
@click.option(
    "-n",
    "--node",
    type=str,
    default=".*",
    show_default=True,
    help="Specify a specific node -- regex enabled",
)
@click.option(
    "-u",
    "--user",
    type=str,
    default=".*",
    show_default=True,
    help="Specify a specific user -- regex enabled",
)
@click.option("-d", "--dense", default=False, is_flag=True)
@click.option("-e", "--error", "only_errors", default=False, is_flag=True)
def gpu_use_cli(node, user, dense, only_errors):
    node = re.compile(node)
    user = re.compile(user)

    session = SessionMaker()

    node_names = [
        name for name in session.query(Node.name).all() if node.match(name) is not None
    ]
    user_names = [
        name
        for name in session.query(Process.user).distinct().all()
        if user.match(name) is not None
    ]
    nodes = (
        session.query(Node)
        .filter(
            Node.name.in_(node_names) & Node.processes.has(Process.user.in_(user_names))
        )
        .all()
    )

    if dense:
        _show_dense(nodes, user, only_errors)
    else:
        _show_non_dense(nodes, user, only_errors)


if __name__ == "__main__":
    node_monitor()
    gpu_use_cli()
