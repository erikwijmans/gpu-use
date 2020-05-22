import argparse
import datetime
import os
import platform
import re
import shlex
import subprocess
import sys

import click

from gpu_use.db.db_schema import GPU, GPUProcess, Node, SLURMJob
from gpu_use.db.session import SessionMaker


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
                    fg="bright_white",
                    bold=True,
                )
            )

        for gpu in node.gpus:
            if user is not None and not (
                any(user.match(proc.user) for proc in gpu.processes)
                or (
                    gpu.slurm_job is not None
                    and user.match(gpu.slurm_job.user) is not None
                )
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
                    click.echo(
                        click.style(
                            "{} {} {} {}".format(node, gpu_record, res_record, err_msg),
                            fg=color,
                        ),
                        nl=False,
                    )
                else:
                    click.echo(
                        click.style(
                            "{} {} {}".format(gpu_record, res_record, err_msg), fg=color
                        ),
                        nl=False,
                    )

                for proc in gpu.processes:
                    color = "bright_white"
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
                        click.echo(
                            click.style(
                                "       "
                                + "{} {} {} {}".format(
                                    proc.id, proc.command, proc.user, err_msg
                                ),
                                fg=color,
                            ),
                            nl=False,
                        )

        if not only_errors:
            click.echo("")


def _show_non_dense(nodes, user: re.Pattern, only_errors):
    for node in nodes:
        gpu_tot = 0
        gpu_res = 0
        gpu_used = 0
        click.echo(click.style(node.name, fg="bright_white", bold=True))

        for gpu in node.gpus:
            if user is not None and not (
                any(user.match(proc.user) for proc in gpu.processes)
                or (
                    gpu.slurm_job is not None
                    and user.match(gpu.slurm_job.user) is not None
                )
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

            click.echo(
                click.style("\t{}{}[{}]".format(res_char, use_char, gpu.id), fg=color),
                nl=False,
            )

        click.echo("")
        click.echo("\t{} / {} / {}".format(gpu_used, gpu_res, gpu_tot))


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
@click.option("-d", "--dense", default=False, is_flag=True)
@click.option("-e", "--error", "only_errors", default=False, is_flag=True)
@click.option("-t", "--display-time", "display_time", default=False, is_flag=True)
def gpu_use_cli(node, user, dense, only_errors, display_time):
    if display_time:
        now = datetime.datetime.now()
        click.secho(now.strftime("%Y-%m-%d %H:%M:%S"), fg="green")
        click.echo()

    node_re = re.compile(node) if node is not None else None
    user_re = re.compile(user) if user is not None else None

    session = SessionMaker()

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

    if dense:
        _show_non_dense(nodes, user_re, only_errors)
    else:
        _show_dense(nodes, user_re, only_errors)


if __name__ == "__main__":
    gpu_use_cli()
