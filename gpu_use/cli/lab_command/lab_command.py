import datetime
import os
import re
from typing import List, Set, Union

import click
import sqlalchemy as sa

from gpu_use.cli.utils import (
    filter_labs,
    is_out_of_date,
    is_valid_use,
    supports_unicode,
)
from gpu_use.db.schema import GPU, GPUProcess, Lab, Node, SLURMJob, User
from gpu_use.db.session import SessionMaker


def _cpu_usage(ent: Union[Lab, User], overcap: bool):
    return sum(
        job.cpus if not is_out_of_date(job.node.update_time) else 0
        for job in ent.slurm_jobs
        if overcap or not job.is_overcap_job
    )


def _gpu_is_overcap(gpu: GPU) -> bool:
    return gpu.slurm_job is not None and gpu.slurm_job.is_overcap_job


def _gpu_usage(ent: Union[Lab, User], overcap: bool):
    return sum(
        1
        if (overcap or not _gpu_is_overcap(gpu)) and not is_out_of_date(gpu.update_time)
        else 0
        for gpu in ent.gpus
    )


def _invalid_gpu_usage(ent: Union[Lab, User], overcap: bool):
    return sum(
        0 if is_valid_use(gpu) or is_out_of_date(gpu.update_time) else 1
        for gpu in ent.gpus
    )


def _idle_gpu_usage(ent: Union[Lab, User], overcap: bool):
    def _is_idle(gpu: GPU):
        if not overcap and _gpu_is_overcap(gpu):
            return False

        if gpu.slurm_job is None:
            return False

        return len(gpu.processes) == 0 and not gpu.slurm_job.is_debug_job

    return sum(
        1 if _is_idle(gpu) and not is_out_of_date(gpu.update_time) else 0
        for gpu in ent.gpus
    )


def _valid_gpu_usage(ent: Union[Lab, User], overcap: bool):
    return (
        _gpu_usage(ent, overcap)
        - _invalid_gpu_usage(ent, overcap)
        - _idle_gpu_usage(ent, overcap)
    )


def _job_gpu_usage(ent: Union[Lab, User], overcap: bool):
    return sum(
        0 if gpu.slurm_job is None or (not overcap and _gpu_is_overcap(gpu)) else 1
        for gpu in ent.gpus
    )


@click.command(name="lab")
@click.option(
    "-a",
    "--lab",
    type=str,
    default=None,
    show_default=True,
    help="Specify a specific lab -- regex enabled",
)
@click.option(
    "--overcap/--no-overcap",
    "-o/-noc",
    default=True,
    show_default=True,
    help="Whether or not to include the overcap lab/account",
)
def gpu_use_lab_command(lab, overcap):
    r"""Display cluster usage by lab
    """

    session = SessionMaker()

    if lab is not None:
        labs = filter_labs(session, lab)
    else:
        labs = session.query(Lab).all()

    if len(labs) == 0:
        raise click.BadArgumentUsage("Given options result in no labs")

    labs = (
        session.query(Lab)
        .order_by(Lab.name)
        .filter(Lab.name.in_([lab.name for lab in labs]))
        .all()
    )

    user_width = (
        max(
            [len(user.name) for lab in labs for user in lab.users]
            + [len(lab.name) for lab in labs]
        )
        + 2
        - 1
    )
    cpu_gpu_width = 14
    invalid_gpu_width = 11

    ROW_BREAK = (
        "|{}-|".format("-" * user_width)
        + "{}|".format("-" * cpu_gpu_width)
        + "{}|".format("-" * invalid_gpu_width)
        + "{}|".format("-" * invalid_gpu_width)
    )

    click.echo()
    click.echo(ROW_BREAK)
    click.echo("|{:>{width}} |".format("Username", width=user_width), nl=False)
    click.echo("  ", nl=False)
    click.secho("  G ", fg="green", bold=True, nl=False)
    click.secho("(   C)", fg="cyan", bold=True, nl=False)
    click.echo("  |", nl=False)
    click.echo(" Invalid G |   Idle G  |")
    click.echo(ROW_BREAK)

    for lab in sorted(labs, key=lambda l: _gpu_usage(l, overcap), reverse=True):
        if _cpu_usage(lab, overcap) == 0 and _gpu_usage(lab, overcap) == 0:
            continue

        click.echo("|", nl=False)
        click.secho(
            "{:>{width}}".format(lab.name, width=user_width), nl=False, bold=True
        )

        click.echo(" |", nl=False)

        click.echo("  ", nl=False)
        click.secho(
            "{:3d} ".format(_gpu_usage(lab, overcap)), fg="green", bold=True, nl=False
        )
        click.secho(
            "({:4d})".format(_cpu_usage(lab, overcap)), fg="cyan", bold=True, nl=False
        )
        click.echo("  |", nl=False)
        click.secho(
            "    {:3d}    ".format(_invalid_gpu_usage(lab, overcap)),
            fg=None if _invalid_gpu_usage(lab, overcap) == 0 else "red",
            nl=False,
        )
        click.echo("|", nl=False)
        click.secho(
            "    {:3d}    ".format(_idle_gpu_usage(lab, overcap)),
            fg=None if _idle_gpu_usage(lab, overcap) == 0 else "red",
            nl=False,
        )
        click.echo("|")
        #  click.echo(ROW_BREAK)

        for user in sorted(
            lab.users, key=lambda u: _gpu_usage(u, overcap), reverse=True
        ):
            if _cpu_usage(user, overcap) == 0 and _gpu_usage(user, overcap) == 0:
                continue

            click.echo("|{:>{width}} |".format(user.name, width=user_width), nl=False)
            click.echo("  ", nl=False)
            click.secho(
                "{:3d} ".format(_gpu_usage(user, overcap)),
                fg="green",
                bold=True,
                nl=False,
            )
            click.secho(
                "({:4.1f})".format(
                    _cpu_usage(user, overcap) / max(_job_gpu_usage(user, overcap), 1)
                ),
                fg="cyan",
                bold=True,
                nl=False,
            )
            click.echo("  |", nl=False)
            click.secho(
                "    {:3d}    ".format(_invalid_gpu_usage(user, overcap)),
                fg=None if _invalid_gpu_usage(user, overcap) == 0 else "red",
                nl=False,
            )
            click.echo("|", nl=False)
            click.secho(
                "    {:3d}    ".format(_idle_gpu_usage(user, overcap)),
                fg=None if _idle_gpu_usage(user, overcap) == 0 else "red",
                nl=False,
            )
            click.echo("|")

        click.echo(ROW_BREAK)
