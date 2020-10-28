import datetime
import os
import re
from typing import List, Set, Union

import click
import sqlalchemy as sa

from gpu_use.cli.utils import filter_labs, is_valid_use, supports_unicode
from gpu_use.db.schema import GPU, GPUProcess, Lab, Node, SLURMJob, User
from gpu_use.db.session import SessionMaker


def _cpu_usage(ent: Union[Lab, User]):
    return sum(job.cpus for job in ent.slurm_jobs)


def _gpu_usage(ent: Union[Lab, User]):
    return len(ent.gpus)


def _invalid_gpu_usage(ent: Union[Lab, User]):
    return sum(0 if is_valid_use(gpu) else 1 for gpu in ent.gpus)


def _idle_gpu_usage(ent: Union[Lab, User]):
    def _is_idle(gpu: GPU):
        if gpu.slurm_job is None:
            return False

        return len(gpu.processes) == 0 and not gpu.slurm_job.is_debug_job

    return sum(1 if _is_idle(gpu) else 0 for gpu in ent.gpus)


def _valid_gpu_usage(ent: Union[Lab, User]):
    return _gpu_usage(ent) - _invalid_gpu_usage(ent) - _idle_gpu_usage(ent)


def _job_gpu_usage(ent: Union[Lab, User]):
    return sum(0 if gpu.slurm_job is None else 1 for gpu in ent.gpus)


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

    if not overcap:
        labs = [lab for lab in labs if lab.name != "overcap"]

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

    for lab in sorted(labs, key=_gpu_usage, reverse=True):
        if _cpu_usage(lab) == 0 and _gpu_usage(lab) == 0:
            continue

        click.echo("|", nl=False)
        click.secho(
            "{:>{width}}".format(lab.name, width=user_width), nl=False, bold=True
        )

        click.echo(" |", nl=False)

        click.echo("  ", nl=False)
        click.secho("{:3d} ".format(_gpu_usage(lab)), fg="green", bold=True, nl=False)
        click.secho("({:4d})".format(_cpu_usage(lab)), fg="cyan", bold=True, nl=False)
        click.echo("  |", nl=False)
        click.secho(
            "    {:3d}    ".format(_invalid_gpu_usage(lab)),
            fg=None if _invalid_gpu_usage(lab) == 0 else "red",
            nl=False,
        )
        click.echo("|", nl=False)
        click.secho(
            "    {:3d}    ".format(_idle_gpu_usage(lab)),
            fg=None if _idle_gpu_usage(lab) == 0 else "red",
            nl=False,
        )
        click.echo("|")
        #  click.echo(ROW_BREAK)

        for user in sorted(lab.users, key=_gpu_usage, reverse=True):
            if _cpu_usage(user) == 0 and _gpu_usage(user) == 0:
                continue

            click.echo("|{:>{width}} |".format(user.name, width=user_width), nl=False)
            click.echo("  ", nl=False)
            click.secho(
                "{:3d} ".format(_gpu_usage(user)), fg="green", bold=True, nl=False
            )
            click.secho(
                "({:4.1f})".format(_cpu_usage(user) / max(_job_gpu_usage(user), 1)),
                fg="cyan",
                bold=True,
                nl=False,
            )
            click.echo("  |", nl=False)
            click.secho(
                "    {:3d}    ".format(_invalid_gpu_usage(user)),
                fg=None if _invalid_gpu_usage(user) == 0 else "red",
                nl=False,
            )
            click.echo("|", nl=False)
            click.secho(
                "    {:3d}    ".format(_idle_gpu_usage(user)),
                fg=None if _idle_gpu_usage(user) == 0 else "red",
                nl=False,
            )
            click.echo("|")

        click.echo(ROW_BREAK)
