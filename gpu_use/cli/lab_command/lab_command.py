import datetime
import os
import re
from typing import List, Set, Union

import click
import sqlalchemy as sa

from gpu_use.cli.utils import filter_labs, supports_unicode
from gpu_use.db.schema import GPU, GPUProcess, Lab, Node, SLURMJob, User
from gpu_use.db.session import SessionMaker


def _cpu_usage(ent: Union[Lab, User]):
    return sum(job.cpus for job in ent.slurm_jobs)


def _gpu_usage(ent: Union[Lab, User]):
    return len(ent.gpus)


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
        .options(
            sa.orm.joinedload(Lab.users),
            sa.orm.joinedload(Lab.slurm_jobs),
            sa.orm.joinedload(Lab.gpus),
        )
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

    ROW_BREAK = "|{}-|".format("-" * user_width) + "{}|".format("-" * cpu_gpu_width)

    click.echo()
    click.echo(ROW_BREAK)
    click.echo("|{:>{width}} |".format("Username", width=user_width), nl=False)
    click.echo("  ", nl=False)
    click.secho("  G ", fg="green", bold=True, nl=False)
    click.secho("(   C)", fg="cyan", bold=True, nl=False)
    click.echo("  |")
    click.echo(ROW_BREAK)

    for lab in labs:
        click.echo("|", nl=False)
        click.secho(
            "{:>{width}}".format(lab.name, width=user_width), nl=False, bold=True
        )

        click.echo(" |", nl=False)

        click.echo("  ", nl=False)
        click.secho("{:3d} ".format(_gpu_usage(lab)), fg="green", bold=True, nl=False)
        click.secho("({:4d})".format(_cpu_usage(lab)), fg="cyan", bold=True, nl=False)
        click.echo("  |")
        #  click.echo(ROW_BREAK)

        for user in sorted(lab.users, key=_gpu_usage, reverse=True):
            click.echo("|{:>{width}} |".format(user.name, width=user_width), nl=False)
            click.echo("  ", nl=False)
            click.secho(
                "{:3d} ".format(_gpu_usage(user)), fg="green", bold=True, nl=False
            )
            click.secho(
                "({:4.1f})".format(_cpu_usage(user) / max(_gpu_usage(user), 1)),
                fg="cyan",
                bold=True,
                nl=False,
            )
            click.echo("  |")

        click.echo(ROW_BREAK)
