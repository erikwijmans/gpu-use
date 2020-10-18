import datetime
import os
import re
from typing import List, Set

import click
import sqlalchemy as sa

from gpu_use.cli.utils import filter_labs, supports_unicode
from gpu_use.cli.view_command.dense_view import show_dense
from gpu_use.cli.view_command.errors_view import show_errors
from gpu_use.cli.view_command.regular_view import show_regular
from gpu_use.db.schema import GPU, GPUProcess, Lab, Node, SLURMJob, User
from gpu_use.db.session import SessionMaker


@click.command(name="view")
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
    "-a",
    "--lab",
    type=str,
    default=None,
    show_default=True,
    help="Specify a specific lab -- regex enabled",
)
@click.option(
    "-d/-nd",
    "--dense/--no-dense",
    help="Use dense output format",
    default=True,
    is_flag=True,
)
@click.option(
    "-e",
    "--error",
    "only_errors",
    help="Display only errors and full information about errors."
    " Cannot be used with --dense",
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
def gpu_use_view_command(
    node, user, lab, dense, only_errors, display_time, display_load
):
    r"""Display real-time information about the GPUs on skynet

There are two blocks per GPU, the left block indicates whether or not the GPU is
alloced and the right block indicates whether or not something is running on it.
Red means there is some error (idle reservation or process running out of SLURM)
and pink/magenta means idle reservation but in the debug partition.


Notes:

    - When using gpu-use with `watch`, add `--color` for the output to be displayed
correctly, i.e. use `watch --color gpu-use -d`.
    """
    if dense and only_errors:
        dense = False

    node_re = re.compile(node) if node is not None else None
    user_re = re.compile(user) if user is not None else None

    session = SessionMaker()

    nodes = session.query(Node)
    users = None
    if node_re is not None:
        node_names = [
            name[0]
            for name in session.query(Node.name).all()
            if node_re.match(name[0]) is not None
        ]
        if len(node_names) == 0:
            raise click.BadArgumentUsage("No nodes matched {}".format(node))

        node_filter = Node.name.in_(node_names)
        nodes = nodes.filter(Node.name.in_(node_names))

    if lab is not None:
        labs = filter_labs(session, lab)

        user_names = list(set(user.name for lab in labs for user in lab.users))
        users = session.query(User).filter(User.name.in_(user_names))

    if user_re is not None:
        user_names = [
            name[0]
            for name in session.query(User.name).all()
            if user_re.match(name[0]) is not None
        ]

        if len(user_names) == 0:
            raise click.BadArgumentUsage("No users matched {}".format(user))

        users = (session.query(User) if users is None else users).filter(
            User.name.in_(user_names)
        )
        nodes = nodes.filter(Node.users.any(User.name.in_(user_names)))

    nodes = (
        nodes.order_by(Node.name)
        .options(
            sa.orm.joinedload(Node.gpus),
            sa.orm.joinedload(Node.slurm_jobs),
            sa.orm.joinedload(Node.gpus).joinedload("processes"),
            sa.orm.joinedload(Node.slurm_jobs).joinedload("processes"),
        )
        .all()
    )

    if users is not None:
        users = users.all()

    if not supports_unicode():
        click.echo(
            "Terminal does not support unicode, do `export LANG=en_US.UTF-8` for a better experience (may also need to start tmux with `-u`)"
        )

    nodes = sorted(nodes, key=lambda n: len(n.gpus))

    if only_errors:
        show_errors(nodes, users)
    elif dense:
        show_dense(nodes, users, display_time, display_load)
    else:
        show_regular(nodes, users, display_time, display_load)
