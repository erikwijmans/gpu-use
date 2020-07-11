from typing import List, Set

import click

from gpu_use.cli.utils import (
    gray_if_out_of_date,
    is_user_on_gpu,
    parse_gpu,
    parse_process,
)
from gpu_use.db.schema import Node


def show_errors(nodes: List[Node], user_names: Set[str]):
    valid_nodes_name_lengths = [
        len(node.name)
        for node in nodes
        if any(
            is_user_on_gpu(gpu, user_names) and parse_gpu(gpu).error
            for gpu in node.gpus
        )
    ]

    if len(valid_nodes_name_lengths) == 0:
        click.echo("No errors for requested node(s)/user(s).  Hooray!")
        return

    longest_name_length = max(valid_nodes_name_lengths)

    for node in nodes:
        for gpu in node.gpus:
            if not is_user_on_gpu(gpu, user_names):
                continue

            res = parse_gpu(gpu)

            if res.error:
                gpu_record = "{}{}[{}]".format(res.res_char, res.use_char, gpu.id)
                click.echo(
                    gray_if_out_of_date(
                        click.style(
                            "{:{width}} {} {} {}".format(
                                node.name,
                                gpu_record,
                                res.res_record,
                                res.err_msg,
                                width=longest_name_length,
                            ),
                            fg=res.color,
                        ),
                        node.update_time,
                    ),
                    color=True,
                )

                for proc in gpu.processes:
                    proc_res = parse_process(gpu, proc)

                    click.echo(
                        gray_if_out_of_date(
                            click.style(
                                (" " * longest_name_length)
                                + "       "
                                + "{} {} {} {}".format(
                                    proc.id, proc.command, proc.user, proc_res.err_msg
                                ),
                                fg=proc_res.color,
                            ),
                            node.update_time,
                        ),
                        color=True,
                    )
