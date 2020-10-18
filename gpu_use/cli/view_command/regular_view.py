from typing import List, Set

import click

from gpu_use.cli.utils import (
    gray_if_out_of_date,
    is_user_on_gpu,
    parse_gpu,
    parse_process,
)
from gpu_use.db.schema import Node, User

NODE_NAME_WITH_TIME = "{}\t\tUpdated: {}"


def show_regular(nodes: List[Node], users: List[User], display_time, display_load):
    longest_name_length = max(len(node.name) for node in nodes)
    for node in nodes:
        name_str = click.style(
            "{:{width}}".format(node.name, width=longest_name_length), bold=True
        )
        name_str = click.style(
            "-------------------------------------\t"
            + name_str
            + "\t-------------------------------------",
            fg="bright_white",
        )

        if display_load:
            name_str = "{}\tLoad: {}".format(name_str, node.load)

        if display_time:
            name_str = NODE_NAME_WITH_TIME.format(
                name_str, node.update_time.strftime("%Y-%m-%d %H:%M:%S")
            )

        click.echo(gray_if_out_of_date(name_str, node.update_time), color=True)

        for gpu in node.gpus:
            if not is_user_on_gpu(gpu, users):
                continue

            res = parse_gpu(gpu)

            gpu_record = gray_if_out_of_date(
                "{}{}[{}]".format(res.res_char, res.use_char, gpu.id), gpu.update_time
            )
            click.echo(
                gray_if_out_of_date(
                    click.style(
                        "{} {} {}".format(gpu_record, res.res_record, res.err_msg),
                        fg=res.color,
                    ),
                    node.update_time,
                ),
                nl=False,
                color=True,
            )

            if res.error:
                for proc in gpu.processes:
                    proc_res = parse_process(gpu, proc)
                    if proc_res.error:
                        click.echo(
                            gray_if_out_of_date(
                                click.style(
                                    "       "
                                    + "{} {} {} {}".format(
                                        proc.id,
                                        proc.command,
                                        proc.user_name,
                                        proc_res.err_msg,
                                    ),
                                    fg=proc_res.color,
                                ),
                                node.update_time,
                            ),
                            nl=False,
                            color=True,
                        )

        click.echo("")
