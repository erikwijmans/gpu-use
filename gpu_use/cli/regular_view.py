from typing import List, Set

import click

from gpu_use.cli.utils import is_user_on_gpu, parse_gpu, parse_process
from gpu_use.db.schema import Node

NODE_NAME_WITH_TIME = "{}\t\tUpdated: {}"


def show_regular(nodes: List[Node], user_names: Set[str], display_time, display_load):
    for node in nodes:
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
            if not is_user_on_gpu(gpu, user_names):
                continue

            res = parse_gpu(gpu)

            gpu_record = "{}{}[{}]".format(res.res_char, res.use_char, gpu.id)
            click.secho(
                "{} {} {}".format(gpu_record, res.res_record, res.err_msg),
                fg=res.color,
                nl=False,
                color=True,
            )

            if res.error:
                for proc in gpu.processes:
                    proc_res = parse_process(gpu, proc)
                    if proc_res.error:
                        click.secho(
                            "       "
                            + "{} {} {} {}".format(
                                proc.id, proc.command, proc.user, proc_res.err_msg
                            ),
                            fg=proc_res.color,
                            nl=False,
                            color=True,
                        )

        click.echo("")