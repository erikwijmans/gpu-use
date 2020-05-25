from typing import List, Set

import click

from gpu_use.cli.regular_view import NODE_NAME_WITH_TIME
from gpu_use.cli.utils import is_user_on_gpu, parse_gpu, parse_process
from gpu_use.db.schema import Node


def show_dense(nodes: List[Node], user_names: Set[str], display_time, display_load):
    longest_name_length = max(len(node.name) for node in nodes)
    max_gpus = max(
        len([gpu for gpu in node.gpus if is_user_on_gpu(gpu, user_names)])
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
        for gpu in node.gpus:
            if not is_user_on_gpu(gpu, user_names):
                continue

            gpu_tot = gpu_tot + 1

            res = parse_gpu(gpu)

            if res.reserved:
                gpu_res += 1

            if res.in_use:
                gpu_used += 1

            gpus_str += click.style(
                "\t{}{}[{}]".format(res.res_char, res.use_char, gpu.id), fg=res.color
            )

        for _ in range(max_gpus - gpu_tot):
            gpus_str += "\t     "

        name_str += gpus_str
        name_str += "\t{} / {} / {}".format(gpu_used, gpu_res, gpu_tot)
        if display_time:
            name_str = NODE_NAME_WITH_TIME.format(
                name_str, node.update_time.strftime("%Y-%m-%d %H:%M:%S")
            )

        click.echo(name_str, color=True)
