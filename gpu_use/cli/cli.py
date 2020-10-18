import click
import click_default_group

from gpu_use import __version__
from gpu_use.cli.lab_command import gpu_use_lab_command
from gpu_use.cli.view_command import gpu_use_view_command


@click.group(
    cls=click_default_group.DefaultGroup,
    default="view",
    default_if_no_args=True,
    name="gpu-use",
)
def gpu_use_cli():
    r"""Display real-time information about usage on skynet on skynet

To see the help string for a given command, use `gpu-use <command> --help`

Executes the `view` command by default
    """
    pass


@gpu_use_cli.command()
def version():
    click.echo("Version: {}".format(__version__))


gpu_use_cli.add_command(gpu_use_view_command)
gpu_use_cli.add_command(gpu_use_lab_command)


if __name__ == "__main__":
    gpu_use_cli()
