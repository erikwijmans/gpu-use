import datetime
import io
import os
import re
from typing import List, Optional, Set

import attr
import click
import sqlalchemy as sa

from gpu_use.db.schema import GPU, GPUProcess, Lab, Node, SLURMJob, User


def supports_unicode() -> bool:
    return "UTF-8" in os.environ.get("LANG", "en_US")


def is_valid_use(gpu: GPU) -> bool:
    return all(
        (proc.slurm_job is not None and proc.slurm_job == gpu.slurm_job)
        for proc in gpu.processes
    )


def is_user_on_gpu(gpu: GPU, users: Optional[List[User]]) -> bool:
    return users is None or gpu.user in users


def filter_labs(session, lab) -> List[Lab]:
    lab_re = re.compile(lab)
    labs = [l for l in session.query(Lab).all() if lab_re.match(l.name) is not None]
    if len(labs) == 0:
        raise click.BadArgumentUsage("No labs matched {}".format(lab))

    return labs


def gray_if_out_of_date(
    string: str,
    update_time: datetime.datetime,
    max_lag_time=datetime.timedelta(minutes=10),
):
    if (datetime.datetime.now() - update_time) < max_lag_time:
        return string

    stripped_string = io.StringIO()
    # Use an echo into a io.StringIO() to strip existing formatting
    click.echo(message=string, color=False, nl=False, file=stripped_string)
    stripped_string.seek(0)
    return click.style(stripped_string.read(), fg="white", dim=True)


@attr.s(auto_attribs=True)
class GPUParseResult:
    reserved: bool = False
    in_use: bool = False
    valid_use: bool = False
    error: bool = False
    err_msg: str = ""
    res_char: str = u"\u25A1" if supports_unicode() else "-"
    use_char: str = u"\u25A1" if supports_unicode() else "-"
    res_record: str = "-"
    color: str = "bright_white"


def parse_gpu(gpu: GPU) -> GPUParseResult:
    res = GPUParseResult()
    res.valid_use = is_valid_use(gpu)
    res.reserved = gpu.slurm_job is not None
    res.in_use = len(gpu.processes) > 0

    if res.reserved:
        res.res_char = u"\u25A0" if supports_unicode() else "#"
        res.res_record = "{} ({})".format(gpu.slurm_job.user_name, gpu.slurm_job.job_id)

    if res.in_use:
        res.use_char = u"\u25A0" if supports_unicode() else "#"

    if res.reserved and res.in_use and not res.valid_use:
        res.color = "red"
        res.err_msg = ""
        res.error = True

    if res.reserved and not res.in_use:
        res.color = "red"
        res.err_msg = "[Idle reservation]"
        res.error = True
        if gpu.slurm_job.is_debug_job:
            res.err_msg = "[Idle reservation - DEBUG]"
            res.color = "magenta"

    if res.in_use and not res.reserved:
        res.color = "red"
        res.err_msg = "[Use without reservation]"
        res.error = True

    return res


@attr.s(auto_attribs=True)
class ProcessParseResult:
    color: Optional[str] = None
    error: bool = False
    err_msg: str = ""


def parse_process(gpu: GPU, proc: GPUProcess) -> ProcessParseResult:
    res = ProcessParseResult()

    if proc.slurm_job is not None and proc.slurm_job_id != gpu.slurm_job_id:
        res.color = "red"
        res.err_msg = "[Wrong Job (" + str(proc.slurm_job_id) + ")]"
        res.error = True

    if proc.slurm_job is None:
        res.color = "red"
        res.err_msg = "[No Job]"
        res.error = True

    return res
