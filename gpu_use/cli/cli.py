import argparse
import os
import platform
import shlex
import subprocess
import sys
from time import gmtime, strftime

import click
from termcolor import colored, cprint

from gpu_use.db.db_schema import GPU, Node, Process
from gpu_use.db.session import SessionMaker

#  DEBUG_CHECK = "scontrol show job {} 2>/dev/null | grep \"Partition\" | awk -F'[ =]+' '{print $3 == \"debug\"}'"


@click.command(name="gpu-use")
@click.option(
    "-n",
    "--node",
    type=str,
    default=".*",
    show_default=True,
    help="Specify a specific node -- regex enabled",
)
@click.option(
    "-u",
    "--user",
    type=str,
    default=".*",
    show_default=True,
    help="Specify a specific user -- regex enabled",
)
@click.option("-d", "--dense", default=False, is_flag=True)
@click.option("-e", "--error", default=False, is_flag=True)
def gpu_use_cli(node, user, dense, error):
    session = SessionMaker()

    processes = (
        session.query(Process)
        .filter(Process.node_name.op("regexp")(node) & Process.user.op("regexp")(user))
        .all()
    )
    nodes = [proc.node for proc in processes]

    if dense:
        for node in nodes:
            if not error:
                click.echo(
                    click.style(
                        "-------------------------------------------------------------------\n"
                        + node.name
                        + "\n-------------------------------------------------------------------",
                        fg="white",
                        bold=True,
                    )
                )

            for gpu in node.gpus:
                reserved = False
                in_use = False
                valid_use = False
                error = False
                err_msg = ""

                if any(proc.slurm_job_id is not None for proc in gpu.processes):
                    reserved = True

                if len(pid_res) > 0:
                    in_use = True

                if all(
                    [str(c) == str(item["jid"]) for c in ([x["jid"] for x in pid_res])]
                ):
                    valid_use = True

                res_char = u"\u25A1".encode("utf_8")
                use_char = u"\u25A1".encode("utf_8")
                res_record = "-"
                color = "white"

                if reserved:
                    res_char = u"\u25A0".encode("utf_8")
                    res_record = item["uid"] + " (" + item["jid"] + ")"

                if in_use:
                    use_char = u"\u25A0".encode("utf_8")

                if reserved and in_use and not valid_use:
                    color = "red"
                    err_msg = ""
                    # [See processes]'
                    error = True

                if reserved and not in_use:
                    color = "red"
                    err_msg = "[Idle reservation]"
                    error = True
                    try:
                        if int(os.popen(DEBUG_CHECK % int(item["jid"])).read().strip()):
                            err_msg = "[Idle reservation - DEBUG]"
                            color = "magenta"
                    except ValueError:
                        pass

                if in_use and not reserved:
                    color = "red"
                    err_msg = "[Use without reservation]"
                    error = True

                if not args.error or (args.error and error):
                    gpu_record = colored(
                        res_char + use_char + "[" + str(item["gpu"]) + "]", color
                    )
                    res_record = colored(res_record, color)
                    err_msg = colored(err_msg, color)

                    if args.error:
                        print(colored(node, color), gpu_record, res_record, err_msg)
                    else:
                        print(gpu_record, res_record, err_msg)

                    for proc in pid_res:
                        color = "white"
                        error = False
                        err_msg = ""

                        if not str(proc["jid"]) == item["jid"]:
                            color = "red"
                            err_msg = "[Wrong Job (" + str(proc["jid"]) + ")]"
                            error = True

                        if proc["jid"] == -1:
                            color = "red"
                            err_msg = "[No Job]"
                            error = True

                        if not args.error or (args.error and error):
                            proc_record = colored(
                                "       + "
                                + str(proc["pid"])
                                + " "
                                + proc["command"]
                                + " "
                                + proc["uid"]
                                + " "
                                + err_msg,
                                color,
                            )
                            print(proc_record)

            if not args.error:
                print("\n")
    else:
        for node in nodes:
            gpu_tot = 0
            gpu_res = 0
            gpu_used = 0
            sys.stdout.write(colored(node, "white", attrs=["bold"]))

            alloc_gpus = sorted(
                gpu_db.search((q.node.matches(node)) & (q.uid.matches(args.user))),
                key=lambda r: r["gpu"],
            )
            pid_gpus = sorted(
                pid_db.search((q.node.matches(node)) & (q.uid.matches(args.user))),
                key=lambda r: r["gpu"],
            )

            gpus = list(
                set([str(x["gpu"]) for x in alloc_gpus]).union(
                    set([str(x["gpu"]) for x in pid_gpus])
                )
            )
            gpu_records = sorted(
                gpu_db.search((q.node.matches(node)) & (q.gpu.one_of(gpus))),
                key=lambda r: r["gpu"],
            )

            for item in gpu_records:
                if str(item["gpu"]) == "NoDevFiles":
                    continue
                gpu_tot = gpu_tot + 1
                pid_res = pid_db.search(
                    (q.node == item["node"])
                    & (q.gpu == int(item["gpu"]))
                    & (q.uid.matches(args.user))
                )

                reserved = False
                in_use = False
                valid_use = False
                error = False
                err_msg = ""

                if not item["jid"] == "-":
                    reserved = True
                    gpu_res = gpu_res + 1

                if len(pid_res) > 0:
                    in_use = True
                    gpu_used = gpu_used + 1

                if all(
                    [str(c) == str(item["jid"]) for c in ([x["jid"] for x in pid_res])]
                ):
                    valid_use = True

                res_char = u"\u25A1".encode("utf_8")
                use_char = u"\u25A1".encode("utf_8")
                color = "white"

                if reserved:
                    res_char = u"\u25A0".encode("utf_8")

                if in_use:
                    use_char = u"\u25A0".encode("utf_8")

                if reserved and in_use and not valid_use:
                    color = "red"

                if reserved and not in_use:
                    color = "red"
                    try:
                        if int(os.popen(DEBUG_CHECK % int(item["jid"])).read().strip()):
                            color = "magenta"
                    except ValueError:
                        pass

                if in_use and not reserved:
                    color = "red"

                gpu_record = colored(
                    "\t" + res_char + use_char + "[" + str(item["gpu"]) + "]", color
                )
                sys.stdout.write(gpu_record)
                # node_string = node_string+'\t'+gpu_record

            if node == "hal":
                sys.stdout.write("\t")
            sys.stdout.write(
                "\t" + str(gpu_used) + " / " + str(gpu_res) + " / " + str(gpu_tot)
            )
            print("\n")


# 	cprint(u"\u25A0".encode('utf_8')+ " [1]",'red')
# 	cprint(u"\u25A0".encode('utf_8'),'white')
# 	cprint(u"\u25A1".encode('utf_8'),'white')
# 	cprint(u"\u25A1".encode('utf_8'),'red')


# 	for item in sorted([x in gpu_states if x['node'] == node], key = lambda r: (r['node'], r['gpu'])):
# 		cprint(item, 'green')

# Collect records for pretty printing
# records = [];
# node = ""
# for item in sorted(gpu_states, key = lambda r: (r['node'], r['gpu'])):
# 	if not item['node'] == node:
# 			records.append([])
# 			node = item['node'];

# 	item_record = [item['node'], item['gpu'], item['jid'], item['uid'], "", "", ""]
# 	records.append(item_record);

# 	# lookup processes running on this node
# 	pid_res = pid_db.search((q.node == item['node']) & (q.gpu == int(item['gpu'])))

# 	for proc in pid_res:
# 		pid = str(proc['pid'])
# 		if proc['jid'] == -1:
# 			pid = pid + " (NO JOB)"
# 		else:
# 			pid = pid + " (" + str(proc['jid'])+")"

# 		item_record = ["", "", "", "|--------------->", pid,proc['uid'], proc['command']]
# 		records.append(item_record)


# #records.append(['========','========','========','========','========','========','========'])

# # Print our pretty table

# print tabulate(records[1:], ['Node', 'GPU', 'Alloc Job', 'Alloc User', "PID (Job)", "User", "Command"], tablefmt="simple")
