"""Who's on my GPU?

Usage: who_s_on_my_gpu.py [--verbose]
"""
from docopt import docopt
import subprocess
import psutil
import xmltodict
import pandas as pd
import tabulate
import re


def call_nvidia_smi():
    """
        Call nvidia-smi.
    :return OrderedDict: nvidia-smi result
    """
    sp = subprocess.Popen(['nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    xml = sp.communicate()[0].decode("utf-8")
    return xmltodict.parse(xml)


def get_processes(info):
    """
        Get processes from nvidia information.
    :param dict info: nvidia-smi
    :return DataFrame: processes using gpus
    """
    nvidia_df = pd.DataFrame()
    for gpu in to_list(info["nvidia_smi_log"]["gpu"]):
        if gpu["processes"]:
            gpu_processes = to_list(gpu["processes"]["process_info"])
            gpu_processes = pd.concat([pd.DataFrame(d, index=[0]) for d in gpu_processes]).reset_index()
            gpu_processes["gpu"] = gpu["minor_number"]
            gpu_processes["pid"] = gpu_processes["pid"].astype(int)
            gpu_processes["used_memory (Mb)"] = gpu_processes["used_memory"].map(get_int)
            gpu_processes["gpu_util (%)"] = get_int(gpu["utilization"]["gpu_util"])
            gpu_processes["memory_util (%)"] = get_int(gpu["utilization"]["memory_util"])
            nvidia_df = nvidia_df.append(gpu_processes, sort=False)
    return nvidia_df


def update_process_users(processes):
    """
        Query and update the users of processes
    :param DataFrame processes: must have an int 'pid' field
    """
    pid_users = {}
    for process in psutil.process_iter():
        try:
            pid_users[process.pid] = process.username()
        except (PermissionError, psutil.AccessDenied):
            pass
    processes["user"] = processes["pid"].map(pid_users)


def summarize(processes):
    """
        Summarize the description of processes.
        For now, a summation over compute and memory usage, grouped by gpu and user.
    :param DataFrame processes: must have 'gpu' and 'user' fields.
    :return DataFrame: the summarized processes
    """
    return processes.groupby(["gpu", "user"], as_index=False).sum().drop(columns=["index", "pid"])


def to_list(x):
    """
        Converts non-lists to singletons
    :param x: any argument
    :return list: x seen as a list
    """
    return x if isinstance(x, list) else [x]


def get_int(s):
    """
        Get the first int in a string
    :param str s: a string
    :return int: the first integer
    """
    matches = re.findall(r'\d+', s)
    return int(matches[0]) if matches else s


def main(args):
    nvidia_smi_info = call_nvidia_smi()
    processes = get_processes(nvidia_smi_info)
    update_process_users(processes)
    if not args['--verbose']:
        processes = summarize(processes)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(tabulate.tabulate(processes, headers='keys', tablefmt='psql', showindex=False))


if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
