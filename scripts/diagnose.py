# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MODULE DOC]
# Environment diagnostics CLI: prints Python, pip and key package versions, verl install info,
# OS/hardware details, selected environment variables, CUDA versions, CPU/GPU memory, and
# optionally tests network connectivity to PyPI (plus regional mirrors). Which sections run is
# controlled by integer flags parsed in parse_args; output goes to stdout.
"""Diagnose script for checking OS/hardware/python/pip/verl/network.
The output of this script can be a very good hint to issue/problem.
"""

import os
import platform
import socket
import subprocess
import sys
import time

import psutil

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
    from urlparse import urlparse
import argparse
import importlib.metadata

import torch

URLS = {
    "PYPI": "https://pypi.python.org/pypi/pip",
}

REGIONAL_URLS = {
    "cn": {
        "PYPI(douban)": "https://pypi.douban.com/",
        "Conda(tsinghua)": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
    }
}


def test_connection(name, url, timeout=10):
    # [FUNCTION DOC]
    # Times DNS resolution of the URL's host and then opening the URL, printing timings or an error
    # message (and returning early) for whichever step fails. Inputs: name label; url; timeout seconds
    # for the open. Outputs: None. Side effects: network I/O, prints.
    """Simple connection test"""
    urlinfo = urlparse(url)
    start = time.time()
    try:
        socket.gethostbyname(urlinfo.netloc)
    except Exception as e:
        print("Error resolving DNS for {}: {}, {}".format(name, url, e))
        return
    dns_elapsed = time.time() - start
    start = time.time()
    try:
        _ = urlopen(url, timeout=timeout)
    except Exception as e:
        print("Error open {}: {}, {}, DNS finished in {} sec.".format(name, url, e, dns_elapsed))
        return
    load_elapsed = time.time() - start
    print("Timing for {}: {}, DNS: {:.4f} sec, LOAD: {:.4f} sec.".format(name, url, dns_elapsed, load_elapsed))


def check_python():
    # [FUNCTION DOC]
    # Prints Python version, compiler, build and architecture info.
    print("----------Python Info----------")
    print("Version      :", platform.python_version())
    print("Compiler     :", platform.python_compiler())
    print("Build        :", platform.python_build())
    print("Arch         :", platform.architecture())


def check_pip():
    # [FUNCTION DOC]
    # Prints pip version and install directory, or a message if pip cannot be imported.
    print("------------Pip Info-----------")
    try:
        import pip

        print("Version      :", pip.__version__)
        print("Directory    :", os.path.dirname(pip.__file__))
    except ImportError:
        print("No corresponding pip install for current python.")


def _get_current_git_commit():
    # [FUNCTION DOC]
    # HELPER: Runs "git rev-parse HEAD" in the current working directory and returns the stripped
    # hash; on command failure or missing git, prints a message and returns None.
    # KNOWN ISSUE: uses the current working directory rather than the verl install directory.
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("Did not find command: git")
        return None


def check_verl():
    # [FUNCTION DOC]
    # Prepends the cwd to sys.path, imports verl and prints its version, directory and current git
    # commit. ImportError prints "No verl installed"; other errors print a traceback (with an extra
    # explanation unless it is an IOError). Side effects: mutates sys.path, prints.
    print("----------verl Info-----------")
    try:
        sys.path.insert(0, os.getcwd())
        import verl

        print("Version      :", verl.__version__)
        verl_dir = os.path.dirname(verl.__file__)
        print("Directory    :", verl_dir)
        try:
            commit_hash = _get_current_git_commit()
            print("Commit Hash  :", commit_hash)
        except AttributeError:
            print("Commit hash not found. ")
    except ImportError as e:
        print(f"No verl installed: {e}")
    except Exception as e:
        import traceback

        if not isinstance(e, IOError):
            print("An error occurred trying to import verl.")
            print("This is very likely due to missing or incompatible library files.")
        print(traceback.format_exc())


def check_os():
    # [FUNCTION DOC]
    # Prints platform string, system, node name, release and version.
    print("----------Platform Info----------")
    print("Platform     :", platform.platform())
    print("system       :", platform.system())
    print("node         :", platform.node())
    print("release      :", platform.release())
    print("version      :", platform.version())


def check_hardware():
    # [FUNCTION DOC]
    # Prints machine and processor, then platform-specific CPU details: on macOS filters "sysctl -a"
    # lines with brand_string/features, on Linux runs lscpu, on Windows runs wmic. Side effects: subprocesses.
    print("----------Hardware Info----------")
    print("machine      :", platform.machine())
    print("processor    :", platform.processor())
    if sys.platform.startswith("darwin"):
        pipe = subprocess.Popen(("sysctl", "-a"), stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
        for line in output.split(b"\n"):
            if b"brand_string" in line or b"features" in line:
                print(line.strip())
    elif sys.platform.startswith("linux"):
        subprocess.call(["lscpu"])
    elif sys.platform.startswith("win32"):
        subprocess.call(["wmic", "cpu", "get", "name"])


def check_network(args):
    # [FUNCTION DOC]
    # Optionally sets a global socket default timeout, merges URLs for each requested comma-separated
    # region into the global URLS mapping (warning for unknown regions), then runs test_connection for
    # each URL. Inputs: args with timeout and region. Side effects: mutates URLS, global socket timeout, network.
    # KNOWN ISSUE: prints args.timeout but sets the socket default timeout to a hardcoded 10.
    print("----------Network Test----------")
    if args.timeout > 0:
        print("Setting timeout: {}".format(args.timeout))
        socket.setdefaulttimeout(10)
    for region in args.region.strip().split(","):
        r = region.strip().lower()
        if not r:
            continue
        if r in REGIONAL_URLS:
            URLS.update(REGIONAL_URLS[r])
        else:
            import warnings

            warnings.warn("Region {} do not need specific test, please refer to global sites.".format(r), stacklevel=2)
    for name, url in URLS.items():
        test_connection(name, url, args.timeout)


def check_environment():
    # [FUNCTION DOC]
    # Prints environment variables starting with VERL_, OMP_ or KMP_, plus CC and CXX.
    print("----------Environment----------")
    for k, v in os.environ.items():
        if k.startswith("VERL_") or k.startswith("OMP_") or k.startswith("KMP_") or k == "CC" or k == "CXX":
            print('{}="{}"'.format(k, v))


def check_pip_package_versions():
    # [FUNCTION DOC]
    # Prints installed versions of vllm, sglang, ray and torch (or "not found") via package metadata.
    packages = ["vllm", "sglang", "ray", "torch"]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            print(f"{package}\t     : {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}\t     : not found.")


def check_cuda_versions():
    # [FUNCTION DOC]
    # If CUDA is available, prints torch's CUDA runtime version and the "release" line from
    # nvcc --version; reports a missing nvcc or any other error as a message. Otherwise prints that
    # CUDA is unavailable.
    if torch.cuda.is_available():
        try:
            cuda_runtime_version = torch.version.cuda
            print(f"CUDA Runtime : {cuda_runtime_version}")
            import subprocess

            nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
            cuda_compiler_version = next((line for line in nvcc_output.splitlines() if "release" in line), None)
            if cuda_compiler_version:
                print(f"CUDA Compiler : {cuda_compiler_version.strip()}")
            else:
                print("Could not determine CUDA compiler version.")
        except FileNotFoundError as e:
            print(f"CUDA compiler : Not found: {e}")
        except Exception as e:
            print(f"An error occurred while checking CUDA versions: {e}")
    else:
        print("CUDA is not available.")


def _get_cpu_memory():
    # [FUNCTION DOC]
    # HELPER: Returns total system RAM in GiB (float) via psutil.
    """
    Get the total CPU memory capacity in GB.
    """
    memory = psutil.virtual_memory()
    return memory.total / (1024**3)


def _get_gpu_info():
    # [FUNCTION DOC]
    # HELPER: Queries nvidia-smi for GPU name and total memory (MiB, CSV). Returns (count, list of
    # {type, memory in GiB}). On nvidia-smi failure or absence prints a message and returns (0, []).
    # KNOWN ISSUE: if nvidia-smi succeeds with empty output the count is 1 and parsing raises ValueError.
    """
    Get GPU type, GPU memory, and GPU count using nvidia-smi command.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_lines = result.stdout.strip().split("\n")
        gpu_count = len(gpu_lines)
        gpu_info = []
        for line in gpu_lines:
            gpu_name, gpu_memory = line.split(", ")
            gpu_info.append(
                {
                    "type": gpu_name,
                    "memory": float(gpu_memory) / 1024,  # Convert to GB
                }
            )
        return gpu_count, gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Failed to execute nvidia-smi command.")
        return 0, []


def _get_system_info():
    # [FUNCTION DOC]
    # HELPER: Combines CPU memory and GPU info into a dict with cpu_memory, gpu_count, gpu_info.
    """
    Get CPU memory capacity, GPU type, GPU memory, and GPU count.
    """
    cpu_memory = _get_cpu_memory()
    gpu_count, gpu_info = _get_gpu_info()
    return {"cpu_memory": cpu_memory, "gpu_count": gpu_count, "gpu_info": gpu_info}


def check_system_info():
    # [FUNCTION DOC]
    # Prints CPU memory, GPU count and each GPU's type and memory from _get_system_info.
    print("----------System Info----------")
    system_info = _get_system_info()
    print(f"CPU Memory\t: {system_info['cpu_memory']:.2f} GB")
    print(f"GPU Count\t: {system_info['gpu_count']}")
    for i, gpu in enumerate(system_info["gpu_info"]):
        print(f"GPU {i + 1}\tType    : {gpu['type']}")
        print(f"GPU {i + 1}\tMemory  : {gpu['memory']:.2f} GB")


def parse_args():
    # [FUNCTION DOC]
    # Builds the argument parser: integer toggles --python, --pip, --verl, --system, --os,
    # --environment (default 1), --network and --hardware (default 0), --region (comma-separated,
    # default empty) and --timeout (default 10). Outputs: parsed namespace.
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Diagnose script for checking the current system.",
    )
    choices = ["python", "pip", "verl", "system", "os", "environment"]
    for choice in choices:
        parser.add_argument("--" + choice, default=1, type=int, help="Diagnose {}.".format(choice))
    parser.add_argument("--network", default=0, type=int, help="Diagnose network.")
    parser.add_argument("--hardware", default=0, type=int, help="Diagnose hardware.")
    parser.add_argument(
        "--region",
        default="",
        type=str,
        help="Additional sites in which region(s) to test. \
                        Specify 'cn' for example to test mirror sites in China.",
    )
    parser.add_argument("--timeout", default=10, type=int, help="Connection test timeout threshold, 0 to disable.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.python:
        check_python()

    if args.pip:
        check_pip()
        check_pip_package_versions()

    if args.verl:
        check_verl()

    if args.os:
        check_os()

    if args.hardware:
        check_hardware()

    if args.network:
        check_network(args)

    if args.environment:
        check_environment()
        check_cuda_versions()

    if args.system:
        check_system_info()
