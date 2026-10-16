# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
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
# Small networking helpers: IP address family detection and obtaining free TCP ports.
import ipaddress
import socket


def is_ipv4(ip_str: str) -> bool:
    # [FUNCTION DOC]
    # Behavior: tries to parse the string as an IPv4 address.
    # Inputs: ip_str - candidate address string.
    # Outputs: True if parseable as IPv4, else False.
    # Exceptions: none raised (AddressValueError is caught).
    # KNOWN ISSUE: only AddressValueError is caught; a non-str input (e.g. None) may raise a different error.
    # Side effects: none.
    """
    Check if the given string is an IPv4 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv4 address, False otherwise
    """
    try:
        ipaddress.IPv4Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_ipv6(ip_str: str) -> bool:
    # [FUNCTION DOC]
    # Behavior: tries to parse the string as an IPv6 address.
    # Inputs: ip_str - candidate address string.
    # Outputs: True if parseable as IPv6, else False.
    # Exceptions: none raised for invalid strings (AddressValueError caught).
    # Side effects: none.
    """
    Check if the given string is an IPv6 address

    Args:
        ip_str: The IP address string to check

    Returns:
        bool: Returns True if it's an IPv6 address, False otherwise
    """
    try:
        ipaddress.IPv6Address(ip_str)
        return True
    except ipaddress.AddressValueError:
        return False


def is_valid_ipv6_address(address: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Same as is_ipv6 but catches the broader ValueError.
    # Inputs: address - candidate string. Outputs: True if valid IPv6, else False.
    # Exceptions: none for invalid strings. Side effects: none.
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_free_port(address: str, with_alive_sock: bool = False) -> tuple[int, socket.socket | None]:
    # [FUNCTION DOC]
    # Behavior: creates a TCP socket of family IPv6 if the address is valid IPv6, else IPv4; sets
    # SO_REUSEADDR, binds to (address, 0) so the OS picks a port, and reads the assigned port.
    # If with_alive_sock is True, returns the still-open socket; otherwise closes it.
    # Inputs: address - local address to bind; with_alive_sock - keep socket open as a reservation.
    # Outputs: tuple (port, socket or None).
    # Exceptions: OSError if binding fails (e.g. address not local).
    # Side effects: opens (and possibly closes) a socket; caller must close the returned socket.
    """Find a free port on the given address.

    By default the socket is closed internally, suitable for immediate use.
    Set with_alive_sock=True to keep the socket open as a port reservation,
    preventing other calls from getting the same port. The caller is
    responsible for closing the socket before the port is actually bound
    by the target service (e.g. NCCL, uvicorn).
    """
    family = socket.AF_INET6 if is_valid_ipv6_address(address) else socket.AF_INET

    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((address, 0))
    port = sock.getsockname()[1]
    if with_alive_sock:
        return port, sock
    sock.close()
    return port, None
