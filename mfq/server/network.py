"""Network environment helpers shared by hub discovery and download jobs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.request import getproxies

_LOCAL_BYPASS = ("127.0.0.1", "localhost", "::1")


def system_proxy_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment that honors OS proxies without proxying loopback traffic."""

    environment = dict(os.environ if base is None else base)
    proxies = getproxies()
    for scheme in ("http", "https"):
        lower = f"{scheme}_proxy"
        upper = lower.upper()
        configured = environment.get(lower) or environment.get(upper)
        discovered = proxies.get(scheme)
        value = configured or discovered
        if value:
            environment.setdefault(lower, value)
            environment.setdefault(upper, value)

    bypass: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        bypass.extend(
            value.strip()
            for value in environment.get(name, "").split(",")
            if value.strip()
        )
    bypass.extend(
        value.strip()
        for value in str(proxies.get("no") or "").split(",")
        if value.strip()
    )
    for value in _LOCAL_BYPASS:
        if value not in bypass:
            bypass.append(value)
    serialized = ",".join(dict.fromkeys(bypass))
    environment["NO_PROXY"] = serialized
    environment["no_proxy"] = serialized
    return environment


def install_system_proxy_environment() -> None:
    """Install discovered proxy values once for in-process HTTP clients."""

    resolved = system_proxy_environment()
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        if value := resolved.get(name):
            os.environ.setdefault(name, value)
    os.environ["NO_PROXY"] = resolved["NO_PROXY"]
    os.environ["no_proxy"] = resolved["no_proxy"]
