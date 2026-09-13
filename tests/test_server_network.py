from __future__ import annotations

from mfq.server import network


def test_system_proxy_environment_discovers_os_proxy_and_bypasses_loopback(monkeypatch) -> None:
    monkeypatch.setattr(
        network,
        "getproxies",
        lambda: {
            "http": "http://proxy.test:8080",
            "https": "http://proxy.test:8080",
            "no": "*.internal.test",
        },
    )

    environment = network.system_proxy_environment({})

    assert environment["HTTP_PROXY"] == "http://proxy.test:8080"
    assert environment["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert environment["NO_PROXY"] == "*.internal.test,127.0.0.1,localhost,::1"
    assert environment["no_proxy"] == environment["NO_PROXY"]


def test_system_proxy_environment_preserves_explicit_proxy_and_bypass(monkeypatch) -> None:
    monkeypatch.setattr(
        network,
        "getproxies",
        lambda: {"https": "http://system-proxy.test:8080"},
    )

    environment = network.system_proxy_environment(
        {
            "HTTPS_PROXY": "http://explicit-proxy.test:9090",
            "NO_PROXY": "*.example.test,localhost",
        }
    )

    assert environment["HTTPS_PROXY"] == "http://explicit-proxy.test:9090"
    assert environment["https_proxy"] == "http://explicit-proxy.test:9090"
    assert environment["NO_PROXY"] == "*.example.test,localhost,127.0.0.1,::1"
