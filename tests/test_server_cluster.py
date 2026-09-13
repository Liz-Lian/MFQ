from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from uuid import UUID

import httpx

from mfq.server.api import create_app
from mfq.server.cluster import ClusterBackend
from mfq.server.models import CreateRemoteNodeRequest, UpdateRemoteNodeRequest
from mfq.server.service import ServerService
from mfq.server.storage import SessionStore
from tests.test_server_service import FakeBackend


def _remote_app(
    response_requests: list[dict[str, object]] | None = None,
    requested_paths: list[str] | None = None,
    *,
    legacy_models_endpoint: bool = False,
    runtime_status_code: int = 200,
):
    session_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal session_count
        path = request.url.path
        if requested_paths is not None:
            requested_paths.append(path)
        if path == "/health":
            return httpx.Response(200, json={"service": "mfq-server"})
        if path == "/v1/models":
            if legacy_models_endpoint:
                return httpx.Response(404)
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if path == "/api/v1/runtime/models" and legacy_models_endpoint:
            return httpx.Response(200, json={"data": [{"id": "remote-model"}]})
        if path == "/api/v1/runtime/status":
            if runtime_status_code != 200:
                return httpx.Response(runtime_status_code)
            return httpx.Response(200, json={"total_requests": 7, "process_resident_bytes": 1024})
        if path == "/api/v1/media":
            assert request.headers["content-type"] == "image/png"
            assert request.content == b"image"
            return httpx.Response(
                201,
                json={
                    "media": {
                        "id": "66666666-6666-4666-8666-666666666666",
                        "sha256": hashlib.sha256(b"image").hexdigest(),
                        "mime_type": "image/png",
                        "byte_size": 5,
                    },
                    "created_at": "2026-08-12T00:00:00Z",
                },
            )
        if path == "/api/v1/sessions":
            remote_session_id = (
                "11111111-1111-4111-8111-"
                f"{111111111111 + session_count:012d}"
            )
            session_count += 1
            return httpx.Response(
                201,
                json={
                    "id": remote_session_id,
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 0,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if path.endswith("/responses"):
            if response_requests is not None:
                response_requests.append(json.loads(request.content))
            frames = [
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 0,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.text.delta",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "delta": "remote",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 1,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {
                        "type": "response.completed",
                        "response_id": "22222222-2222-4222-8222-222222222222",
                        "finish_reason": "stop",
                    },
                },
                {
                    "protocol_version": "1.0",
                    "session_id": "11111111-1111-4111-8111-111111111111",
                    "sequence": 2,
                    "timestamp": "2026-08-12T00:00:00Z",
                    "payload": {"type": "session.state", "state": "idle", "revision": 2},
                },
            ]
            content = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
            return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
        if path.endswith("/responses/cancel"):
            assert request.method == "POST"
            return httpx.Response(200, json={"status": "cancelled"})
        if path.endswith("/fork"):
            return httpx.Response(
                201,
                json={
                    "id": "44444444-4444-4444-8444-444444444444",
                    "model": "remote-model",
                    "mode": "text",
                    "state": "idle",
                    "revision": 2,
                    "title": None,
                    "runtime_instance_id": None,
                    "created_at": "2026-08-12T00:00:00Z",
                    "updated_at": "2026-08-12T00:00:00Z",
                    "metadata": {},
                },
            )
        if request.method == "DELETE" and "/api/v1/sessions/" in path:
            return httpx.Response(204)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_cluster_registers_probes_and_routes_matching_model(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        response_requests: list[dict[str, object]] = []
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(response_requests, requested_paths)
        )
        local = FakeBackend()
        cluster = ClusterBackend(local, store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201
            assert created.json()["healthy"] is True
            assert created.json()["models"] == ["remote-model"]
            node_id = created.json()["id"]
            listed = await api.get("/api/v1/cluster/nodes?refresh=true")
            assert listed.json()["data"][0]["healthy"] is True
            assert listed.json()["data"][0]["metrics"]["total_requests"] == 7
            assert "/v1/models" in requested_paths
            assert "/api/v1/runtime/models" not in requested_paths
            models = await cluster.runtime_models()
            assert any(item["id"] == "remote-model" for item in models["data"])

            chunks = []
            async for delta in cluster.stream(
                model="remote-model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(b"image").decode("ascii")
                                },
                            }
                        ],
                    }
                ],
                sampling=__import__(
                    "mfq.server.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
                session_id=UUID("33333333-3333-4333-8333-333333333333"),
            ):
                chunks.append(delta)
            assert "".join(item.content_delta for item in chunks) == "remote"
            assert chunks[-1].finish_reason == "stop"
            assert response_requests[0]["sampling"] == {}
            replacement = [
                delta
                async for delta in cluster.stream(
                    model="remote-model",
                    messages=[{"role": "user", "content": "edited"}],
                    sampling=__import__(
                        "mfq.server.models", fromlist=["SamplingParams"]
                    ).SamplingParams(),
                    session_id=UUID("33333333-3333-4333-8333-333333333333"),
                )
            ]
            assert replacement[-1].finish_reason == "stop"
            assert requested_paths.count("/api/v1/sessions") == 2
            assert (
                "/api/v1/sessions/11111111-1111-4111-8111-111111111111"
                in requested_paths
            )
            remote = cluster._sessions[
                (
                    UUID(node_id),
                    UUID("33333333-3333-4333-8333-333333333333"),
                )
            ]
            assert remote.remote_id == UUID(
                "11111111-1111-4111-8111-111111111112"
            )
            assert await cluster.cancel_response(
                UUID("33333333-3333-4333-8333-333333333333")
            )
            state = cluster._states[UUID(node_id)]
            tool_result = await cluster._message_parts(
                state.resource,
                {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                {},
            )
            assert tool_result == [
                {
                    "type": "tool_result",
                    "call_id": "call-1",
                    "result": "done",
                    "is_error": False,
                }
            ]
            tool_call = await cluster._message_parts(
                state.resource,
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "checking",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
                {},
            )
            assert [part["type"] for part in tool_call] == ["reasoning", "tool_call"]
            assert tool_call[1]["arguments"] == {"q": "x"}
            status = await cluster.runtime_status()
            assert status["cluster_total_requests"] == 7
            assert status["cluster_process_resident_bytes"] == 1024
            forked_id = UUID("55555555-5555-4555-8555-555555555555")
            assert await cluster.fork_session(
                UUID("33333333-3333-4333-8333-333333333333"), forked_id
            )
            assert await cluster.close_session(forked_id)

            assert (await api.delete(f"/api/v1/cluster/nodes/{node_id}")).status_code == 204
            assert cluster._sessions == {}

        await client.aclose()

    asyncio.run(run())


def test_remote_node_configuration_never_persists_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REMOTE_NODE_TOKEN", "private-token")
    store = SessionStore(tmp_path / "mfq.server.sqlite3")
    node = store.create_remote_node(
        __import__(
            "mfq.server.models", fromlist=["CreateRemoteNodeRequest"]
        ).CreateRemoteNodeRequest(
            name="secure", url="https://worker.example", api_key_env="REMOTE_NODE_TOKEN"
        )
    )
    assert node.api_key_env == "REMOTE_NODE_TOKEN"
    assert b"private-token" not in (tmp_path / "mfq.server.sqlite3").read_bytes()


def test_cluster_falls_back_to_legacy_runtime_inventory(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(
                requested_paths=requested_paths,
                legacy_models_endpoint=True,
            )
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "legacy-worker", "url": "http://legacy-worker:8090"},
            )
            assert created.status_code == 201
            assert created.json()["models"] == ["remote-model"]
            assert "/v1/models" in requested_paths
            assert "/api/v1/runtime/models" in requested_paths

        await client.aclose()

    asyncio.run(run())


def test_stateless_remote_stream_releases_ephemeral_session(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        requested_paths: list[str] = []
        client = httpx.AsyncClient(
            transport=_remote_app(requested_paths=requested_paths)
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201

            chunks = [
                delta
                async for delta in cluster.stream(
                    model="remote-model",
                    messages=[{"role": "user", "content": "hello"}],
                    sampling=__import__(
                        "mfq.server.models", fromlist=["SamplingParams"]
                    ).SamplingParams(),
                )
            ]
            assert chunks[-1].finish_reason == "stop"
            assert cluster._sessions == {}
            assert (
                "/api/v1/sessions/11111111-1111-4111-8111-111111111111"
                in requested_paths
            )

            interrupted = cluster.stream(
                model="remote-model",
                messages=[{"role": "user", "content": "stop early"}],
                sampling=__import__(
                    "mfq.server.models", fromlist=["SamplingParams"]
                ).SamplingParams(),
            )
            assert (await anext(interrupted)).content_delta == "remote"
            await interrupted.aclose()
            assert cluster._sessions == {}

        await client.aclose()

    asyncio.run(run())


def test_cluster_routes_when_remote_metrics_are_unavailable(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        client = httpx.AsyncClient(
            transport=_remote_app(runtime_status_code=404)
        )
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            assert created.status_code == 201
            assert created.json()["healthy"] is True
            assert created.json()["models"] == ["remote-model"]
            assert created.json()["metrics"] == {}

        await client.aclose()

    asyncio.run(run())


def test_cluster_serializes_probe_with_node_configuration_changes(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        old_probe_started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            path = request.url.path
            if path == "/health":
                if host == "old-worker":
                    old_probe_started.set()
                    await asyncio.sleep(0.05)
                return httpx.Response(200)
            if path == "/v1/models":
                model = "old-model" if host == "old-worker" else "new-model"
                return httpx.Response(200, json={"data": [{"id": model}]})
            if path == "/api/v1/runtime/status":
                return httpx.Response(200, json={})
            return httpx.Response(404)

        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        resource = store.create_remote_node(
            CreateRemoteNodeRequest(
                name="worker-a",
                url="http://old-worker:8090",
            )
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cluster = ClusterBackend(FakeBackend(), store, client=client)

        old_refresh = asyncio.create_task(cluster.nodes(force=True))
        await old_probe_started.wait()
        await asyncio.to_thread(
            store.update_remote_node,
            resource.id,
            UpdateRemoteNodeRequest(
                name="worker-a",
                url="http://new-worker:8090",
            ),
        )
        new_refresh = asyncio.create_task(cluster.nodes(force=True))
        await asyncio.gather(old_refresh, new_refresh)

        [node] = await cluster.nodes()
        assert node.url == "http://new-worker:8090"
        assert node.models == ["new-model"]
        await client.aclose()

    asyncio.run(run())


def test_disabled_remote_node_stops_advertising_stale_models(tmp_path: Path) -> None:
    async def run() -> None:
        store = SessionStore(tmp_path / "mfq.server.sqlite3")
        client = httpx.AsyncClient(transport=_remote_app())
        cluster = ClusterBackend(FakeBackend(), store, client=client)
        service = ServerService(store, cluster, cluster=cluster)
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/v1/cluster/nodes",
                json={"name": "worker-a", "url": "http://worker-a:8090"},
            )
            node_id = created.json()["id"]
            assert created.json()["models"] == ["remote-model"]

            disabled = await api.put(
                f"/api/v1/cluster/nodes/{node_id}",
                json={
                    "name": "worker-a",
                    "url": "http://worker-a:8090",
                    "enabled": False,
                },
            )
            assert disabled.status_code == 200
            assert disabled.json()["healthy"] is False
            assert disabled.json()["models"] == []
            models = await cluster.runtime_models()
            assert all(item["id"] != "remote-model" for item in models["data"])

        await client.aclose()

    asyncio.run(run())
