"""Lifecycle management for the private native inference worker."""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mfq.architectures.tensor_schema import graph_spec_for_source_names
from mfq.compat.legacy_model_graph import legacy_model_graph
from mfq.formats.assets import (
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    MODEL_GRAPH_ASSET,
    TOKENIZER_GGUF_ASSET,
)
from mfq.formats.io import open_mmap
from mfq.server.hf_tokenizer import (
    ensure_hf_tokenizer_gguf,
    ensure_mfq_tokenizer_gguf,
    native_hf_asset_environment,
)


class NativeRuntimeError(RuntimeError):
    """Raised when the private native worker cannot be started."""


@dataclass(frozen=True)
class RuntimeRoute:
    """Resolved runtime ownership for one model artifact."""

    architecture_family: str
    backbone: str = ""
    python_mlx_worker: bool = False
    vision_available: bool = False
    continuous_batching: bool = False


@dataclass(frozen=True)
class _RuntimeImplementationRegistration:
    backbone: str
    python_mlx_worker: bool = False
    continuous_batching: bool = False


_RUNTIME_IMPLEMENTATION_REGISTRY = (
    _RuntimeImplementationRegistration(
        backbone="qwen3_5",
        continuous_batching=True,
    ),
    _RuntimeImplementationRegistration(
        backbone="qwen4_exp",
    ),
    _RuntimeImplementationRegistration(
        backbone="glm5_next",
        python_mlx_worker=True,
    ),
)


def _normalize_architecture(architecture: str) -> str:
    return architecture.strip().lower().replace("-", "_")


def _runtime_implementation(
    backbone: str,
) -> _RuntimeImplementationRegistration | None:
    return next(
        (
            registration
            for registration in _RUNTIME_IMPLEMENTATION_REGISTRY
            if backbone == registration.backbone
        ),
        None,
    )


def _runtime_graph(architecture: str, model: str | Path) -> dict[str, object] | None:
    model_path = Path(model).expanduser().resolve()
    if model_path.is_dir():
        try:
            config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                return None
            index_path = model_path / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    return None
                source_names = tuple(str(name) for name in weight_map)
            else:
                names: list[str] = []
                for shard in sorted(model_path.glob("*.safetensors")):
                    with shard.open("rb") as stream:
                        raw_size = stream.read(8)
                        if len(raw_size) != 8:
                            return None
                        size = struct.unpack("<Q", raw_size)[0]
                        header = json.loads(stream.read(size))
                    if not isinstance(header, dict):
                        return None
                    names.extend(
                        str(name) for name in header if name != "__metadata__"
                    )
                source_names = tuple(names)
            spec = graph_spec_for_source_names(config, source_names)
            return None if spec is None else spec.as_dict()
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None
    if not model_path.is_file():
        return None
    try:
        with open_mmap(model_path) as store:
            if MODEL_GRAPH_ASSET in store.records:
                value = json.loads(store.read_blob(MODEL_GRAPH_ASSET))
                return value if isinstance(value, dict) else None
            config = None
            if MODEL_CONFIG_ASSET in store.records:
                candidate = json.loads(store.read_blob(MODEL_CONFIG_ASSET))
                if isinstance(candidate, dict):
                    config = candidate
            return legacy_model_graph(architecture, store.records, config)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def resolve_runtime_route(architecture: str, model: str | Path) -> RuntimeRoute:
    """Resolve runtime ownership from component/backbone implementation IDs."""

    graph = _runtime_graph(architecture, model)
    if graph is None:
        return RuntimeRoute(architecture_family=_normalize_architecture(architecture))
    graph_descriptor = graph.get("graph")
    backbone = (
        str(graph_descriptor.get("backbone") or "")
        if isinstance(graph_descriptor, Mapping)
        else ""
    )
    family = str(graph.get("architecture") or backbone)
    components = graph.get("components")
    component_kinds = {
        str(component.get("kind"))
        for component in components
        if isinstance(component, Mapping)
    } if isinstance(components, list) else set()
    registration = _runtime_implementation(backbone)
    if registration is None:
        return RuntimeRoute(
            architecture_family=family,
            backbone=backbone,
            vision_available="vision" in component_kinds,
        )
    return RuntimeRoute(
        architecture_family=family,
        backbone=backbone,
        python_mlx_worker=registration.python_mlx_worker,
        vision_available="vision" in component_kinds,
        continuous_batching=registration.continuous_batching,
    )


def native_request_capacity(
    *,
    backend: str,
    route: RuntimeRoute,
    routed_expert_bytes: int,
    requested: int,
) -> int:
    """Return concurrency actually supported by one native model worker."""

    if requested < 1:
        raise ValueError("requested runtime concurrency must be positive")
    if (
        backend == "cuda"
        and route.continuous_batching
        and routed_expert_bytes == 0
    ):
        return requested
    return 1


def python_mlx_runtime_command(
    controller_command: Sequence[str],
    *,
    model: str | Path,
    model_name: str,
    host: str,
    port: int,
    context_size: int,
    prefill_chunk_size: int = 2_048,
) -> list[str]:
    if not controller_command:
        raise NativeRuntimeError("the Python MLX runtime has no MFQ CLI launcher")
    return [
        *(str(value) for value in controller_command),
        "_flash-next-worker",
        "--model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--model-name",
        model_name,
        "--ctx-size",
        str(context_size),
        "--prefill-chunk-size",
        str(prefill_chunk_size),
    ]


def find_native_runtime_resource(executable: str | Path, name: str) -> Path | None:
    """Find a native resource in build, release, or application layouts."""

    executable_path = Path(executable).expanduser().resolve()
    directory = executable_path.parent
    candidates = (
        directory / name,
        directory / "lib" / name,
        directory.parent / "Resources" / name,
        directory.parent / "Frameworks" / name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def native_runtime_environment(
    executable: str | Path,
    backend: str,
    base: Mapping[str, str] | None = None,
    model: str | Path | None = None,
) -> dict[str, str]:
    """Return source assets plus any backend-specific runtime resources."""

    environment = dict(os.environ if base is None else base)
    if model is not None:
        environment.update(native_hf_asset_environment(model))
    if backend != "metal":
        return environment
    if not environment.get("MFQ_MLX_METALLIB"):
        metallib = find_native_runtime_resource(executable, "mlx.metallib")
        if metallib is not None:
            environment["MFQ_MLX_METALLIB"] = str(metallib)
    if not environment.get("MFQ_AVFOUNDATION_VIDEO_LIBRARY"):
        video_library = find_native_runtime_resource(
            executable,
            "libmfq_avfoundation_video.dylib",
        )
        if video_library is not None:
            environment["MFQ_AVFOUNDATION_VIDEO_LIBRARY"] = str(video_library)
    return environment


def reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def native_tokenizer_arguments(model: str | Path) -> list[str]:
    model_path = Path(model).expanduser().resolve()
    if model_path.is_dir():
        tokenizer = ensure_hf_tokenizer_gguf(model_path)
    elif model_path.is_file():
        with open_mmap(model_path) as store:
            has_embedded_tokenizer = (
                TOKENIZER_GGUF_ASSET in store.records
                or all(
                    name in store.records
                    for name in (
                        MODEL_CONFIG_ASSET,
                        HF_TOKENIZER_JSON_ASSET,
                        HF_TOKENIZER_CONFIG_ASSET,
                    )
                )
            )
        if not has_embedded_tokenizer:
            return []
        tokenizer = ensure_mfq_tokenizer_gguf(model_path)
    else:
        return []
    return ["--tokenizer-gguf", str(tokenizer)]


@dataclass
class NativeRuntime:
    executable: Path
    model: Path
    model_name: str
    backend: str
    context_size: int = 0
    prefill_chunk_size: int = 2048
    continuous_batching: int = 0
    routed_expert_bytes: int = 0
    startup_timeout: float = 1800.0
    environment: Mapping[str, str] | None = None
    architecture: str = ""
    controller_command: Sequence[str] = ()
    process: subprocess.Popen[bytes] | None = None
    port: int | None = None

    @property
    def base_url(self) -> str:
        if self.port is None:
            raise NativeRuntimeError("native runtime has not been started")
        return f"http://127.0.0.1:{self.port}"

    def command(self, port: int) -> list[str]:
        if self.continuous_batching < 0:
            raise NativeRuntimeError("continuous batching must be non-negative")
        if self.continuous_batching > 0 and self.backend != "cuda":
            raise NativeRuntimeError("continuous batching currently requires CUDA")
        route = resolve_runtime_route(self.architecture, self.model)
        if route.python_mlx_worker:
            if self.backend != "metal":
                raise NativeRuntimeError("the Python MLX worker currently requires Metal")
            return python_mlx_runtime_command(
                self.controller_command,
                model=self.model,
                model_name=self.model_name,
                host="127.0.0.1",
                port=port,
                context_size=self.context_size,
                prefill_chunk_size=self.prefill_chunk_size,
            )
        command = [
            str(self.executable),
            "--mfq",
            str(self.model),
            "--server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-name",
            self.model_name,
        ]
        if self.backend == "metal":
            command.extend(["--prefill-chunk-size", str(self.prefill_chunk_size)])
        if self.context_size > 0:
            command.extend(["--ctx-size", str(self.context_size)])
        request_capacity = native_request_capacity(
            backend=self.backend,
            route=route,
            routed_expert_bytes=self.routed_expert_bytes,
            requested=max(1, self.continuous_batching),
        )
        if self.continuous_batching > 0 and request_capacity > 1:
            command.extend(
                ["--continuous-batching", str(request_capacity)]
            )
        command.extend(native_tokenizer_arguments(self.model))
        return command

    def start(self) -> None:
        if self.process is not None:
            raise NativeRuntimeError("native runtime is already running")
        if not self.executable.is_file():
            raise NativeRuntimeError(f"native runtime executable does not exist: {self.executable}")
        if not (self.model.is_file() or self.model.is_dir()):
            raise NativeRuntimeError(f"model does not exist: {self.model}")
        self.port = reserve_loopback_port()
        command = self.command(self.port)
        print(
            "Starting native inference worker:",
            subprocess.list2cmdline(command),
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            env=native_runtime_environment(
                self.executable,
                self.backend,
                base={**os.environ, **dict(self.environment or {})},
                model=self.model,
            ),
        )
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        if self.process is None:
            raise NativeRuntimeError("native runtime has not been started")
        deadline = time.monotonic() + self.startup_timeout
        last_error = "worker did not accept connections"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.process = None
                raise NativeRuntimeError(
                    f"native inference worker exited during startup with status {status}"
                )
            try:
                with opener.open(f"{self.base_url}/health", timeout=1.0) as response:
                    payload = json.loads(response.read())
                if isinstance(payload, dict) and payload.get("model"):
                    return
                last_error = "worker returned an invalid health response"
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = str(error)
            time.sleep(0.25)
        self.stop()
        raise NativeRuntimeError(
            f"native inference worker did not become ready within {self.startup_timeout:g}s: "
            f"{last_error}"
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> NativeRuntime:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
