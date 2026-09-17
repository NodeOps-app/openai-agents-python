from __future__ import annotations

import asyncio
import io
import json
import queue
import subprocess
import sys
import tarfile
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agents.extensions.sandbox.createos import (
    DEFAULT_CREATEOS_WORKSPACE_ROOT,
    CreateOSSandboxClient,
    CreateOSSandboxClientOptions,
    CreateOSSandboxSession,
    CreateOSSandboxSessionState,
    CreateOSSandboxTimeouts,
)
from agents.sandbox import Manifest
from agents.sandbox.errors import (
    ErrorCode,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    SandboxRuntimeError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
)
from agents.sandbox.snapshot import LocalSnapshot, NoopSnapshot
from agents.sandbox.types import ExecResult


@dataclass
class _Options:
    timeout: float | None = None
    environment_variables: dict[str, str] | None = None
    after_sequence: int = 0


class _Request:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


@dataclass
class _CommandResult:
    standard_output: str = ""
    standard_error: str = ""
    exit_code: int = 0
    error_message: str = ""


@dataclass
class _CommandResponse:
    result: _CommandResult


class _APIError(Exception):
    def __init__(self, status_code: int, message: str = "provider error") -> None:
        super().__init__(message)
        self.status_code = status_code


class _OperationTimeout(TimeoutError):
    pass


class _Download(io.BytesIO):
    def __enter__(self) -> _Download:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _FakeFiles:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.uploads: list[tuple[str, bytes, object]] = []
        self.downloads: list[tuple[str, object]] = []

    def upload(self, path: str, data: bytes, options: object) -> None:
        self.uploads.append((path, data, options))
        self.data[path] = data

    def download(self, path: str, options: object) -> _Download:
        self.downloads.append((path, options))
        if path not in self.data:
            raise _APIError(404, "not found")
        return _Download(self.data[path])


@dataclass
class _ProcessEvent:
    type: str
    data: bytes = b""
    exit_code: int | None = None
    error_message: str = ""
    sequence: int = 0


class _FakeProcessStream:
    _END = object()

    def __init__(self, events: queue.Queue[object], *, timeout: float | None = None) -> None:
        self._events = events
        self._closed = False
        self._response = types.SimpleNamespace(
            request=types.SimpleNamespace(
                extensions={
                    "timeout": {
                        "connect": timeout,
                        "read": timeout,
                        "write": timeout,
                        "pool": timeout,
                    }
                }
            )
        )

    def __iter__(self) -> _FakeProcessStream:
        return self

    def __next__(self) -> _ProcessEvent:
        event = self._events.get(timeout=5)
        if event is self._END:
            self._closed = True
            raise StopIteration
        assert isinstance(event, _ProcessEvent)
        return event

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._events.put(self._END)

    def __enter__(self) -> _FakeProcessStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _FakeProcesses:
    def __init__(self, lifecycle_events: list[str]) -> None:
        self._lifecycle_events = lifecycle_events
        self.created_requests: list[object] = []
        self.inputs: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self._events: dict[str, queue.Queue[object]] = {}

    def create(self, request: object) -> _Request:
        self.created_requests.append(request)
        process_id = f"process-{len(self.created_requests)}"
        self._events[process_id] = queue.Queue()
        return _Request(process_id=process_id)

    def connect(self, process_id: str, _options: object) -> _FakeProcessStream:
        return _FakeProcessStream(
            self._events[process_id],
            timeout=getattr(_options, "timeout", None),
        )

    def input(self, process_id: str, data: str) -> int:
        self.inputs.append((process_id, data))
        events = self._events[process_id]
        if data == "exit\n":
            events.put(_ProcessEvent(type="data", data=b"terminal-done\n"))
            events.put(_ProcessEvent(type="exit", exit_code=0))
            events.put(_FakeProcessStream._END)
        else:
            events.put(_ProcessEvent(type="data", data=b"terminal-ready\n"))
        return len(self.inputs)

    def delete(self, process_id: str) -> _Request:
        self._lifecycle_events.append("process_delete")
        self.deleted.append(process_id)
        events = self._events.get(process_id)
        if events is not None:
            events.put(_ProcessEvent(type="exit", exit_code=0))
            events.put(_FakeProcessStream._END)
        return _Request(process_id=process_id, exit_code=0)


def _tar_bytes() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        payload = b"persisted\n"
        info = tarfile.TarInfo("state.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class _FakeSandbox:
    def __init__(self, sandbox_id: str = "sb-1", *, status: str = "running") -> None:
        self.id = sandbox_id
        self.status = status
        self.files = _FakeFiles()
        self.lifecycle_events: list[str] = []
        self.processes = _FakeProcesses(self.lifecycle_events)
        self.requests: list[tuple[object, object]] = []
        self.next_result: _CommandResult | None = None
        self.next_error: Exception | None = None
        self.paused = False
        self.resumed = False
        self.destroyed = False
        self.pause_calls = 0
        self.destroy_calls = 0
        self.wait_until_paused_calls = 0
        self.wait_until_destroyed_calls = 0
        self.pause_errors: list[Exception] = []
        self.destroy_errors: list[Exception] = []
        self.wait_until_paused_errors: list[Exception] = []
        self.wait_until_destroyed_errors: list[Exception] = []
        self.waited_until_destroyed = False
        self.destroy_wait_error_status: str | None = None

    def run_command(self, request: object, options: object) -> _CommandResponse:
        self.requests.append((request, options))
        if self.next_error is not None:
            error = self.next_error
            self.next_error = None
            raise error
        if self.next_result is not None:
            result = self.next_result
            self.next_result = None
            return _CommandResponse(result)

        command = tuple(request.arguments[4:])
        if command and "resolve-workspace-path-" in command[0]:
            return _CommandResponse(_CommandResult(standard_output=f"{command[2]}\n"))
        command_text = " ".join(command)
        if "createos-persist-" in command_text and " -cf " in command_text:
            tar_path = next(part for part in command_text.split() if "createos-persist-" in part)
            self.files.data[tar_path] = _tar_bytes()
        return _CommandResponse(_CommandResult())

    def refresh(self) -> _FakeSandbox:
        return self

    def preview_url(self, port: int) -> str:
        return f"https://sandbox.example.test/{port}"

    def pause(self) -> _FakeSandbox:
        self.lifecycle_events.append("pause")
        self.pause_calls += 1
        if self.pause_errors:
            raise self.pause_errors.pop(0)
        self.paused = True
        self.status = "paused"
        return self

    def resume(self) -> _FakeSandbox:
        self.resumed = True
        self.status = "running"
        return self

    def destroy(self) -> None:
        self.lifecycle_events.append("destroy")
        self.destroy_calls += 1
        if self.destroy_errors:
            raise self.destroy_errors.pop(0)
        self.destroyed = True
        self.status = "destroyed"

    def wait_until_paused(self, _options: object) -> _FakeSandbox:
        self.wait_until_paused_calls += 1
        if self.wait_until_paused_errors:
            raise self.wait_until_paused_errors.pop(0)
        self.status = "paused"
        return self

    def wait_until_running(self, _options: object) -> _FakeSandbox:
        self.status = "running"
        return self

    def wait_until_destroyed(self, _options: object) -> _FakeSandbox:
        self.wait_until_destroyed_calls += 1
        self.waited_until_destroyed = True
        if self.wait_until_destroyed_errors:
            raise self.wait_until_destroyed_errors.pop(0)
        if self.destroy_wait_error_status is not None:
            self.status = self.destroy_wait_error_status
            raise RuntimeError(f"destruction settled as {self.status}")
        self.status = "destroyed"
        return self


class _FakeClient:
    current: _FakeClient | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.created_requests: list[tuple[object, object]] = []
        self.sandboxes: dict[str, _FakeSandbox] = {}
        self.closed = False
        type(self).current = self

    def create_sandbox(self, request: object, options: object) -> _FakeSandbox:
        self.created_requests.append((request, options))
        sandbox = _FakeSandbox(f"sb-{len(self.created_requests)}")
        self.sandboxes[sandbox.id] = sandbox
        return sandbox

    def get_sandbox(self, sandbox_id: str) -> _FakeSandbox:
        if sandbox_id not in self.sandboxes:
            raise _APIError(404, "not found")
        return self.sandboxes[sandbox_id]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_createos(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    sdk = types.SimpleNamespace(
        Client=_FakeClient,
        CreateSandboxRequest=_Request,
        RunCommandRequest=_Request,
        RequestOptions=_Options,
        ExecOptions=_Options,
        WaitOptions=_Options,
        ManagedProcessConnectOptions=_Options,
        ManagedProcessCreateRequest=_Request,
        NetworkEntry=_Request,
        PTYSize=_Request,
        OperationTimeout=_OperationTimeout,
        APIError=_APIError,
    )
    monkeypatch.setitem(sys.modules, "createos", sdk)
    return sdk


def _state(*, pause_on_exit: bool = False) -> CreateOSSandboxSessionState:
    return CreateOSSandboxSessionState(
        snapshot=NoopSnapshot(id="test-snapshot"),
        manifest=Manifest(root=DEFAULT_CREATEOS_WORKSPACE_ROOT),
        sandbox_id="sb-1",
        shape="s-4vcpu-4gb",
        rootfs="devbox:1",
        pause_on_exit=pause_on_exit,
    )


def test_package_re_exports_createos_symbols() -> None:
    from agents.extensions import sandbox as package

    assert package.CreateOSSandboxClient is CreateOSSandboxClient
    assert package.CreateOSSandboxSession is CreateOSSandboxSession


def test_package_omits_createos_exports_without_optional_dependency() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['createos'] = None; "
            "from agents.extensions import sandbox; "
            "assert not any('CreateOS' in name or 'CREATEOS' in name "
            "for name in sandbox.__all__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_options_are_positional_and_round_trip() -> None:
    options = CreateOSSandboxClientOptions(
        "s-4vcpu-4gb",
        "devbox:1",
        {"BASE": "1"},
        True,
        "demo",
        (8080,),
        network_ids=("network-1",),
        disk_mib=8192,
        egress_rules=("tcp:443",),
        ssh_public_keys=("ssh-ed25519 test",),
        host_id="host-1",
        node_selector={"pool": "gpu"},
        region="us-east",
        auto_pause_after_seconds=300,
    )
    restored = CreateOSSandboxClientOptions.model_validate(options.model_dump())
    assert restored == options
    assert restored.type == "createos"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": "   "},
        {"shape": "s-4vcpu-4gb", "disk_mib": 0},
        {"shape": "s-4vcpu-4gb", "auto_pause_after_seconds": 0},
        {"shape": "s-4vcpu-4gb", "exposed_ports": (0,)},
        {"shape": "s-4vcpu-4gb", "exposed_ports": (65_536,)},
    ],
)
def test_options_reject_invalid_js_parity_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CreateOSSandboxClientOptions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", [None, "caller-name", ""])
async def test_create_uses_provider_compatible_name(name: str | None) -> None:
    client = CreateOSSandboxClient()
    session = await client.create(
        options=CreateOSSandboxClientOptions("s-1vcpu-2gb", "devbox:1", name=name)
    )

    sdk_client = _FakeClient.current
    assert sdk_client is not None
    request, _ = sdk_client.created_requests[0]
    assert request.name == session.state.name
    if name is None:
        assert request.name
        assert len(request.name) <= 22
    else:
        assert request.name == name


@pytest.mark.parametrize("rootfs", [None, "", "   "])
async def test_create_requires_rootfs_before_provider_effects(rootfs: str | None) -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None

    with pytest.raises(ValueError, match="non-empty rootfs"):
        await client.create(options=CreateOSSandboxClientOptions("s-1vcpu-2gb", rootfs))

    assert sdk_client.created_requests == []


async def test_example_passes_current_createos_api_key_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.sandbox.extensions import createos_runner

    monkeypatch.setenv("OPENAI_API_KEY", "model-key")
    monkeypatch.setenv("CREATEOS_API_KEY", "provider-key")
    monkeypatch.delenv("CREATEOS_SANDBOX_API_KEY", raising=False)
    run = AsyncMock(return_value=types.SimpleNamespace(final_output="ok"))
    monkeypatch.setattr(createos_runner.Runner, "run", run)

    await createos_runner.main(
        model="test-model",
        question="Inspect the workspace",
        shape="s-1vcpu-2gb",
        rootfs="devbox:1",
        pause_on_exit=False,
        stream=False,
    )

    sdk_client = _FakeClient.current
    assert sdk_client is not None
    assert sdk_client.kwargs["api_key"] == "provider-key"
    assert sdk_client.closed is True
    assert run.await_count == 1


async def test_create_environment_is_live_only_and_resume_rebinds_trusted_values() -> None:
    creator = CreateOSSandboxClient(env_vars={"API_TOKEN": "initial-secret"})
    session = await creator.create(options=CreateOSSandboxClientOptions("s-1vcpu-2gb", "devbox:1"))
    original_sandbox = session._inner._sandbox
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    request, _ = sdk_client.created_requests[0]
    assert request.environment_variables == {"API_TOKEN": "initial-secret"}

    payload = creator.serialize_session_state(session.state)
    assert "initial-secret" not in json.dumps(payload)
    assert "base_env_vars" not in payload
    restored = creator.deserialize_session_state(
        {**payload, "base_env_vars": {"API_TOKEN": "untrusted-secret"}}
    )
    assert isinstance(restored, CreateOSSandboxSessionState)
    assert restored._base_env_vars == {}

    resumed_client = CreateOSSandboxClient(env_vars={"API_TOKEN": "rotated-secret"})
    resumed_sdk_client = _FakeClient.current
    assert resumed_sdk_client is not None
    resumed_sdk_client.sandboxes[original_sandbox.id] = original_sandbox
    resumed = await resumed_client.resume(restored)
    assert resumed.state is not restored
    assert restored._base_env_vars == {}
    result = await resumed.exec("printenv", "API_TOKEN", shell=False)
    assert result.ok()
    _, exec_options = original_sandbox.requests[-1]
    assert exec_options.environment_variables == {"API_TOKEN": "rotated-secret"}

    recreated_state = resumed_client.deserialize_session_state(payload)
    resumed_sdk_client.sandboxes.pop(original_sandbox.id)
    recreated = await resumed_client.resume(recreated_state)
    recreated_request, _ = resumed_sdk_client.created_requests[-1]
    assert recreated_request.environment_variables == {"API_TOKEN": "rotated-secret"}
    assert recreated._inner._sandbox is not original_sandbox


async def test_create_start_exec_and_file_transfer() -> None:
    client = CreateOSSandboxClient(api_key="secret", base_url="https://api.example.test")
    session = await client.create(
        manifest=Manifest(root=DEFAULT_CREATEOS_WORKSPACE_ROOT),
        options=CreateOSSandboxClientOptions(
            "s-4vcpu-4gb",
            "devbox:1",
            {"BASE": "1"},
            exposed_ports=(8080,),
            network_ids=("network-1", "network-2"),
            disk_mib=8192,
            egress_rules=("tcp:443",),
            ssh_public_keys=("ssh-ed25519 test",),
            host_id="host-1",
            node_selector={"pool": "gpu"},
            region="us-east",
            auto_pause_after_seconds=300,
        ),
    )
    inner = session._inner
    assert isinstance(inner, CreateOSSandboxSession)
    sandbox = inner._sandbox

    await session.start()
    sandbox.next_result = _CommandResult(
        standard_output="hello\n",
        standard_error="warning\n",
        exit_code=3,
    )
    result = await session.exec("example", "argument", shell=False)
    assert result.stdout == b"hello\n"
    assert result.stderr == b"warning\n"
    assert result.exit_code == 3
    request, options = sandbox.requests[-1]
    assert request.arguments[-2:] == ["example", "argument"]
    assert options.environment_variables == {"BASE": "1"}

    await session.write(Path("note.txt"), io.BytesIO(b"contents"))
    assert sandbox.files.data["/workspace/note.txt"] == b"contents"
    restored = await session.read(Path("note.txt"))
    assert restored.read() == b"contents"

    endpoint = await session.resolve_exposed_port(8080)
    assert (endpoint.host, endpoint.port, endpoint.tls) == (
        "sandbox.example.test",
        443,
        True,
    )

    sdk_client = _FakeClient.current
    assert sdk_client is not None
    create_request, _ = sdk_client.created_requests[0]
    assert create_request.shape == "s-4vcpu-4gb"
    assert create_request.rootfs == "devbox:1"
    assert create_request.ingress_enabled is True
    assert [network.id for network in create_request.networks] == ["network-1", "network-2"]
    assert create_request.disk_mib == 8192
    assert create_request.egress_rules == ["tcp:443"]
    assert create_request.ssh_public_keys == ["ssh-ed25519 test"]
    assert create_request.host_id == "host-1"
    assert create_request.node_selector == {"pool": "gpu"}
    assert create_request.region == "us-east"
    assert create_request.auto_pause_after_seconds == 300


async def test_managed_process_pty_exec_and_stdin() -> None:
    client = CreateOSSandboxClient()
    session = await client.create(options=CreateOSSandboxClientOptions("s-4vcpu-4gb", "devbox:1"))
    sandbox = session._inner._sandbox

    assert session.supports_pty() is True
    started = await session.pty_exec_start(
        "echo",
        "ready",
        shell=False,
        tty=True,
        yield_time_s=0.25,
    )
    assert started.process_id is not None
    assert started.output == b"terminal-ready\n"
    assert started.exit_code is None

    finished = await session.pty_write_stdin(
        session_id=started.process_id,
        chars="exit\n",
        yield_time_s=0.25,
    )
    assert finished.process_id is None
    assert finished.output == b"terminal-done\n"
    assert finished.exit_code == 0
    assert sandbox.processes.inputs == [
        ("process-1", "echo ready\n"),
        ("process-1", "exit\n"),
    ]
    assert sandbox.processes.deleted == ["process-1"]


async def test_shutdown_waits_for_inflight_pty_input_before_process_delete() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(pause_on_exit=True), sandbox=sandbox)
    started = await session.pty_exec_start("sh", shell=False, tty=True, yield_time_s=0)
    assert started.process_id is not None

    input_started = threading.Event()
    release_input = threading.Event()
    delete_started = threading.Event()
    original_input = sandbox.processes.input
    original_delete = sandbox.processes.delete

    def blocking_input(process_id: str, data: str) -> int:
        if data == "echo writing\n":
            input_started.set()
            if not release_input.wait(timeout=5):
                raise TimeoutError("test did not release PTY input")
            sandbox.lifecycle_events.append("process_input")
        return original_input(process_id, data)

    def observed_delete(process_id: str) -> _Request:
        delete_started.set()
        return original_delete(process_id)

    sandbox.processes.input = blocking_input  # type: ignore[method-assign]
    sandbox.processes.delete = observed_delete  # type: ignore[method-assign]
    write_task = asyncio.create_task(
        session.pty_write_stdin(
            session_id=started.process_id,
            chars="echo writing\n",
            yield_time_s=0,
        )
    )
    assert await asyncio.to_thread(input_started.wait, 2)
    shutdown_task = asyncio.create_task(session.shutdown())
    try:
        await asyncio.sleep(0)
        assert shutdown_task.done() is False
        assert not await asyncio.to_thread(delete_started.wait, 0.1)
        assert sandbox.processes.deleted == []
    finally:
        release_input.set()
        results = await asyncio.gather(write_task, shutdown_task, return_exceptions=True)

    assert all(not isinstance(result, BaseException) for result in results)
    assert sandbox.lifecycle_events.index("process_input") < sandbox.lifecycle_events.index(
        "process_delete"
    )
    assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
        "pause"
    )


async def test_pty_rejects_user_before_provider_effects() -> None:
    session = CreateOSSandboxSession.from_state(_state(), sandbox=_FakeSandbox())

    with pytest.raises(NotImplementedError, match="does not support `user`"):
        await session.pty_exec_start("id", shell=False, user="sandbox")

    assert session._sandbox.processes.created_requests == []


async def test_pty_support_requires_complete_process_api() -> None:
    sandbox = _FakeSandbox()
    sandbox.processes.connect = None  # type: ignore[method-assign]
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)

    assert session.supports_pty() is False
    with pytest.raises(NotImplementedError, match="not supported"):
        await session.pty_exec_start("echo", "ready", shell=False)
    assert sandbox.processes.created_requests == []


async def test_pty_initial_input_failure_cleans_up_process() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)

    def fail_input(_process_id: str, _data: str) -> int:
        raise RuntimeError("input failed")

    sandbox.processes.input = fail_input  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="input failed"):
        await session.pty_exec_start("echo", "ready", shell=False)

    assert sandbox.processes.deleted == ["process-1"]
    assert session._pty_sessions == {}


async def test_shutdown_waits_for_failed_pty_setup_cleanup_before_pause() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    delete_started = threading.Event()
    release_delete = threading.Event()
    original_delete = sandbox.processes.delete

    def fail_input(_process_id: str, _data: str) -> int:
        raise RuntimeError("input failed")

    def blocking_delete(process_id: str) -> _Request:
        delete_started.set()
        if not release_delete.wait(timeout=5):
            raise TimeoutError("test did not release PTY deletion")
        return original_delete(process_id)

    sandbox.processes.input = fail_input  # type: ignore[method-assign]
    sandbox.processes.delete = blocking_delete  # type: ignore[method-assign]
    pty = asyncio.create_task(session.pty_exec_start("echo", "ready", shell=False))
    assert await asyncio.to_thread(delete_started.wait, 2)
    shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    release_delete.set()

    with pytest.raises(RuntimeError, match="input failed"):
        await pty
    await shutdown
    assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
        "pause"
    )


async def test_failed_pty_setup_delete_is_visible_to_queued_shutdown_retry() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    delete_started = threading.Event()
    release_delete = threading.Event()
    original_delete = sandbox.processes.delete
    delete_calls = 0

    def fail_input(_process_id: str, _data: str) -> int:
        raise RuntimeError("input failed")

    def fail_delete_once(process_id: str) -> _Request:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            delete_started.set()
            if not release_delete.wait(timeout=5):
                raise TimeoutError("test did not release PTY deletion")
            raise RuntimeError("delete failed")
        return original_delete(process_id)

    sandbox.processes.input = fail_input  # type: ignore[method-assign]
    sandbox.processes.delete = fail_delete_once  # type: ignore[method-assign]
    pty = asyncio.create_task(session.pty_exec_start("echo", "ready", shell=False))
    assert await asyncio.to_thread(delete_started.wait, 2)
    shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    release_delete.set()

    with pytest.raises(RuntimeError, match="delete failed"):
        await pty
    await shutdown
    assert delete_calls == 2
    assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
        "pause"
    )


async def test_cancelled_pty_setup_cleanup_blocks_queued_shutdown() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    input_started = threading.Event()
    release_input = threading.Event()
    delete_started = threading.Event()
    release_delete = threading.Event()
    original_input = sandbox.processes.input
    original_delete = sandbox.processes.delete

    def blocking_input(process_id: str, data: str) -> int:
        input_started.set()
        if not release_input.wait(timeout=5):
            raise TimeoutError("test did not release PTY input")
        return original_input(process_id, data)

    def blocking_delete(process_id: str) -> _Request:
        delete_started.set()
        if not release_delete.wait(timeout=5):
            raise TimeoutError("test did not release PTY deletion")
        return original_delete(process_id)

    sandbox.processes.input = blocking_input  # type: ignore[method-assign]
    sandbox.processes.delete = blocking_delete  # type: ignore[method-assign]
    pty = asyncio.create_task(session.pty_exec_start("echo", "ready", shell=False))
    assert await asyncio.to_thread(input_started.wait, 2)
    pty.cancel()
    release_input.set()
    assert await asyncio.to_thread(delete_started.wait, 2)
    shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    release_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await pty
    await shutdown
    assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
        "pause"
    )


async def test_shutdown_waits_for_inflight_pty_creation_before_pause() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    creation_started = threading.Event()
    release_creation = threading.Event()
    original_create = sandbox.processes.create

    def blocking_create(request: object) -> _Request:
        creation_started.set()
        if not release_creation.wait(timeout=5):
            raise TimeoutError("test did not release PTY creation")
        return original_create(request)

    sandbox.processes.create = blocking_create  # type: ignore[method-assign]
    exec_task = asyncio.create_task(
        session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    )
    assert await asyncio.to_thread(creation_started.wait, 2)
    shutdown_task = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    release_creation.set()

    await exec_task
    await shutdown_task

    assert sandbox.processes.deleted == ["process-1"]
    assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
        "pause"
    )


@pytest.mark.parametrize("operation", ["stop", "terminate"])
async def test_pty_cleanup_waits_for_inflight_creation(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    creation_started = threading.Event()
    release_creation = threading.Event()
    original_create = sandbox.processes.create

    def blocking_create(request: object) -> _Request:
        creation_started.set()
        if not release_creation.wait(timeout=5):
            raise TimeoutError("test did not release PTY creation")
        return original_create(request)

    async def record_snapshot() -> None:
        sandbox.lifecycle_events.append("persist_snapshot")

    sandbox.processes.create = blocking_create  # type: ignore[method-assign]
    monkeypatch.setattr(session, "_persist_snapshot", record_snapshot)
    exec_task = asyncio.create_task(
        session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    )
    cleanup_task: asyncio.Task[None] | None = None
    try:
        assert await asyncio.to_thread(creation_started.wait, 2)
        cleanup_task = asyncio.create_task(
            session.stop() if operation == "stop" else session.pty_terminate_all()
        )
        await asyncio.sleep(0)
        assert not cleanup_task.done()
    finally:
        release_creation.set()
        if cleanup_task is not None:
            await asyncio.gather(exec_task, cleanup_task, return_exceptions=True)
        else:
            await asyncio.gather(exec_task, return_exceptions=True)

    exec_task.result()
    assert cleanup_task is not None
    cleanup_task.result()
    assert sandbox.processes.deleted == ["process-1"]
    if operation == "stop":
        assert sandbox.lifecycle_events.index("process_delete") < sandbox.lifecycle_events.index(
            "persist_snapshot"
        )
    else:
        assert "persist_snapshot" not in sandbox.lifecycle_events


async def test_stop_keeps_queued_pty_creation_out_of_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    snapshot_started = asyncio.Event()
    release_snapshot = asyncio.Event()
    admission_captured = asyncio.Event()
    original_create = sandbox.processes.create
    original_capture = session._capture_pty_admission_generation

    def record_create(request: object) -> _Request:
        sandbox.lifecycle_events.append("process_create")
        return original_create(request)

    def record_admission() -> int:
        admission_captured.set()
        return original_capture()

    async def blocking_snapshot() -> None:
        sandbox.lifecycle_events.append("snapshot_start")
        snapshot_started.set()
        await release_snapshot.wait()
        sandbox.lifecycle_events.append("snapshot_end")

    sandbox.processes.create = record_create  # type: ignore[method-assign]
    monkeypatch.setattr(session, "_capture_pty_admission_generation", record_admission)
    monkeypatch.setattr(session, "_persist_snapshot", blocking_snapshot)
    stop_task: asyncio.Task[None] | None = None
    exec_task: asyncio.Task[object] | None = None
    try:
        async with session._lifecycle_lock:
            stop_task = asyncio.create_task(session.stop())
            await asyncio.sleep(0)
            exec_task = asyncio.create_task(
                session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
            )
            await asyncio.wait_for(admission_captured.wait(), timeout=2)

        await asyncio.wait_for(snapshot_started.wait(), timeout=2)
        assert sandbox.processes.created_requests == []
    finally:
        release_snapshot.set()
        if stop_task is not None and exec_task is not None:
            await asyncio.gather(stop_task, exec_task, return_exceptions=True)

    assert stop_task is not None
    assert exec_task is not None
    stop_task.result()
    exec_task.result()
    assert sandbox.lifecycle_events.index("snapshot_end") < sandbox.lifecycle_events.index(
        "process_create"
    )
    await session.pty_terminate_all()


async def test_shutdown_closes_late_pty_stream_and_waits_for_reader() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    connect_started = threading.Event()
    release_connect = threading.Event()
    connected_streams: list[_FakeProcessStream] = []
    original_connect = sandbox.processes.connect

    def blocking_connect(process_id: str, options: object) -> _FakeProcessStream:
        connect_started.set()
        if not release_connect.wait(timeout=5):
            raise TimeoutError("test did not release PTY connection")
        stream = original_connect(process_id, options)
        connected_streams.append(stream)
        return stream

    sandbox.processes.connect = blocking_connect  # type: ignore[method-assign]
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    assert update.process_id is not None
    assert await asyncio.to_thread(connect_started.wait, 2)
    entry = session._pty_sessions[update.process_id]
    shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    release_connect.set()

    await shutdown
    assert len(connected_streams) == 1
    assert connected_streams[0]._closed is True
    assert entry.reader_task is None
    assert session._pty_sessions == {}
    assert sandbox.status == "paused"


async def test_quiet_pty_reconnects_after_idle_timeout_without_losing_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents.extensions.sandbox.createos import sandbox as createos_adapter

    monkeypatch.setattr(createos_adapter, "_PTY_STREAM_READ_TIMEOUT_S", 0.02)
    sandbox = _FakeSandbox()
    state = _state(pause_on_exit=True)
    session = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
    connected_streams: list[_FakeProcessStream] = []
    reconnected = threading.Event()

    class QuietProcessStream(_FakeProcessStream):
        def close(self) -> None:
            self._closed = True

        def __next__(self) -> _ProcessEvent:
            read_timeout = self._response.request.extensions["timeout"]["read"]
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                threading.Event().wait(timeout=read_timeout)
                raise httpx.ReadTimeout("quiet stream") from None
            if event is self._END:
                raise StopIteration
            assert isinstance(event, _ProcessEvent)
            event.sequence = 1
            return event

    def connect(process_id: str, options: object) -> _FakeProcessStream:
        if connected_streams:
            assert cast(_Options, options).after_sequence == 1
            stream = _FakeProcessStream(sandbox.processes._events[process_id])
            reconnected.set()
        else:
            stream = QuietProcessStream(
                sandbox.processes._events[process_id],
                timeout=getattr(options, "timeout", None),
            )
        connected_streams.append(stream)
        return stream

    sandbox.processes.connect = connect  # type: ignore[method-assign]
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.05)

    assert update.process_id is not None
    assert await asyncio.to_thread(reconnected.wait, 2)
    assert len(connected_streams) == 2
    timeout_config = connected_streams[0]._response.request.extensions["timeout"]
    assert timeout_config["connect"] == state.timeouts.fast_op_s
    assert timeout_config["read"] == 0.02
    assert session._pty_sessions[update.process_id].output_closed.is_set() is False
    await session.pty_write_stdin(
        session_id=update.process_id,
        chars="exit\n",
        yield_time_s=0.05,
    )

    await session.shutdown()

    assert connected_streams[0]._closed is True
    assert session._pty_sessions == {}
    assert sandbox.status == "paused"


async def test_pty_replay_gap_error_stops_reconnecting_and_reports_failure() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    cursors: list[int] = []

    class GapStream(_FakeProcessStream):
        def __init__(self, event: _ProcessEvent) -> None:
            super().__init__(queue.Queue())
            self._event: _ProcessEvent | None = event

        def __next__(self) -> _ProcessEvent:
            if self._event is not None:
                event, self._event = self._event, None
                return event
            threading.Event().wait(timeout=0.02)
            raise httpx.ReadTimeout("idle after event")

        def close(self) -> None:
            self._closed = True

    def connect(_process_id: str, options: object) -> GapStream:
        cursors.append(cast(_Options, options).after_sequence)
        event = (
            _ProcessEvent("data", data=b"ready\n", sequence=1)
            if len(cursors) == 1
            else _ProcessEvent("error", error_message="output replay gap", sequence=1)
        )
        return GapStream(event)

    sandbox.processes.connect = connect  # type: ignore[method-assign]
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)

    assert update.process_id is None
    assert update.exit_code == 1
    assert b"ready\n" in update.output
    assert b"output replay gap" in update.output
    assert cursors == [0, 1]
    assert sandbox.processes.deleted == ["process-1"]


async def test_pty_termination_settles_when_closing_stream_does_not_wake_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents.extensions.sandbox.createos import sandbox as createos_adapter

    monkeypatch.setattr(createos_adapter, "_PTY_STREAM_READ_TIMEOUT_S", 0.1)
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    read_blocked = threading.Event()

    class BlockingProcessStream(_FakeProcessStream):
        def __init__(self, events: queue.Queue[object], *, timeout: float | None = None) -> None:
            super().__init__(events, timeout=timeout)
            self._initial_output_read = False

        def __next__(self) -> _ProcessEvent:
            if not self._initial_output_read:
                self._initial_output_read = True
                return super().__next__()
            read_blocked.set()
            read_timeout = self._response.request.extensions["timeout"]["read"]
            threading.Event().wait(timeout=read_timeout)
            raise httpx.ReadTimeout("provider stream stayed idle after process deletion")

        def close(self) -> None:
            self._closed = True

    def connect(process_id: str, options: object) -> BlockingProcessStream:
        return BlockingProcessStream(
            sandbox.processes._events[process_id],
            timeout=getattr(options, "timeout", None),
        )

    sandbox.processes.connect = connect  # type: ignore[method-assign]
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    assert update.process_id is not None
    assert await asyncio.to_thread(read_blocked.wait, 2)

    await asyncio.wait_for(session.pty_terminate_all(), timeout=2)

    assert sandbox.processes.deleted == ["process-1"]
    assert session._pty_sessions == {}


async def test_incompatible_pty_stream_closes_before_iteration_and_cleans_up() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    connected_streams: list[_FakeProcessStream] = []
    iteration_started = False

    class IncompatibleProcessStream(_FakeProcessStream):
        def __init__(self, events: queue.Queue[object]) -> None:
            super().__init__(events)
            del self._response

        def __next__(self) -> _ProcessEvent:
            nonlocal iteration_started
            iteration_started = True
            return super().__next__()

    def connect(process_id: str, _options: object) -> _FakeProcessStream:
        stream = IncompatibleProcessStream(sandbox.processes._events[process_id])
        connected_streams.append(stream)
        return stream

    sandbox.processes.connect = connect  # type: ignore[method-assign]

    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)

    assert update.process_id is None
    assert update.output == b""
    assert len(connected_streams) == 1
    assert connected_streams[0]._closed is True
    assert iteration_started is False
    assert sandbox.processes.deleted == ["process-1"]
    assert session._pty_sessions == {}


async def test_shutdown_rejects_pty_queued_before_lifecycle_admission() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    start_blocked = asyncio.Event()
    release_start = asyncio.Event()
    pty_generation_captured = asyncio.Event()
    original_ensure_started = session._ensure_backend_started
    original_capture_generation = session._capture_pty_admission_generation

    async def blocking_ensure_started() -> None:
        start_blocked.set()
        await release_start.wait()
        await original_ensure_started()

    def capture_generation() -> int:
        generation = original_capture_generation()
        pty_generation_captured.set()
        return generation

    session._ensure_backend_started = blocking_ensure_started  # type: ignore[method-assign]
    session._capture_pty_admission_generation = capture_generation  # type: ignore[method-assign]
    start_task = asyncio.create_task(session.start())
    await start_blocked.wait()
    pty_task = asyncio.create_task(
        session.pty_exec_start("echo", "queued", shell=False, yield_time_s=0.25)
    )
    await pty_generation_captured.wait()
    shutdown_task = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    release_start.set()

    await start_task
    with pytest.raises(SandboxRuntimeError, match="stopping"):
        await pty_task
    await shutdown_task

    assert sandbox.processes.created_requests == []
    assert sandbox.status == "paused"


async def test_concurrent_shutdown_keeps_pty_admission_closed_after_start() -> None:
    sandbox = _FakeSandbox(status="paused")
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    wait_started = threading.Event()
    release_wait = threading.Event()
    original_wait = sandbox.wait_until_running

    def blocking_wait(options: object) -> _FakeSandbox:
        wait_started.set()
        if not release_wait.wait(timeout=5):
            raise TimeoutError("test did not release running wait")
        return original_wait(options)

    sandbox.wait_until_running = blocking_wait  # type: ignore[method-assign]
    start_task = asyncio.create_task(session.start())
    assert await asyncio.to_thread(wait_started.wait, 2)
    shutdown_task = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    release_wait.set()

    await start_task
    await shutdown_task

    with pytest.raises(SandboxRuntimeError, match="stopping"):
        await session.pty_exec_start("echo", "ready", shell=False)
    assert sandbox.processes.created_requests == []
    assert sandbox.status == "paused"


async def test_pty_delete_failure_retains_retry_ownership_before_pause() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    assert update.process_id is not None
    original_delete = sandbox.processes.delete
    delete_calls = 0

    def fail_once(process_id: str) -> _Request:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise RuntimeError("delete failed")
        return original_delete(process_id)

    sandbox.processes.delete = fail_once  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="delete failed"):
        await session.shutdown()
    assert update.process_id in session._pty_sessions
    assert sandbox.pause_calls == 0

    await session.shutdown()
    assert session._pty_sessions == {}
    assert sandbox.processes.deleted == ["process-1"]
    assert sandbox.status == "paused"


async def test_cancelled_pty_delete_records_success_before_shutdown_retry() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    update = await session.pty_exec_start("sh", shell=False, yield_time_s=0.25)
    assert update.process_id is not None
    delete_started = threading.Event()
    release_delete = threading.Event()
    original_delete = sandbox.processes.delete
    delete_calls = 0

    def blocking_delete(process_id: str) -> _Request:
        nonlocal delete_calls
        delete_calls += 1
        delete_started.set()
        if not release_delete.wait(timeout=5):
            raise TimeoutError("test did not release PTY deletion")
        return original_delete(process_id)

    sandbox.processes.delete = blocking_delete  # type: ignore[method-assign]
    write = asyncio.create_task(
        session.pty_write_stdin(
            session_id=update.process_id,
            chars="exit\n",
            yield_time_s=0.25,
        )
    )
    assert await asyncio.to_thread(delete_started.wait, 2)
    write.cancel()
    release_delete.set()

    with pytest.raises(asyncio.CancelledError):
        await write
    assert session._pty_sessions[update.process_id].provider_deleted is True

    await session.shutdown()
    assert delete_calls == 1
    assert session._pty_sessions == {}
    assert sandbox.status == "paused"


async def test_unregistered_pty_delete_failure_retains_retry_ownership() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    original_delete = sandbox.processes.delete

    def fail_input(_process_id: str, _data: str) -> int:
        raise RuntimeError("input failed")

    def fail_delete(_process_id: str) -> _Request:
        raise RuntimeError("delete failed")

    sandbox.processes.input = fail_input  # type: ignore[method-assign]
    sandbox.processes.delete = fail_delete  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="delete failed"):
        await session.pty_exec_start("echo", "ready", shell=False)

    assert len(session._pty_sessions) == 1
    sandbox.processes.delete = original_delete  # type: ignore[method-assign]
    await session.pty_terminate_all()
    assert session._pty_sessions == {}
    assert sandbox.processes.deleted == ["process-1"]


async def test_unregistered_pty_cleanup_records_delete_success_after_timeout() -> None:
    sandbox = _FakeSandbox()
    state = _state(pause_on_exit=True)
    state.timeouts = CreateOSSandboxTimeouts(cleanup_s=1)
    session = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
    original_delete = sandbox.processes.delete
    delete_calls = 0
    delete_started = threading.Event()

    def fail_input(_process_id: str, _data: str) -> int:
        raise RuntimeError("input failed")

    def slow_delete(process_id: str) -> _Request:
        nonlocal delete_calls
        delete_calls += 1
        delete_started.set()
        time.sleep(1.05)
        return original_delete(process_id)

    sandbox.processes.input = fail_input  # type: ignore[method-assign]
    sandbox.processes.delete = slow_delete  # type: ignore[method-assign]
    pty = asyncio.create_task(session.pty_exec_start("echo", "ready", shell=False))
    assert await asyncio.to_thread(delete_started.wait, 2)
    shutdown = asyncio.create_task(session.shutdown())
    await asyncio.sleep(0)
    assert shutdown.done() is False
    with pytest.raises(TimeoutError):
        await pty

    assert len(session._pty_sessions) == 1
    entry = next(iter(session._pty_sessions.values()))
    assert entry.provider_deleted is True

    await shutdown
    assert delete_calls == 1
    assert session._pty_sessions == {}
    assert sandbox.status == "paused"


async def test_terminal_mount_cleanup_rejects_existing_pty_input() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    update = await session.pty_exec_start("echo", "ready", shell=False, yield_time_s=0.25)
    assert update.process_id is not None
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    original_terminate_all = session._pty_terminate_all_locked

    async def blocking_terminate_all() -> None:
        cleanup_started.set()
        await release_cleanup.wait()
        await original_terminate_all()

    session._pty_terminate_all_locked = blocking_terminate_all  # type: ignore[method-assign]
    termination = asyncio.create_task(session._terminate_ambiguous_mount_transition())
    await cleanup_started.wait()

    with pytest.raises(SandboxRuntimeError, match="unavailable"):
        await session.pty_write_stdin(
            session_id=update.process_id,
            chars="blocked",
            yield_time_s=0.25,
        )

    release_cleanup.set()
    await termination
    assert ("process-1", "blocked") not in sandbox.processes.inputs
    assert sandbox.status == "destroyed"


async def test_cancelled_pty_creation_cleans_up_returned_process() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    creation_started = threading.Event()
    release_creation = threading.Event()
    original_create = sandbox.processes.create

    def blocking_create(request: object) -> _Request:
        creation_started.set()
        if not release_creation.wait(timeout=5):
            raise TimeoutError("test did not release PTY creation")
        return original_create(request)

    sandbox.processes.create = blocking_create  # type: ignore[method-assign]
    exec_task = asyncio.create_task(session.pty_exec_start("echo", "ready", shell=False))
    assert await asyncio.to_thread(creation_started.wait, 2)
    exec_task.cancel()
    release_creation.set()

    with pytest.raises(asyncio.CancelledError):
        await exec_task
    assert sandbox.processes.deleted == ["process-1"]
    assert session._pty_sessions == {}


async def test_cancelled_create_destroys_provider_resource() -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    create_started = threading.Event()
    release_create = threading.Event()
    original_create = sdk_client.create_sandbox

    def blocking_create(request: object, options: object) -> _FakeSandbox:
        create_started.set()
        if not release_create.wait(timeout=5):
            raise TimeoutError("test did not release sandbox creation")
        return original_create(request, options)

    sdk_client.create_sandbox = blocking_create  # type: ignore[method-assign]
    create = asyncio.create_task(
        client.create(options=CreateOSSandboxClientOptions("s-4vcpu-4gb", "devbox:1"))
    )
    assert await asyncio.to_thread(create_started.wait, 2)
    create.cancel()
    release_create.set()

    with pytest.raises(asyncio.CancelledError):
        await create
    assert len(sdk_client.created_requests) == 1
    created = next(iter(sdk_client.sandboxes.values()))
    assert created.destroyed is True
    assert created.waited_until_destroyed is True


@pytest.mark.parametrize(
    ("failure_attr", "message"),
    [
        ("destroy_errors", "destroy cleanup failed"),
        ("wait_until_destroyed_errors", "destroy wait cleanup failed"),
    ],
)
async def test_cancelled_create_surfaces_cleanup_failure_with_sandbox_id(
    failure_attr: str,
    message: str,
) -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    sandbox = _FakeSandbox("sb-interrupted")
    getattr(sandbox, failure_attr).append(RuntimeError(message))
    create_started = threading.Event()
    release_create = threading.Event()

    def blocking_create(request: object, options: object) -> _FakeSandbox:
        sdk_client.created_requests.append((request, options))
        create_started.set()
        if not release_create.wait(timeout=5):
            raise TimeoutError("test did not release sandbox creation")
        sdk_client.sandboxes[sandbox.id] = sandbox
        return sandbox

    sdk_client.create_sandbox = blocking_create  # type: ignore[method-assign]
    create = asyncio.create_task(
        client.create(options=CreateOSSandboxClientOptions("s-4vcpu-4gb", "devbox:1"))
    )
    assert await asyncio.to_thread(create_started.wait, 2)
    create.cancel()
    release_create.set()

    with pytest.raises(SandboxRuntimeError, match="sb-interrupted") as exc_info:
        await create
    assert exc_info.value.context["sandbox_id"] == "sb-interrupted"
    assert isinstance(exc_info.value.cause, RuntimeError)


async def test_exec_maps_timeout_and_transport_errors() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    provider_timeout = _OperationTimeout("slow")
    sandbox.next_error = provider_timeout
    with pytest.raises(ExecTimeoutError) as timeout_info:
        await session.exec("slow", shell=False, timeout=1)
    assert timeout_info.value.context["provider_error"] == "slow"
    assert timeout_info.value.cause is provider_timeout

    sandbox.next_error = _APIError(401, "unauthorized")
    with pytest.raises(ExecTransportError) as exc_info:
        await session.exec("private", shell=False)
    assert exc_info.value.retryable is False
    assert exc_info.value.context["http_status"] == 401


async def test_read_maps_provider_not_found() -> None:
    session = CreateOSSandboxSession.from_state(_state(), sandbox=_FakeSandbox())
    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read("missing.txt")


async def test_exposed_port_must_be_configured() -> None:
    session = CreateOSSandboxSession.from_state(_state(), sandbox=_FakeSandbox())
    with pytest.raises(ExposedPortUnavailableError) as exc_info:
        await session.resolve_exposed_port(8080)
    assert exc_info.value.context["reason"] == "not_configured"


async def test_port_provider_failure_is_normalized() -> None:
    state = _state()
    state.exposed_ports = (8080,)
    sandbox = _FakeSandbox()

    def fail(_port: int) -> str:
        raise _APIError(503)

    sandbox.preview_url = fail  # type: ignore[method-assign]
    session = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
    with pytest.raises(ExposedPortUnavailableError) as exc_info:
        await session.resolve_exposed_port(8080)
    assert exc_info.value.retryable is True


async def test_portable_workspace_persist_and_hydrate() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    persisted = await session.persist_workspace()
    with tarfile.open(fileobj=persisted, mode="r:") as archive:
        assert archive.extractfile("state.txt").read() == b"persisted\n"  # type: ignore[union-attr]

    await session.hydrate_workspace(io.BytesIO(_tar_bytes()))
    assert any("createos-hydrate-" in path for path, _, _ in sandbox.files.uploads)
    assert any(
        "createos-hydrate-" in " ".join(request.arguments) for request, _ in sandbox.requests
    )


async def test_hydrate_detaches_and_restores_ephemeral_mount() -> None:
    session = CreateOSSandboxSession.from_state(_state(), sandbox=_FakeSandbox())
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock()
    strategy.restore_after_snapshot = AsyncMock()
    mount = MagicMock(mount_strategy=strategy)
    mount_path = Path("/workspace/mounted")
    manifest = MagicMock(wraps=session.state.manifest)
    manifest.root = session.state.manifest.root
    manifest.environment = session.state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, mount_path)]
    session.state.manifest = manifest

    await session.hydrate_workspace(io.BytesIO(_tar_bytes()))

    strategy.teardown_for_snapshot.assert_awaited_once_with(mount, session, mount_path)
    strategy.restore_after_snapshot.assert_awaited_once_with(mount, session, mount_path)


async def test_hydrate_restores_mount_after_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = CreateOSSandboxSession.from_state(_state(), sandbox=_FakeSandbox())
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock()
    strategy.restore_after_snapshot = AsyncMock()
    mount = MagicMock(mount_strategy=strategy)
    mount_path = Path("/workspace/mounted")
    manifest = MagicMock(wraps=session.state.manifest)
    manifest.root = session.state.manifest.root
    manifest.environment = session.state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, mount_path)]
    session.state.manifest = manifest

    async def fail_extract(*command: str | Path, timeout: float | None = None) -> ExecResult:
        _ = timeout
        if command[:2] == ("sh", "-c") and "tar -C" in str(command[2]):
            return ExecResult(stdout=b"", stderr=b"extract failed", exit_code=1)
        return ExecResult(stdout=b"", stderr=b"", exit_code=0)

    monkeypatch.setattr(session, "_exec_internal", fail_extract)
    with pytest.raises(WorkspaceArchiveWriteError):
        await session.hydrate_workspace(io.BytesIO(_tar_bytes()))

    strategy.restore_after_snapshot.assert_awaited_once_with(mount, session, mount_path)


async def test_hydrate_restores_prior_mount_after_partial_teardown_failure() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    first_strategy = MagicMock()
    first_strategy.teardown_for_snapshot = AsyncMock()
    first_strategy.restore_after_snapshot = AsyncMock()
    second_strategy = MagicMock()
    second_strategy.teardown_for_snapshot = AsyncMock(side_effect=RuntimeError("detach failed"))
    second_strategy.restore_after_snapshot = AsyncMock()
    first_mount = MagicMock(mount_strategy=first_strategy)
    second_mount = MagicMock(mount_strategy=second_strategy)
    first_path = Path("/workspace/first")
    second_path = Path("/workspace/second")
    manifest = MagicMock(wraps=session.state.manifest)
    manifest.root = session.state.manifest.root
    manifest.environment = session.state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [
        (first_mount, first_path),
        (second_mount, second_path),
    ]
    session.state.manifest = manifest

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.hydrate_workspace(io.BytesIO(_tar_bytes()))

    first_strategy.restore_after_snapshot.assert_awaited_once_with(first_mount, session, first_path)
    second_strategy.restore_after_snapshot.assert_not_awaited()
    assert sandbox.destroyed is True
    assert sandbox.paused is False


async def test_start_snapshot_mount_failure_destroys_without_waiting_on_own_lock(
    tmp_path: Path,
) -> None:
    sandbox = _FakeSandbox()
    state = _state(pause_on_exit=True)
    snapshot = LocalSnapshot(id="restorable-workspace", base_path=tmp_path)
    await snapshot.persist(io.BytesIO(_tar_bytes()))
    state.snapshot = snapshot
    session = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock(side_effect=RuntimeError("detach failed"))
    mount = MagicMock(mount_strategy=strategy)
    manifest = MagicMock(wraps=state.manifest)
    manifest.root = state.manifest.root
    manifest.environment = state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, Path("/workspace/mounted"))]
    state.manifest = manifest

    with pytest.raises(WorkspaceArchiveWriteError):
        await asyncio.wait_for(session.start(), timeout=2)

    assert sandbox.destroyed is True
    assert sandbox.waited_until_destroyed is True
    assert sandbox.paused is False
    assert session._active_lifecycle_owner is None


async def test_stop_snapshot_mount_failure_destroys_without_waiting_on_own_lock(
    tmp_path: Path,
) -> None:
    sandbox = _FakeSandbox()
    state = _state(pause_on_exit=True)
    state.snapshot = LocalSnapshot(id="persisted-workspace", base_path=tmp_path)
    session = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock(side_effect=RuntimeError("detach failed"))
    mount = MagicMock(mount_strategy=strategy)
    manifest = MagicMock(wraps=state.manifest)
    manifest.root = state.manifest.root
    manifest.environment = state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, Path("/workspace/mounted"))]
    state.manifest = manifest

    with pytest.raises(WorkspaceArchiveReadError):
        await asyncio.wait_for(session.stop(), timeout=2)

    assert sandbox.destroyed is True
    assert sandbox.waited_until_destroyed is True
    assert sandbox.paused is False
    assert session._active_lifecycle_owner is None


async def test_persist_restore_ambiguity_force_destroys_paused_session() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock()
    strategy.restore_after_snapshot = AsyncMock(side_effect=RuntimeError("restore failed"))
    mount = MagicMock(mount_strategy=strategy)
    mount_path = Path("/workspace/mounted")
    manifest = MagicMock(wraps=session.state.manifest)
    manifest.root = session.state.manifest.root
    manifest.environment = session.state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, mount_path)]
    session.state.manifest = manifest

    with pytest.raises(WorkspaceArchiveReadError):
        await session.persist_workspace()

    assert sandbox.destroyed is True
    assert sandbox.waited_until_destroyed is True
    assert sandbox.paused is False
    with pytest.raises(SandboxRuntimeError) as exc_info:
        await session.exec("true", shell=False)
    assert exc_info.value.error_code == ErrorCode.MOUNT_FAILED


async def test_hydrate_waits_for_cancelled_provider_call_before_restoring_mount() -> None:
    sandbox = _FakeSandbox()
    original_run_command = sandbox.run_command
    extract_started = threading.Event()
    release_extract = threading.Event()

    def blocking_run_command(request: object, options: object) -> _CommandResponse:
        command_text = " ".join(request.arguments[4:])
        if "createos-hydrate-" in command_text and " -xf " in command_text:
            extract_started.set()
            if not release_extract.wait(timeout=5):
                raise TimeoutError("test did not release archive extraction")
        return original_run_command(request, options)

    sandbox.run_command = blocking_run_command  # type: ignore[method-assign]
    session = CreateOSSandboxSession.from_state(_state(), sandbox=sandbox)
    strategy = MagicMock()
    strategy.teardown_for_snapshot = AsyncMock()
    strategy.restore_after_snapshot = AsyncMock()
    mount = MagicMock(mount_strategy=strategy)
    mount_path = Path("/workspace/mounted")
    manifest = MagicMock(wraps=session.state.manifest)
    manifest.root = session.state.manifest.root
    manifest.environment = session.state.manifest.environment
    manifest.ephemeral_mount_targets.return_value = [(mount, mount_path)]
    session.state.manifest = manifest

    hydrate = asyncio.create_task(session.hydrate_workspace(io.BytesIO(_tar_bytes())))
    assert await asyncio.to_thread(extract_started.wait, 2)
    hydrate.cancel()
    await asyncio.sleep(0)
    strategy.restore_after_snapshot.assert_not_awaited()

    release_extract.set()
    with pytest.raises(asyncio.CancelledError):
        await hydrate
    strategy.restore_after_snapshot.assert_awaited_once_with(mount, session, mount_path)


@pytest.mark.parametrize("pause_on_exit", [False, True])
async def test_shutdown_obeys_pause_on_exit(pause_on_exit: bool) -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=pause_on_exit),
        sandbox=sandbox,
    )
    await session.shutdown()
    assert sandbox.paused is pause_on_exit
    assert sandbox.destroyed is not pause_on_exit
    assert sandbox.wait_until_paused_calls == int(pause_on_exit)
    assert sandbox.wait_until_destroyed_calls == int(not pause_on_exit)


@pytest.mark.parametrize("shutdown_first", [False, True])
async def test_explicit_delete_destroys_pause_on_exit_session(shutdown_first: bool) -> None:
    client = CreateOSSandboxClient()
    session = await client.create(
        options=CreateOSSandboxClientOptions("s-1vcpu-2gb", "devbox:1", pause_on_exit=True)
    )
    sandbox = session._inner._sandbox
    if shutdown_first:
        await session.shutdown()
        assert sandbox.status == "paused"

    await client.delete(session)

    assert sandbox.status == "destroyed"
    assert sandbox.destroy_calls == 1
    assert sandbox.wait_until_destroyed_calls == 1


async def test_restarted_paused_session_pauses_again_on_next_shutdown() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )

    await session.shutdown()
    await session.start()
    await session.shutdown()

    assert sandbox.status == "paused"
    assert sandbox.pause_calls == 2
    assert sandbox.wait_until_paused_calls == 2


async def test_cancelled_restart_finishes_activation_before_next_pause() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    await session.shutdown()
    wait_started = threading.Event()
    release_wait = threading.Event()

    def blocking_wait(_options: object) -> _FakeSandbox:
        wait_started.set()
        if not release_wait.wait(timeout=5):
            raise TimeoutError("test did not release running wait")
        sandbox.status = "running"
        return sandbox

    sandbox.wait_until_running = blocking_wait  # type: ignore[method-assign]
    start = asyncio.create_task(session.start())
    assert await asyncio.to_thread(wait_started.wait, 2)
    start.cancel()
    release_wait.set()

    with pytest.raises(asyncio.CancelledError):
        await start
    await session.shutdown()

    assert sandbox.status == "paused"
    assert sandbox.pause_calls == 2
    assert sandbox.wait_until_paused_calls == 2


@pytest.mark.parametrize(
    ("pause_on_exit", "failure_point", "error_attr", "message"),
    [
        (True, "pause", "pause_errors", "pause failed"),
        (True, "pause_wait", "wait_until_paused_errors", "pause wait failed"),
        (False, "destroy", "destroy_errors", "destroy failed"),
        (False, "destroy_wait", "wait_until_destroyed_errors", "destroy wait failed"),
    ],
)
async def test_shutdown_propagates_cleanup_failures_and_retries(
    pause_on_exit: bool,
    failure_point: str,
    error_attr: str,
    message: str,
) -> None:
    sandbox = _FakeSandbox()
    getattr(sandbox, error_attr).append(RuntimeError(message))
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=pause_on_exit),
        sandbox=sandbox,
    )

    with pytest.raises(RuntimeError, match=message):
        await session.shutdown()
    await session.shutdown()
    await session.shutdown()

    if pause_on_exit:
        assert sandbox.pause_calls == (2 if failure_point == "pause" else 1)
        assert sandbox.wait_until_paused_calls == (2 if failure_point == "pause_wait" else 1)
        assert sandbox.destroy_calls == 0
    else:
        assert sandbox.destroy_calls == (2 if failure_point == "destroy" else 1)
        assert sandbox.wait_until_destroyed_calls == (2 if failure_point == "destroy_wait" else 1)
        assert sandbox.pause_calls == 0


@pytest.mark.parametrize(
    ("pause_on_exit", "mutation_name"),
    [(True, "pause"), (False, "destroy")],
)
async def test_cancelled_cleanup_records_success_before_retry(
    pause_on_exit: bool,
    mutation_name: str,
) -> None:
    client = CreateOSSandboxClient()
    session = await client.create(
        options=CreateOSSandboxClientOptions(
            "s-4vcpu-4gb",
            "devbox:1",
            pause_on_exit=pause_on_exit,
        )
    )
    sandbox = session._inner._sandbox
    mutation_started = threading.Event()
    release_mutation = threading.Event()
    original_mutation = getattr(sandbox, mutation_name)

    def blocking_mutation() -> object:
        mutation_started.set()
        if not release_mutation.wait(timeout=5):
            raise TimeoutError("test did not release lifecycle mutation")
        return original_mutation()

    setattr(sandbox, mutation_name, blocking_mutation)
    shutdown = asyncio.create_task(session.shutdown())
    assert await asyncio.to_thread(mutation_started.wait, 2)
    shutdown.cancel()
    release_mutation.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown
    await session.shutdown()

    assert getattr(sandbox, f"{mutation_name}_calls") == 1
    if pause_on_exit:
        assert sandbox.wait_until_paused_calls == 1
    else:
        assert sandbox.wait_until_destroyed_calls == 1


async def test_failed_running_wait_still_starts_a_new_pause_cycle() -> None:
    sandbox = _FakeSandbox()
    session = CreateOSSandboxSession.from_state(
        _state(pause_on_exit=True),
        sandbox=sandbox,
    )
    await session.shutdown()

    def fail_running_wait(_options: object) -> _FakeSandbox:
        raise RuntimeError("activation wait failed")

    sandbox.wait_until_running = fail_running_wait  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="activation wait failed"):
        await session.start()

    await session.shutdown()
    assert sandbox.status == "paused"
    assert sandbox.pause_calls == 2
    assert sandbox.wait_until_paused_calls == 2


async def test_resume_reconnects_paused_sandbox() -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    sandbox = _FakeSandbox(status="paused")
    sdk_client.sandboxes[sandbox.id] = sandbox

    resumed = await client.resume(_state(pause_on_exit=True))
    assert resumed._inner._sandbox is sandbox
    assert sandbox.resumed is False
    assert resumed._inner._workspace_state_preserved_on_start() is True

    await resumed.start()
    assert sandbox.resumed is True


async def test_paused_resume_is_owned_before_cancellable_activation() -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    sandbox = _FakeSandbox(status="paused")
    wait_started = threading.Event()
    release_wait = threading.Event()

    def blocking_wait(_options: object) -> _FakeSandbox:
        wait_started.set()
        if not release_wait.wait(timeout=5):
            raise TimeoutError("test did not release running wait")
        return sandbox

    sandbox.wait_until_running = blocking_wait  # type: ignore[method-assign]
    sdk_client.sandboxes[sandbox.id] = sandbox
    resumed = await client.resume(_state(pause_on_exit=True))
    assert sandbox.resumed is False

    start = asyncio.create_task(resumed.start())
    assert await asyncio.to_thread(wait_started.wait, 2)
    start.cancel()
    release_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert sandbox.resumed is True
    await resumed.shutdown()
    assert sandbox.paused is True


async def test_pausing_resume_settles_pause_before_owned_activation() -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    sandbox = _FakeSandbox(status="pausing")
    sdk_client.sandboxes[sandbox.id] = sandbox

    resumed = await client.resume(_state(pause_on_exit=True))
    assert sandbox.resumed is False

    await resumed.start()

    assert sandbox.resumed is True
    assert sandbox.status == "running"


@pytest.mark.parametrize("status", ["destroying", "destroyed", "failed", "error"])
async def test_resume_recreates_terminal_sandbox(status: str) -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    terminal = _FakeSandbox(status=status)
    sdk_client.sandboxes[terminal.id] = terminal

    resumed = await client.resume(_state())

    assert resumed._inner._sandbox is not terminal
    assert resumed._inner._workspace_state_preserved_on_start() is False
    assert len(sdk_client.created_requests) == 1
    assert terminal.waited_until_destroyed is (status == "destroying")


@pytest.mark.parametrize("settled_status", ["failed", "error"])
async def test_resume_recreates_destroying_sandbox_that_settles_terminally(
    settled_status: str,
) -> None:
    client = CreateOSSandboxClient()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    terminal = _FakeSandbox(status="destroying")
    terminal.destroy_wait_error_status = settled_status
    sdk_client.sandboxes[terminal.id] = terminal

    resumed = await client.resume(_state())

    assert resumed._inner._sandbox is not terminal
    assert resumed._inner._workspace_state_preserved_on_start() is False
    assert len(sdk_client.created_requests) == 1
    assert terminal.waited_until_destroyed is True


async def test_resume_recreates_only_missing_sandbox() -> None:
    client = CreateOSSandboxClient()
    state = _state()
    state.network_ids = ("network-1",)
    state.disk_mib = 8192
    state.egress_rules = ("tcp:443",)
    state.ssh_public_keys = ("ssh-ed25519 test",)
    state.host_id = "host-1"
    state.node_selector = {"pool": "gpu"}
    state.region = "us-east"
    state.auto_pause_after_seconds = 300
    resumed = await client.resume(state)
    assert resumed.state.sandbox_id == "sb-1"
    assert resumed._inner._workspace_state_preserved_on_start() is False

    sdk_client = _FakeClient.current
    assert sdk_client is not None
    create_request, _ = sdk_client.created_requests[0]
    assert [network.id for network in create_request.networks] == ["network-1"]
    assert create_request.disk_mib == 8192
    assert create_request.egress_rules == ["tcp:443"]
    assert create_request.ssh_public_keys == ["ssh-ed25519 test"]
    assert create_request.host_id == "host-1"
    assert create_request.node_selector == {"pool": "gpu"}
    assert create_request.region == "us-east"
    assert create_request.auto_pause_after_seconds == 300
    sdk_client.get_sandbox = lambda _sandbox_id: (_ for _ in ()).throw(_APIError(503))  # type: ignore[method-assign]
    with pytest.raises(_APIError, match="provider error"):
        await client.resume(_state())


async def test_client_state_serialization_and_close() -> None:
    client = CreateOSSandboxClient()
    payload = client.serialize_session_state(_state())
    restored = client.deserialize_session_state(payload)
    assert isinstance(restored, CreateOSSandboxSessionState)
    assert restored.shape == "s-4vcpu-4gb"

    await client.close()
    sdk_client = _FakeClient.current
    assert sdk_client is not None
    assert sdk_client.closed is True
