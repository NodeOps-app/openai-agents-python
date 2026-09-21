"""CreateOS sandbox implementation.

This module adapts the synchronous ``createos`` Python SDK to the asynchronous
Agents SDK sandbox interfaces. The dependency is optional and imported lazily.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from ....logger import log_tool_action_debug
from ....sandbox._mount_security import redact_mount_error_data
from ....sandbox.errors import (
    ErrorCode,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    SandboxRuntimeError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceWriteTypeError,
)
from ....sandbox.manifest import Manifest
from ....sandbox.session import SandboxSession, SandboxSessionState
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.session.dependencies import Dependencies
from ....sandbox.session.manager import Instrumentation
from ....sandbox.session.mount_lifecycle import with_ephemeral_mounts_removed
from ....sandbox.session.pty_output import collect_pty_output
from ....sandbox.session.pty_types import (
    PTY_PROCESSES_MAX,
    PTY_PROCESSES_WARNING,
    PtyExecUpdate,
    _settle_pty_cleanup,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
    process_id_to_prune_from_meta,
    resolve_pty_write_yield_time_ms,
)
from ....sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript
from ....sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from ....sandbox.session.tar_workspace import shell_tar_exclude_args
from ....sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from ....sandbox.types import ExecResult, ExposedPortEndpoint, User
from ....sandbox.util.retry import TRANSIENT_HTTP_STATUS_CODES, iter_exception_chain, retry_async
from ....sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from ....sandbox.workspace_paths import coerce_posix_path, posix_path_as_path, sandbox_path_str

DEFAULT_CREATEOS_WORKSPACE_ROOT = "/workspace"
logger = logging.getLogger(__name__)
_TERMINAL_CREATEOS_STATUSES = frozenset({"destroying", "destroyed", "failed", "error"})
_PTY_STREAM_READ_TIMEOUT_S = 5.0
_lifecycle_owner: ContextVar[asyncio.Task[None] | None] = ContextVar(
    "createos_lifecycle_owner", default=None
)


def _import_createos_sdk() -> Any:
    try:
        import createos

        return createos
    except ImportError as e:
        raise ImportError(
            "CreateOSSandboxClient requires the optional `createos-sandbox` dependency.\n"
            "Install the CreateOS extra before using this sandbox backend."
        ) from e


def _provider_status_code(error: BaseException) -> int | None:
    for candidate in iter_exception_chain(error):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _provider_retryability(error: BaseException) -> bool | None:
    status = _provider_status_code(error)
    if status is None:
        return True if isinstance(error, ConnectionError | TimeoutError) else None
    if status in TRANSIENT_HTTP_STATUS_CODES:
        return True
    if 400 <= status < 500:
        return False
    return None


def _require_rootfs(rootfs: str | None) -> str:
    if rootfs is None or not rootfs.strip():
        raise ValueError("CreateOS sandbox creation requires a non-empty rootfs")
    return rootfs


def _set_pty_stream_read_timeout(stream: Any, timeout: float) -> None:
    """Bound an idle stream read so process cleanup can settle its reader."""
    response = getattr(stream, "_response", None)
    request = getattr(response, "request", None)
    extensions = getattr(request, "extensions", None)
    timeout_config = extensions.get("timeout") if isinstance(extensions, MutableMapping) else None
    if not isinstance(timeout_config, MutableMapping):
        raise RuntimeError(
            "CreateOS managed process stream does not expose configurable HTTP timeouts"
        )
    timeout_config["read"] = timeout


async def _to_thread_settled(
    function: Any,
    /,
    *args: Any,
    timeout: float | None = None,
    interrupted_result_cleanup: Callable[[Any], Awaitable[None]] | None = None,
    on_success: Callable[[Any], None] | None = None,
) -> Any:
    """Keep a synchronous provider call owned until its worker thread exits."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    caller_cancelled = False
    timed_out = False
    try:
        if timeout is None:
            await asyncio.shield(task)
        else:
            done, _ = await asyncio.wait({task}, timeout=timeout)
            timed_out = not done
    except asyncio.CancelledError:
        caller_cancelled = True

    if caller_cancelled or timed_out:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                caller_cancelled = True
        try:
            result = task.result()
        except BaseException:
            if timed_out:
                raise TimeoutError from None
            if caller_cancelled:
                raise asyncio.CancelledError from None
            raise
        if on_success is not None:
            on_success(result)
        if interrupted_result_cleanup is not None:
            cleanup_task: asyncio.Future[None] = asyncio.ensure_future(
                interrupted_result_cleanup(result)
            )
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    caller_cancelled = True
            cleanup_task.result()
        if timed_out:
            raise TimeoutError from None
        raise asyncio.CancelledError from None

    result = task.result()
    if on_success is not None:
        on_success(result)
    return result


def _raise_exec_error(
    error: BaseException,
    *,
    command: tuple[str | Path, ...],
    timeout: float,
) -> NoReturn:
    sdk = _import_createos_sdk()
    context: dict[str, object] = {"backend": "createos"}
    status = _provider_status_code(error)
    if status is not None:
        context["http_status"] = status
    detail = str(error).strip()
    if detail:
        context["provider_error"] = detail

    if isinstance(error, sdk.OperationTimeout | TimeoutError) or status in {
        408,
        504,
    }:
        raise ExecTimeoutError(
            command=command,
            timeout_s=timeout,
            context=context,
            cause=error,
        ) from error
    raise ExecTransportError(
        command=command,
        context=context,
        cause=error,
        retryable=_provider_retryability(error),
    ) from error


class CreateOSSandboxTimeouts(BaseModel):
    """Timeout configuration for CreateOS sandbox operations."""

    model_config = {"frozen": True}

    exec_timeout_unbounded_s: float = Field(default=24 * 60 * 60, ge=1)
    create_s: float = Field(default=120, ge=1)
    lifecycle_s: float = Field(default=120, ge=1)
    fast_op_s: float = Field(default=30, ge=1)
    file_upload_s: float = Field(default=1800, ge=1)
    file_download_s: float = Field(default=1800, ge=1)
    workspace_tar_s: float = Field(default=300, ge=1)
    cleanup_s: float = Field(default=30, ge=1)


class CreateOSSandboxClientOptions(BaseSandboxClientOptions):
    """Creation and lifecycle settings for a CreateOS sandbox."""

    type: Literal["createos"] = "createos"
    shape: str = Field(min_length=1)
    rootfs: str | None = None
    env_vars: dict[str, str] | None = None
    pause_on_exit: bool = False
    name: str | None = None
    exposed_ports: tuple[int, ...] = ()
    timeouts: CreateOSSandboxTimeouts | dict[str, object] | None = None
    network_ids: tuple[str, ...] = ()
    disk_mib: int | None = Field(default=None, gt=0)
    egress_rules: tuple[str, ...] = ()
    ssh_public_keys: tuple[str, ...] = ()
    host_id: str | None = None
    node_selector: dict[str, str] | None = None
    region: str | None = None
    auto_pause_after_seconds: int | None = Field(default=None, gt=0)

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("shape must not be blank")
        return value

    @field_validator("exposed_ports")
    @classmethod
    def _validate_exposed_ports(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(port < 1 or port > 65_535 for port in value):
            raise ValueError("exposed_ports must contain valid TCP ports")
        return value

    def __init__(
        self,
        shape: str,
        rootfs: str | None = None,
        env_vars: dict[str, str] | None = None,
        pause_on_exit: bool = False,
        name: str | None = None,
        exposed_ports: tuple[int, ...] = (),
        timeouts: CreateOSSandboxTimeouts | dict[str, object] | None = None,
        network_ids: tuple[str, ...] = (),
        disk_mib: int | None = None,
        egress_rules: tuple[str, ...] = (),
        ssh_public_keys: tuple[str, ...] = (),
        host_id: str | None = None,
        node_selector: dict[str, str] | None = None,
        region: str | None = None,
        auto_pause_after_seconds: int | None = None,
        *,
        type: Literal["createos"] = "createos",
    ) -> None:
        super().__init__(
            type=type,
            shape=shape,
            rootfs=rootfs,
            env_vars=env_vars,
            pause_on_exit=pause_on_exit,
            name=name,
            exposed_ports=exposed_ports,
            timeouts=timeouts,
            network_ids=network_ids,
            disk_mib=disk_mib,
            egress_rules=egress_rules,
            ssh_public_keys=ssh_public_keys,
            host_id=host_id,
            node_selector=node_selector,
            region=region,
            auto_pause_after_seconds=auto_pause_after_seconds,
        )


class CreateOSSandboxSessionState(SandboxSessionState):
    """Serializable state for a CreateOS-backed sandbox session."""

    type: Literal["createos"] = "createos"
    sandbox_id: str
    shape: str = Field(min_length=1)
    rootfs: str | None = None
    _base_env_vars: dict[str, str] = PrivateAttr(default_factory=dict)
    pause_on_exit: bool = False
    name: str | None = None
    timeouts: CreateOSSandboxTimeouts = Field(default_factory=CreateOSSandboxTimeouts)
    network_ids: tuple[str, ...] = ()
    disk_mib: int | None = Field(default=None, gt=0)
    egress_rules: tuple[str, ...] = ()
    ssh_public_keys: tuple[str, ...] = ()
    host_id: str | None = None
    node_selector: dict[str, str] | None = None
    region: str | None = None
    auto_pause_after_seconds: int | None = Field(default=None, gt=0)

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("shape must not be blank")
        return value

    @field_validator("exposed_ports")
    @classmethod
    def _validate_exposed_ports(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(port < 1 or port > 65_535 for port in value):
            raise ValueError("exposed_ports must contain valid TCP ports")
        return value


@dataclass
class _CreateOSPtySessionEntry:
    provider_process_id: str
    termination_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_chunks: deque[bytes] = field(default_factory=deque)
    output_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_notify: asyncio.Event = field(default_factory=asyncio.Event)
    output_closed: asyncio.Event = field(default_factory=asyncio.Event)
    last_used: float = field(default_factory=time.monotonic)
    exit_code: int | None = None
    reader_task: asyncio.Task[None] | None = None
    stream: Any | None = None
    termination_started: bool = False
    provider_deleted: bool = False

    def mark_provider_deleted(self, _result: Any) -> None:
        self.provider_deleted = True


class CreateOSSandboxSession(BaseSandboxSession):
    """CreateOS-backed implementation of the provider-neutral sandbox session."""

    state: CreateOSSandboxSessionState

    def __init__(self, *, state: CreateOSSandboxSessionState, sandbox: Any) -> None:
        self.state = state
        self._sandbox = sandbox
        self._mount_transition_terminal = False
        self._pause_issued = False
        self._pause_completed = False
        self._destroy_issued = False
        self._destroy_completed = False
        self._lifecycle_lock = asyncio.Lock()
        self._active_lifecycle_owner: asyncio.Task[None] | None = None
        self._pty_lock = asyncio.Lock()
        self._pty_sessions: dict[int, _CreateOSPtySessionEntry] = {}
        self._reserved_pty_process_ids: set[int] = set()
        self._pty_admission_generation = 0
        self._pty_admission_open = True

    @classmethod
    def from_state(
        cls,
        state: CreateOSSandboxSessionState,
        *,
        sandbox: Any,
    ) -> CreateOSSandboxSession:
        return cls(state=state, sandbox=sandbox)

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    def _assert_session_usable(self) -> None:
        if self._mount_transition_terminal:
            raise SandboxRuntimeError(
                message="sandbox session is unavailable after an ambiguous mount transition",
                error_code=ErrorCode.MOUNT_FAILED,
                op="shutdown",
                context={"backend": "createos"},
                retryable=False,
            )

    def _mark_pause_issued(self, _result: Any) -> None:
        self._pause_issued = True

    def _mark_destroy_issued(self, _result: Any) -> None:
        self._destroy_issued = True

    def _mark_resume_completed(self, _result: Any) -> None:
        # A successful resume starts a new pause lifecycle even if waiting for
        # the running state subsequently fails.
        self._pause_issued = False
        self._pause_completed = False

    async def start(self) -> None:
        admission_generation = self._pty_admission_generation
        async with self._lifecycle_lock:
            owner = asyncio.current_task()
            assert owner is not None
            self._active_lifecycle_owner = owner
            token = _lifecycle_owner.set(owner)
            try:
                await _settle_pty_cleanup(super().start())
                if admission_generation == self._pty_admission_generation:
                    self._pty_admission_open = True
            finally:
                self._active_lifecycle_owner = None
                _lifecycle_owner.reset(token)

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            owner = asyncio.current_task()
            assert owner is not None
            self._active_lifecycle_owner = owner
            token = _lifecycle_owner.set(owner)
            try:
                await _settle_pty_cleanup(super().stop())
            finally:
                self._active_lifecycle_owner = None
                _lifecycle_owner.reset(token)

    async def _before_stop(self) -> None:
        # stop() already holds the lifecycle lock through snapshot persistence.
        await self._pty_terminate_all_locked()

    async def _ensure_backend_started(self) -> None:
        await self._ensure_backend_started_locked()

    async def _ensure_backend_started_locked(self) -> None:
        self._assert_session_usable()
        if self._destroy_issued or self._destroy_completed:
            raise SandboxRuntimeError(
                message="CreateOS sandbox session cannot restart after destruction",
                error_code=ErrorCode.WORKSPACE_START_ERROR,
                op="start",
                context={"backend": "createos", "sandbox_id": self.state.sandbox_id},
                retryable=False,
            )
        sdk = _import_createos_sdk()
        status = str(self._sandbox.status)
        resumed = False
        if status == "pausing":
            await _to_thread_settled(
                self._sandbox.wait_until_paused,
                sdk.WaitOptions(timeout=self.state.timeouts.lifecycle_s),
                timeout=self.state.timeouts.lifecycle_s + 1,
            )
            status = str(self._sandbox.status)
        if status == "paused":
            await _to_thread_settled(
                self._sandbox.resume,
                timeout=self.state.timeouts.lifecycle_s,
                on_success=self._mark_resume_completed,
            )
            resumed = True
            status = str(self._sandbox.status)
        if resumed or status != "running":
            await _to_thread_settled(
                self._sandbox.wait_until_running,
                sdk.WaitOptions(timeout=self.state.timeouts.lifecycle_s),
                timeout=self.state.timeouts.lifecycle_s + 1,
            )

    def _close_pty_admission(self) -> None:
        self._pty_admission_open = False
        self._pty_admission_generation += 1

    def _capture_pty_admission_generation(self) -> int:
        if not self._pty_admission_open:
            raise SandboxRuntimeError(
                message="CreateOS sandbox session is stopping",
                error_code=ErrorCode.EXEC_TRANSPORT_ERROR,
                op="exec",
                context={"backend": "createos", "sandbox_id": self.state.sandbox_id},
                retryable=False,
            )
        return self._pty_admission_generation

    def _assert_pty_admission_open(self, generation: int) -> None:
        if not self._pty_admission_open or generation != self._pty_admission_generation:
            raise SandboxRuntimeError(
                message="CreateOS sandbox session is stopping",
                error_code=ErrorCode.EXEC_TRANSPORT_ERROR,
                op="exec",
                context={"backend": "createos", "sandbox_id": self.state.sandbox_id},
                retryable=False,
            )

    async def _before_shutdown(self) -> None:
        # shutdown() already holds the lifecycle lock.
        await self._pty_terminate_all_locked()

    async def shutdown(self) -> None:
        self._close_pty_admission()
        async with self._lifecycle_lock:
            await _settle_pty_cleanup(super().shutdown())

    async def _cleanup_provider_process(self, process: Any) -> None:
        process_id = str(getattr(process, "process_id", ""))
        if not process_id:
            return
        entry = _CreateOSPtySessionEntry(provider_process_id=process_id)
        entry.output_closed.set()
        try:
            await _to_thread_settled(
                self._sandbox.processes.delete,
                process_id,
                timeout=self.state.timeouts.cleanup_s,
                on_success=entry.mark_provider_deleted,
            )
        except BaseException:
            async with self._pty_lock:
                local_process_id = allocate_pty_process_id(self._reserved_pty_process_ids)
                self._reserved_pty_process_ids.add(local_process_id)
                self._pty_sessions[local_process_id] = entry
            raise

    async def _prepare_backend_workspace(self) -> None:
        root = sandbox_path_str(self.state.manifest.root)
        result = await self._exec_at_cwd(
            ("mkdir", "-p", "--", root),
            cwd="/",
            timeout=self.state.timeouts.fast_op_s,
        )
        if not result.ok():
            raise WorkspaceArchiveWriteError(
                path=self._workspace_root_path(),
                context={
                    "reason": "workspace_root_create_failed",
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
            )

    async def _validate_path_access(
        self,
        path: Path | str,
        *,
        for_write: bool = False,
    ) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    async def _resolved_envs(self) -> dict[str, str]:
        manifest_envs = await self.state.manifest.environment.resolve()
        return {**self.state._base_env_vars, **manifest_envs}

    def _coerce_exec_timeout(self, timeout: float | None) -> float:
        if timeout is None:
            return self.state.timeouts.exec_timeout_unbounded_s
        return max(float(timeout), 0.001)

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        return await self._exec_at_cwd(
            tuple(str(part) for part in command),
            cwd=sandbox_path_str(self.state.manifest.root),
            timeout=timeout,
        )

    async def _exec_at_cwd(
        self,
        command: tuple[str, ...],
        *,
        cwd: str,
        timeout: float | None,
    ) -> ExecResult:
        self._assert_session_usable()
        sdk = _import_createos_sdk()
        effective_timeout = self._coerce_exec_timeout(timeout)
        request = sdk.RunCommandRequest(
            command="sh",
            arguments=[
                "-lc",
                'cd "$1" && shift && exec "$@"',
                "sh",
                cwd,
                *command,
            ],
        )
        options = sdk.ExecOptions(
            timeout=effective_timeout,
            environment_variables=await self._resolved_envs() or None,
        )
        try:
            response = await _to_thread_settled(
                self._sandbox.run_command,
                request,
                options,
                timeout=effective_timeout + 1,
            )
        except Exception as e:
            _raise_exec_error(
                e,
                command=command,
                timeout=effective_timeout,
            )

        result = response.result
        return ExecResult(
            stdout=str(result.standard_output or "").encode("utf-8", errors="replace"),
            stderr=str(result.standard_error or result.error_message or "").encode(
                "utf-8", errors="replace"
            ),
            exit_code=int(result.exit_code or 0),
        )

    def supports_pty(self) -> bool:
        processes = getattr(self._sandbox, "processes", None)
        return all(
            callable(getattr(processes, name, None))
            for name in ("create", "connect", "input", "delete")
        )

    async def pty_exec_start(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
        tty: bool = False,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        if user is not None:
            raise NotImplementedError("CreateOS PTY execution does not support `user`")
        if not self.supports_pty():
            raise NotImplementedError("PTY execution is not supported by this CreateOS SDK")

        generation = self._capture_pty_admission_generation()
        sanitized = self._prepare_exec_command(*command, shell=shell, user=None)
        command_text = shlex.join(str(part) for part in sanitized)
        effective_timeout = self._coerce_exec_timeout(timeout)
        sdk = _import_createos_sdk()
        process: Any | None = None
        entry: _CreateOSPtySessionEntry | None = None
        registered = False
        pruned: tuple[int, _CreateOSPtySessionEntry] | None = None
        process_id = 0
        process_count = 0

        async with self._lifecycle_lock:
            try:
                self._assert_pty_admission_open(generation)
                request = sdk.ManagedProcessCreateRequest(
                    command="/bin/sh",
                    working_directory=sandbox_path_str(self.state.manifest.root),
                    environment_variables=await self._resolved_envs(),
                    pty=sdk.PTYSize(rows=24, cols=80) if tty else None,
                )
                process = await _to_thread_settled(
                    self._sandbox.processes.create,
                    request,
                    timeout=effective_timeout,
                    interrupted_result_cleanup=self._cleanup_provider_process,
                )
                provider_process_id = str(getattr(process, "process_id", ""))
                if not provider_process_id:
                    raise RuntimeError("CreateOS managed process creation returned no process id")
                entry = _CreateOSPtySessionEntry(provider_process_id=provider_process_id)
                await _to_thread_settled(
                    self._sandbox.processes.input,
                    provider_process_id,
                    f"{command_text}\n",
                    timeout=self.state.timeouts.fast_op_s,
                )

                async with self._pty_lock:
                    process_id = allocate_pty_process_id(self._reserved_pty_process_ids)
                    self._reserved_pty_process_ids.add(process_id)
                    pruned = self._prune_pty_sessions_if_needed()
                    self._pty_sessions[process_id] = entry
                    process_count = len(self._pty_sessions)
                    registered = True
                entry.reader_task = asyncio.create_task(self._run_pty_reader(entry))
            except BaseException:
                if process is not None and not registered:
                    await _settle_pty_cleanup(self._cleanup_provider_process(process))
                raise

        assert entry is not None
        if pruned is not None:
            pruned_process_id, pruned_entry = pruned
            await self._terminate_pty_entry(pruned_entry)
            await self._remove_pty_entry(pruned_process_id, pruned_entry)
        if process_count >= PTY_PROCESSES_WARNING:
            logger.warning(
                "PTY process count reached warning threshold: %s active sessions",
                process_count,
            )

        yield_time_ms = 10_000 if yield_time_s is None else int(yield_time_s * 1000)
        output, original_token_count, output_closed = await self._collect_pty_output(
            entry=entry,
            yield_time_ms=clamp_pty_yield_time_ms(yield_time_ms),
            max_output_tokens=max_output_tokens,
        )
        return await self._finalize_pty_update(
            process_id=process_id,
            entry=entry,
            output=output,
            original_token_count=original_token_count,
            output_closed=output_closed,
        )

    async def _run_pty_reader(self, entry: _CreateOSPtySessionEntry) -> None:
        sdk = _import_createos_sdk()
        loop = asyncio.get_running_loop()

        def connect(after_sequence: int) -> Any:
            stream = self._sandbox.processes.connect(
                entry.provider_process_id,
                sdk.ManagedProcessConnectOptions(
                    timeout=self.state.timeouts.fast_op_s,
                    after_sequence=after_sequence,
                ),
            )
            try:
                _set_pty_stream_read_timeout(stream, _PTY_STREAM_READ_TIMEOUT_S)
            except BaseException:
                try:
                    stream.close()
                except Exception:
                    pass
                raise
            return stream

        def append_output(data: bytes) -> None:
            entry.output_chunks.append(data)
            entry.output_notify.set()

        def consume(stream: Any, after_sequence: int) -> tuple[int, bool]:
            sequence = after_sequence
            try:
                with stream:
                    for event in stream:
                        event_type = str(getattr(event, "type", ""))
                        if event_type == "error":
                            message = str(getattr(event, "error_message", "") or "")
                            loop.call_soon_threadsafe(
                                append_output,
                                (message or "CreateOS PTY output stream failed").encode(
                                    "utf-8", errors="replace"
                                ),
                            )
                            entry.exit_code = 1
                            return sequence, False
                        event_sequence = int(getattr(event, "sequence", 0) or 0)
                        if event_sequence > 0:
                            if event_sequence <= sequence:
                                continue
                            sequence = event_sequence
                        if event_type == "data":
                            data = bytes(getattr(event, "data", b"") or b"")
                            if data:
                                loop.call_soon_threadsafe(append_output, data)
                        elif event_type == "exit":
                            exit_code = getattr(event, "exit_code", None)
                            if exit_code is not None:
                                entry.exit_code = int(exit_code)
                            return sequence, False
            except (httpx.ReadTimeout, TimeoutError):
                return sequence, True
            return sequence, False

        try:
            sequence = 0
            while not entry.termination_started:
                stream = await _to_thread_settled(
                    connect,
                    sequence,
                    timeout=self.state.timeouts.fast_op_s + 1,
                    interrupted_result_cleanup=self._close_pty_stream,
                )
                if entry.termination_started:
                    await self._close_pty_stream(stream)
                    break
                entry.stream = stream
                consume_task = asyncio.create_task(asyncio.to_thread(consume, stream, sequence))
                try:
                    sequence, idle_timeout = await asyncio.shield(consume_task)
                except asyncio.CancelledError:
                    await _settle_pty_cleanup(self._close_pty_stream(stream))
                    await _settle_pty_cleanup(cast(Awaitable[None], consume_task))
                    raise
                finally:
                    if entry.stream is stream:
                        entry.stream = None
                if not idle_timeout:
                    break
        except Exception as e:
            log_tool_action_debug(logger, "CreateOS PTY output stream failed", e)
        finally:
            entry.output_closed.set()
            entry.output_notify.set()

    async def _close_pty_stream(self, stream: Any) -> None:
        await _to_thread_settled(
            stream.close,
            timeout=self.state.timeouts.cleanup_s,
        )

    async def pty_write_stdin(
        self,
        *,
        session_id: int,
        chars: str,
        yield_time_s: float | None = None,
        max_output_tokens: int | None = None,
    ) -> PtyExecUpdate:
        self._assert_session_usable()
        generation = self._capture_pty_admission_generation()
        async with self._pty_lock:
            entry = self._resolve_pty_session_entry(
                pty_processes=self._pty_sessions,
                session_id=session_id,
            )

        if chars:
            async with entry.termination_lock:
                self._assert_session_usable()
                self._assert_pty_admission_open(generation)
                if entry.termination_started:
                    raise SandboxRuntimeError(
                        message="CreateOS PTY process is stopping",
                        error_code=ErrorCode.EXEC_TRANSPORT_ERROR,
                        op="exec",
                        context={"backend": "createos", "sandbox_id": self.state.sandbox_id},
                        retryable=False,
                    )
                await _to_thread_settled(
                    self._sandbox.processes.input,
                    entry.provider_process_id,
                    chars,
                    timeout=self.state.timeouts.fast_op_s,
                )

        yield_time_ms = 250 if yield_time_s is None else int(yield_time_s * 1000)
        output, original_token_count, output_closed = await self._collect_pty_output(
            entry=entry,
            yield_time_ms=resolve_pty_write_yield_time_ms(
                yield_time_ms=yield_time_ms,
                input_empty=chars == "",
            ),
            max_output_tokens=max_output_tokens,
        )
        entry.last_used = time.monotonic()
        return await self._finalize_pty_update(
            process_id=session_id,
            entry=entry,
            output=output,
            original_token_count=original_token_count,
            output_closed=output_closed,
        )

    async def _collect_pty_output(
        self,
        *,
        entry: _CreateOSPtySessionEntry,
        yield_time_ms: int,
        max_output_tokens: int | None,
    ) -> tuple[bytes, int | None, bool]:
        return await collect_pty_output(
            output_chunks=entry.output_chunks,
            output_lock=entry.output_lock,
            output_notify=entry.output_notify,
            is_done=entry.output_closed.is_set,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )

    async def _finalize_pty_update(
        self,
        *,
        process_id: int,
        entry: _CreateOSPtySessionEntry,
        output: bytes,
        original_token_count: int | None,
        output_closed: bool,
    ) -> PtyExecUpdate:
        exit_code = entry.exit_code if output_closed else None
        live_process_id: int | None = process_id
        if output_closed:
            await self._terminate_pty_entry(entry)
            await self._remove_pty_entry(process_id, entry)
            live_process_id = None
        return PtyExecUpdate(
            process_id=live_process_id,
            output=output,
            exit_code=exit_code,
            original_token_count=original_token_count,
        )

    async def pty_terminate_all(self) -> None:
        async with self._lifecycle_lock:
            await self._pty_terminate_all_locked()

    async def _pty_terminate_all_locked(self) -> None:
        async with self._pty_lock:
            entries = list(self._pty_sessions.items())
        cleanup_error: BaseException | None = None
        for process_id, entry in entries:
            try:
                await self._terminate_pty_entry(entry)
            except BaseException as e:
                if cleanup_error is None:
                    cleanup_error = e
            else:
                await self._remove_pty_entry(process_id, entry)
        if cleanup_error is not None:
            raise cleanup_error

    async def _remove_pty_entry(
        self,
        process_id: int,
        entry: _CreateOSPtySessionEntry,
    ) -> None:
        async with self._pty_lock:
            if self._pty_sessions.get(process_id) is entry:
                self._pty_sessions.pop(process_id)
                self._reserved_pty_process_ids.discard(process_id)

    def _prune_pty_sessions_if_needed(
        self,
    ) -> tuple[int, _CreateOSPtySessionEntry] | None:
        if len(self._pty_sessions) < PTY_PROCESSES_MAX:
            return None
        meta = [
            (process_id, entry.last_used, entry.output_closed.is_set())
            for process_id, entry in self._pty_sessions.items()
        ]
        process_id = process_id_to_prune_from_meta(meta)
        if process_id is None:
            return None
        entry = self._pty_sessions.get(process_id)
        if entry is None:
            return None
        return process_id, entry

    async def _terminate_pty_entry(self, entry: _CreateOSPtySessionEntry) -> None:
        async with entry.termination_lock:
            entry.termination_started = True
            deletion_error: BaseException | None = None
            if not entry.provider_deleted:
                try:
                    await _to_thread_settled(
                        self._sandbox.processes.delete,
                        entry.provider_process_id,
                        timeout=self.state.timeouts.cleanup_s,
                        on_success=entry.mark_provider_deleted,
                    )
                except BaseException as e:
                    deletion_error = e
            stream = entry.stream
            entry.stream = None
            if stream is not None:
                try:
                    await _to_thread_settled(
                        stream.close,
                        timeout=self.state.timeouts.cleanup_s,
                    )
                except Exception as e:
                    log_tool_action_debug(logger, "CreateOS PTY stream close failed", e)
            reader_task = entry.reader_task
            entry.reader_task = None
            if reader_task is not None and reader_task is not asyncio.current_task():
                await _settle_pty_cleanup(reader_task)
            if deletion_error is not None:
                raise deletion_error

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)

        try:
            return io.BytesIO(
                await self._download_file(
                    sandbox_path_str(workspace_path),
                    timeout=self.state.timeouts.file_download_s,
                )
            )
        except Exception as e:
            if _provider_status_code(e) == 404:
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            raise WorkspaceArchiveReadError(path=error_path, cause=e) from e

    async def write(
        self,
        path: Path | str,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)
        workspace_path = await self._validate_path_access(path, for_write=True)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)

        try:
            await self._upload_file(
                sandbox_path_str(workspace_path),
                bytes(payload),
                timeout=self.state.timeouts.file_upload_s,
            )
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=workspace_path, cause=e) from e

    async def _download_file(self, path: str, *, timeout: float) -> bytes:
        self._assert_session_usable()
        sdk = _import_createos_sdk()

        def download() -> bytes:
            with self._sandbox.files.download(
                path,
                sdk.RequestOptions(timeout=timeout),
            ) as stream:
                return bytes(stream.read())

        return bytes(await _to_thread_settled(download, timeout=timeout + 1))

    async def _upload_file(self, path: str, data: bytes, *, timeout: float) -> None:
        self._assert_session_usable()
        sdk = _import_createos_sdk()
        await _to_thread_settled(
            self._sandbox.files.upload,
            path,
            data,
            sdk.RequestOptions(timeout=timeout),
            timeout=timeout + 1,
        )

    async def running(self) -> bool:
        self._assert_session_usable()
        try:
            await _to_thread_settled(
                self._sandbox.refresh,
                timeout=self.state.timeouts.fast_op_s,
            )
        except Exception as e:
            log_tool_action_debug(logger, "CreateOS sandbox health check failed", e)
            return False
        return str(self._sandbox.status) == "running"

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        self._assert_session_usable()
        try:
            url = await _to_thread_settled(
                self._sandbox.preview_url,
                port,
                timeout=self.state.timeouts.fast_op_s,
            )
            split = urlsplit(url)
            if split.hostname is None or split.scheme not in {"http", "https"}:
                raise ValueError("CreateOS returned an invalid preview URL")
            return ExposedPortEndpoint(
                host=split.hostname,
                port=split.port or (443 if split.scheme == "https" else 80),
                tls=split.scheme == "https",
            )
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "createos"},
                cause=e,
                retryable=_provider_retryability(e),
            ) from e

    def _tar_exclude_args(self) -> list[str]:
        return shell_tar_exclude_args(self._persist_workspace_skip_relpaths())

    @retry_async(
        retry_if=lambda exc, self: (
            isinstance(exc, TimeoutError)
            or _provider_status_code(exc) in TRANSIENT_HTTP_STATUS_CODES
        )
    )
    async def persist_workspace(self) -> io.IOBase:
        root = self._workspace_root_path()
        tar_path = f"/tmp/createos-persist-{self.state.session_id.hex}.tar"
        excludes = " ".join(self._tar_exclude_args())
        tar_cmd = (
            f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        ).strip()

        async def create_archive() -> bytes:
            try:
                result = await self._exec_internal(
                    "sh", "-c", tar_cmd, timeout=self.state.timeouts.workspace_tar_s
                )
                if not result.ok():
                    raise WorkspaceArchiveReadError(
                        path=root,
                        context={
                            "reason": "tar_failed",
                            "stderr": result.stderr.decode("utf-8", errors="replace"),
                        },
                        retryable=False,
                    )
                return await self._download_file(
                    tar_path,
                    timeout=self.state.timeouts.file_download_s,
                )
            except WorkspaceArchiveReadError:
                raise
            except Exception as e:
                raise WorkspaceArchiveReadError(path=root, cause=e) from e
            finally:
                try:
                    await self._exec_internal(
                        "rm", "-f", "--", tar_path, timeout=self.state.timeouts.cleanup_s
                    )
                except Exception as e:
                    log_tool_action_debug(logger, "CreateOS persist cleanup failed", e)

        raw = await with_ephemeral_mounts_removed(
            self,
            create_archive,
            error_path=root,
            error_cls=WorkspaceArchiveReadError,
            operation_error_context_key="snapshot_error_before_remount_corruption",
        )
        return io.BytesIO(raw)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/createos-hydrate-{self.state.session_id.hex}.tar"
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)
        raw = bytes(payload)
        try:
            validate_tar_bytes(raw, allow_external_symlink_targets=False)
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={
                    "reason": "unsafe_or_invalid_tar",
                    "member": e.member,
                    "detail": str(e),
                },
                cause=e,
            ) from e

        async def extract_archive() -> None:
            try:
                await self.mkdir(root, parents=True)
                await self._upload_file(
                    tar_path,
                    raw,
                    timeout=self.state.timeouts.file_upload_s,
                )
                result = await self._exec_internal(
                    "sh",
                    "-c",
                    f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                    timeout=self.state.timeouts.workspace_tar_s,
                )
                if not result.ok():
                    raise WorkspaceArchiveWriteError(
                        path=root,
                        context={
                            "reason": "tar_extract_failed",
                            "stderr": result.stderr.decode("utf-8", errors="replace"),
                        },
                    )
            except WorkspaceArchiveWriteError:
                raise
            except Exception as e:
                raise WorkspaceArchiveWriteError(path=root, cause=e) from e
            finally:
                try:
                    await self._exec_internal(
                        "rm", "-f", "--", tar_path, timeout=self.state.timeouts.cleanup_s
                    )
                except Exception as e:
                    log_tool_action_debug(logger, "CreateOS hydrate cleanup failed", e)

        await with_ephemeral_mounts_removed(
            self,
            extract_archive,
            error_path=root,
            error_cls=WorkspaceArchiveWriteError,
            operation_error_context_key="hydrate_error_before_remount_corruption",
        )

    async def _shutdown_backend(self) -> None:
        sdk = _import_createos_sdk()
        if self._mount_transition_terminal or not self.state.pause_on_exit:
            if self._destroy_completed:
                return
            if not self._destroy_issued:
                await _to_thread_settled(
                    self._sandbox.destroy,
                    timeout=self.state.timeouts.lifecycle_s,
                    on_success=self._mark_destroy_issued,
                )
            await _to_thread_settled(
                self._sandbox.wait_until_destroyed,
                sdk.WaitOptions(timeout=self.state.timeouts.lifecycle_s),
                timeout=self.state.timeouts.lifecycle_s + 1,
            )
            self._destroy_completed = True
        else:
            if self._pause_completed:
                return
            if not self._pause_issued:
                await _to_thread_settled(
                    self._sandbox.pause,
                    timeout=self.state.timeouts.lifecycle_s,
                    on_success=self._mark_pause_issued,
                )
            await _to_thread_settled(
                self._sandbox.wait_until_paused,
                sdk.WaitOptions(timeout=self.state.timeouts.lifecycle_s),
                timeout=self.state.timeouts.lifecycle_s + 1,
            )
            self._pause_completed = True

    async def _terminate_ambiguous_mount_transition(self) -> None:
        self._mount_transition_terminal = True
        self._close_pty_admission()
        # Mount cleanup runs in an owned child task. During snapshot hydration,
        # start or stop already holds the lifecycle lock while awaiting that child.
        # Inherited ownership lets the child terminate without waiting on its
        # parent; stale children cannot bypass a later owner's lock.
        owner = _lifecycle_owner.get()
        if (
            owner is not None
            and owner is self._active_lifecycle_owner
            and self._lifecycle_lock.locked()
        ):
            await _settle_pty_cleanup(self._terminate_ambiguous_mount_transition_locked())
            return
        async with self._lifecycle_lock:
            await _settle_pty_cleanup(self._terminate_ambiguous_mount_transition_locked())

    async def _terminate_ambiguous_mount_transition_locked(self) -> None:
        cleanup_error: BaseException | None = None
        try:
            await self._before_shutdown()
        except BaseException as e:
            cleanup_error = e
        try:
            await self._shutdown_backend()
        except BaseException as e:
            if cleanup_error is None:
                cleanup_error = e
        try:
            await self._after_shutdown()
        except BaseException as e:
            if cleanup_error is None:
                cleanup_error = e
        if cleanup_error is not None:
            raise cleanup_error


class CreateOSSandboxClient(BaseSandboxClient[CreateOSSandboxClientOptions]):
    """Client that manages Agents SDK sessions through the CreateOS SDK."""

    backend_id = "createos"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        sdk = _import_createos_sdk()
        self._client = sdk.Client(api_key=api_key, base_url=base_url, timeout=timeout)
        self._instrumentation = instrumentation or Instrumentation()
        self._dependencies = dependencies
        self._env_vars = dict(env_vars or {})

    @staticmethod
    def _normalize_timeouts(
        value: CreateOSSandboxTimeouts | dict[str, object] | None,
    ) -> CreateOSSandboxTimeouts:
        if isinstance(value, CreateOSSandboxTimeouts):
            return value
        if value is None:
            return CreateOSSandboxTimeouts()
        return CreateOSSandboxTimeouts.model_validate(value)

    def _create_request(
        self,
        *,
        shape: str,
        rootfs: str | None,
        name: str | None,
        env_vars: dict[str, str],
        exposed_ports: tuple[int, ...],
        network_ids: tuple[str, ...],
        disk_mib: int | None,
        egress_rules: tuple[str, ...],
        ssh_public_keys: tuple[str, ...],
        host_id: str | None,
        node_selector: dict[str, str] | None,
        region: str | None,
        auto_pause_after_seconds: int | None,
    ) -> Any:
        sdk = _import_createos_sdk()
        return sdk.CreateSandboxRequest(
            shape=shape,
            rootfs=rootfs or "",
            name=name or "",
            networks=[sdk.NetworkEntry(id=network_id) for network_id in network_ids],
            disk_mib=disk_mib or 0,
            egress_rules=list(egress_rules),
            environment_variables=env_vars,
            ssh_public_keys=list(ssh_public_keys),
            host_id=host_id or "",
            node_selector=dict(node_selector or {}),
            ingress_enabled=bool(exposed_ports),
            region=region or "",
            auto_pause_after_seconds=auto_pause_after_seconds or 0,
        )

    async def _create_sandbox(
        self,
        *,
        shape: str,
        rootfs: str | None,
        name: str | None,
        env_vars: dict[str, str],
        exposed_ports: tuple[int, ...],
        network_ids: tuple[str, ...],
        disk_mib: int | None,
        egress_rules: tuple[str, ...],
        ssh_public_keys: tuple[str, ...],
        host_id: str | None,
        node_selector: dict[str, str] | None,
        region: str | None,
        auto_pause_after_seconds: int | None,
        timeout: float,
        cleanup_timeout: float,
    ) -> Any:
        sdk = _import_createos_sdk()
        request = self._create_request(
            shape=shape,
            rootfs=rootfs,
            name=name,
            env_vars=env_vars,
            exposed_ports=exposed_ports,
            network_ids=network_ids,
            disk_mib=disk_mib,
            egress_rules=egress_rules,
            ssh_public_keys=ssh_public_keys,
            host_id=host_id,
            node_selector=node_selector,
            region=region,
            auto_pause_after_seconds=auto_pause_after_seconds,
        )

        async def destroy_interrupted_creation(sandbox: Any) -> None:
            try:
                await _to_thread_settled(
                    sandbox.destroy,
                    timeout=cleanup_timeout,
                )
                await _to_thread_settled(
                    sandbox.wait_until_destroyed,
                    sdk.WaitOptions(timeout=cleanup_timeout),
                    timeout=cleanup_timeout + 1,
                )
            except BaseException as e:
                sandbox_id = str(getattr(sandbox, "id", "unknown"))
                raise SandboxRuntimeError(
                    message=(
                        "CreateOS interrupted sandbox creation cleanup failed for "
                        f"sandbox {sandbox_id}"
                    ),
                    error_code=ErrorCode.WORKSPACE_START_ERROR,
                    op="start",
                    context={"backend": "createos", "sandbox_id": sandbox_id},
                    cause=e,
                    retryable=_provider_retryability(e),
                ) from e

        return await _to_thread_settled(
            self._client.create_sandbox,
            request,
            sdk.RequestOptions(timeout=timeout),
            timeout=timeout + 1,
            interrupted_result_cleanup=destroy_interrupted_creation,
        )

    @redact_mount_error_data
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: CreateOSSandboxClientOptions,
    ) -> SandboxSession:
        manifest = manifest or Manifest(root=DEFAULT_CREATEOS_WORKSPACE_ROOT)
        self._validate_manifest_for_create(manifest)
        rootfs = _require_rootfs(options.rootfs)
        timeouts = self._normalize_timeouts(options.timeouts)
        session_id = uuid.uuid4()
        name = session_id.hex[:22] if options.name is None else options.name
        env_vars = dict(self._env_vars if options.env_vars is None else options.env_vars)
        sandbox = await self._create_sandbox(
            shape=options.shape,
            rootfs=rootfs,
            name=name,
            env_vars=env_vars,
            exposed_ports=options.exposed_ports,
            network_ids=options.network_ids,
            disk_mib=options.disk_mib,
            egress_rules=options.egress_rules,
            ssh_public_keys=options.ssh_public_keys,
            host_id=options.host_id,
            node_selector=options.node_selector,
            region=options.region,
            auto_pause_after_seconds=options.auto_pause_after_seconds,
            timeout=timeouts.create_s,
            cleanup_timeout=timeouts.lifecycle_s,
        )
        state = CreateOSSandboxSessionState(
            session_id=session_id,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
            manifest=manifest,
            exposed_ports=options.exposed_ports,
            sandbox_id=sandbox.id,
            shape=options.shape,
            rootfs=rootfs,
            pause_on_exit=options.pause_on_exit,
            name=name,
            timeouts=timeouts,
            network_ids=options.network_ids,
            disk_mib=options.disk_mib,
            egress_rules=options.egress_rules,
            ssh_public_keys=options.ssh_public_keys,
            host_id=options.host_id,
            node_selector=(
                dict(options.node_selector) if options.node_selector is not None else None
            ),
            region=options.region,
            auto_pause_after_seconds=options.auto_pause_after_seconds,
        )
        state._base_env_vars = env_vars
        return self._wrap_session(
            CreateOSSandboxSession.from_state(state, sandbox=sandbox),
            instrumentation=self._instrumentation,
        )

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, CreateOSSandboxSession):
            raise TypeError("CreateOSSandboxClient.delete expects a CreateOSSandboxSession")
        inner.state.pause_on_exit = False
        await inner.shutdown()
        return session

    @redact_mount_error_data
    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, CreateOSSandboxSessionState):
            raise TypeError("CreateOSSandboxClient.resume expects a CreateOSSandboxSessionState")
        state.assert_path_grants_rebound()
        state = state.model_copy()
        state._base_env_vars = dict(self._env_vars)
        sdk = _import_createos_sdk()
        sandbox: Any | None = None
        try:
            sandbox = await _to_thread_settled(
                self._client.get_sandbox,
                state.sandbox_id,
                timeout=state.timeouts.fast_op_s,
            )
        except Exception as e:
            if _provider_status_code(e) != 404:
                raise
            log_tool_action_debug(logger, "CreateOS sandbox no longer exists; recreating", e)

        reconnected = False
        if sandbox is not None:
            status = str(sandbox.status)
            if status == "destroying":
                try:
                    await _to_thread_settled(
                        sandbox.wait_until_destroyed,
                        sdk.WaitOptions(timeout=state.timeouts.lifecycle_s),
                        timeout=state.timeouts.lifecycle_s + 1,
                    )
                except Exception:
                    if str(sandbox.status) not in _TERMINAL_CREATEOS_STATUSES - {"destroying"}:
                        raise
                sandbox = None
            elif status in _TERMINAL_CREATEOS_STATUSES:
                sandbox = None
            else:
                reconnected = True

        if sandbox is None:
            rootfs = _require_rootfs(state.rootfs)
            sandbox = await self._create_sandbox(
                shape=state.shape,
                rootfs=rootfs,
                name=state.name,
                env_vars=state._base_env_vars,
                exposed_ports=state.exposed_ports,
                network_ids=state.network_ids,
                disk_mib=state.disk_mib,
                egress_rules=state.egress_rules,
                ssh_public_keys=state.ssh_public_keys,
                host_id=state.host_id,
                node_selector=state.node_selector,
                region=state.region,
                auto_pause_after_seconds=state.auto_pause_after_seconds,
                timeout=state.timeouts.create_s,
                cleanup_timeout=state.timeouts.lifecycle_s,
            )
            state.sandbox_id = cast(Any, sandbox).id
            state.workspace_root_ready = False

        inner = CreateOSSandboxSession.from_state(state, sandbox=sandbox)
        inner._set_start_state_preserved(reconnected, system=reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def close(self) -> None:
        await _to_thread_settled(self._client.close)

    async def __aenter__(self) -> CreateOSSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, CreateOSSandboxSessionState)


__all__ = [
    "DEFAULT_CREATEOS_WORKSPACE_ROOT",
    "CreateOSSandboxClient",
    "CreateOSSandboxClientOptions",
    "CreateOSSandboxSession",
    "CreateOSSandboxSessionState",
    "CreateOSSandboxTimeouts",
]
