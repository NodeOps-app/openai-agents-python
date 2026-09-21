"""Minimal CreateOS-backed sandbox example for manual validation."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from openai.types.responses import ResponseTextDeltaEvent

from agents import ModelSettings, Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.sandbox.misc.example_support import text_manifest
from examples.sandbox.misc.workspace_shell import WorkspaceShellCapability

try:
    from agents.extensions.sandbox import (
        DEFAULT_CREATEOS_WORKSPACE_ROOT,
        CreateOSSandboxClient,
        CreateOSSandboxClientOptions,
    )
except Exception as exc:  # pragma: no cover - import path depends on optional extras
    raise SystemExit(
        "CreateOS sandbox examples require the optional repo extra.\n"
        "Install it with: uv sync --extra createos"
    ) from exc


DEFAULT_QUESTION = "Summarize this cloud sandbox workspace in 2 sentences."


def _manifest() -> Manifest:
    manifest = text_manifest(
        {
            "README.md": (
                "# CreateOS Demo Workspace\n\n"
                "This workspace validates the CreateOS sandbox backend for the Agents SDK.\n"
            ),
            "status.md": (
                "# Status\n\n"
                "- Sandbox creation is configured.\n"
                "- Command execution and file transfer are ready for validation.\n"
            ),
        }
    )
    return manifest.model_copy(update={"root": DEFAULT_CREATEOS_WORKSPACE_ROOT})


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set before running this example.")
    return value


async def main(
    *,
    model: str,
    question: str,
    shape: str,
    rootfs: str | None,
    pause_on_exit: bool,
    stream: bool,
) -> None:
    _require_env("OPENAI_API_KEY")
    api_key = _require_env("CREATEOS_API_KEY")

    agent = SandboxAgent(
        name="CreateOS Sandbox Assistant",
        model=model,
        instructions=(
            "Inspect the sandbox workspace before answering. Keep the answer concise and cite "
            "the file names you inspected."
        ),
        default_manifest=_manifest(),
        capabilities=[WorkspaceShellCapability()],
        model_settings=ModelSettings(tool_choice="required"),
    )
    client = CreateOSSandboxClient(api_key=api_key)
    run_config = RunConfig(
        sandbox=SandboxRunConfig(
            client=client,
            options=CreateOSSandboxClientOptions(
                shape=shape,
                rootfs=rootfs,
                pause_on_exit=pause_on_exit,
            ),
        ),
        workflow_name="CreateOS sandbox example",
    )

    try:
        if not stream:
            result = await Runner.run(agent, question, run_config=run_config)
            print(result.final_output)
            return

        stream_result = Runner.run_streamed(agent, question, run_config=run_config)
        saw_text_delta = False
        async for event in stream_result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                if not saw_text_delta:
                    print("assistant> ", end="", flush=True)
                    saw_text_delta = True
                print(event.data.delta, end="", flush=True)
        if saw_text_delta:
            print()
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-sol", help="Model name to use.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Prompt to send.")
    parser.add_argument("--shape", default="s-4vcpu-4gb", help="CreateOS sandbox shape.")
    parser.add_argument("--rootfs", default="devbox:1", help="CreateOS root filesystem.")
    parser.add_argument("--pause-on-exit", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
