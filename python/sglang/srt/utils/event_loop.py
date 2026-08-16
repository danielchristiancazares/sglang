"""Event-loop helpers with an asyncio fallback for native Windows."""

from __future__ import annotations

import asyncio
import sys
from typing import Awaitable, TypeVar

try:
    import uvloop
except ImportError:  # uvloop does not provide Windows wheels.
    uvloop = None


T = TypeVar("T")


def install_event_loop_policy() -> None:
    """Use uvloop where available and the platform asyncio loop elsewhere."""
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    elif sys.platform == "win32":
        # pyzmq requires add_reader/add_writer, which the default Windows
        # ProactorEventLoop does not implement.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_event_loop(awaitable: Awaitable[T]) -> T:
    """Run an awaitable on the fastest event loop available on this platform."""
    if uvloop is not None:
        return uvloop.run(awaitable)
    return asyncio.run(awaitable)


def granian_loop_name() -> str:
    """Return the Granian loop selector matching the installed implementation."""
    return "uvloop" if uvloop is not None else "asyncio"


def windows_selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Build the selector loop required by pyzmq on native Windows."""
    return asyncio.SelectorEventLoop()


def uvicorn_loop_name() -> str:
    """Return a Uvicorn loop factory that preserves pyzmq compatibility."""
    if uvloop is not None:
        return "uvloop"
    if sys.platform == "win32":
        # Uvicorn 0.36+ selects ProactorEventLoop for its built-in ``asyncio``
        # factory, independently of the process-wide event-loop policy.
        return "sglang.srt.utils.event_loop:windows_selector_loop_factory"
    return "asyncio"
