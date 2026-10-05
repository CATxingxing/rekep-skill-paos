from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from dora import Node
from forge_msgs import ToolMessage
from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL, ToolAccepted, ToolControlResponse, ToolEndpointDescriptor,
    ToolError, ToolEvent, ToolExecutionKey, ToolExecutionStatus,
    ToolOperationDescriptor, ToolResult, ToolResultResponse,
    control_request_from_payload, invoke_request_from_envelope,
    make_control_response_envelope, make_event_envelope,
    make_invoke_response_envelope, make_registration_envelope,
    make_result_response_envelope, make_status_response_envelope,
)
from forge_tool.dora import tool_envelope_to_message, tool_message_to_envelope


class ProviderFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass
class _Record:
    context: Any
    phase: str = "accepted"
    result: ToolResult | None = None
    error: ToolError | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    sequence: int = 0


Handler = Callable[[dict[str, Any], threading.Event, Callable[[dict[str, Any]], None]], dict[str, Any]]
InputHandler = Callable[[str, Any, dict[str, Any]], None]


class Provider:
    """One concurrency-safe Forge query or action provider."""

    def __init__(self, *, endpoint_id: str, operation: str, semantics: str, handler: Handler, input_handler: InputHandler | None = None):
        self.endpoint_id = endpoint_id
        self.operation = operation
        self.semantics = semantics
        self.handler = handler
        self.input_handler = input_handler
        self.instance_id = f"{endpoint_id}-{uuid.uuid4()}"
        self.node = Node()
        self.outputs: queue.Queue[tuple[str, Any, dict[str, Any] | None]] = queue.Queue()
        self.records: dict[ToolExecutionKey, _Record] = {}
        self.lock = threading.Lock()
        self.descriptor = ToolEndpointDescriptor(
            protocol_version=TOOL_ENDPOINT_PROTOCOL,
            endpoint_id=endpoint_id,
            operations=(ToolOperationDescriptor(
                name=operation,
                semantics=semantics,
                cancellable=semantics == "action",
                status_supported=semantics == "action",
                max_concurrency=1,
            ),),
        )

    def send_output(self, output_id: str, value: Any, *, metadata: dict[str, Any] | None = None) -> None:
        self.outputs.put((output_id, value, metadata))

    def _emit(self, envelope: Any) -> None:
        self.send_output("tool_out", tool_envelope_to_message(envelope).to_arrow())

    def _register(self) -> None:
        self._emit(make_registration_envelope(self.descriptor, endpoint_instance_id=self.instance_id, request_id=f"register-{uuid.uuid4()}"))

    def _progress(self, record: _Record, data: dict[str, Any]) -> None:
        with self.lock:
            sequence, record.sequence = record.sequence, record.sequence + 1
        self._emit(make_event_envelope(ToolEvent(type="progress", data=data), record.context, endpoint_instance_id=self.instance_id, sequence=sequence))

    def _run_action(self, record: _Record, arguments: dict[str, Any]) -> None:
        with self.lock:
            record.phase = "running"
        error = None
        try:
            outputs = self.handler(arguments, record.cancel, lambda data: self._progress(record, data))
            phase, status, event_type = (("cancelled", "cancelled", "cancelled") if record.cancel.is_set() else ("completed", "succeeded", "executor_completed"))
            result = ToolResult(status=status, outputs=outputs)
        except ProviderFailure as exc:
            error = ToolError(code=exc.code, message=str(exc), retryable=False, details=exc.details)
            phase, event_type, result = "failed", "executor_failed", ToolResult(status="failed", error=error)
        except Exception as exc:
            error = ToolError(code="REKEP_INTERNAL", message=str(exc), retryable=False)
            phase, event_type, result = "failed", "executor_failed", ToolResult(status="failed", error=error)
        with self.lock:
            record.phase, record.result, record.error = phase, result, error
            sequence, record.sequence = record.sequence, record.sequence + 1
        data: dict[str, Any] = {"status": result.status}
        if error:
            data["error"] = {"code": error.code, "message": error.message}
        self._emit(make_event_envelope(ToolEvent(type=event_type, data=data), record.context, endpoint_instance_id=self.instance_id, sequence=sequence))

    def _handle_tool(self, value: Any) -> None:
        envelope = tool_message_to_envelope(ToolMessage.from_arrow(value))
        if envelope.message_type == "endpoint.registry.response":
            return
        key = ToolExecutionKey(invocation_id=envelope.invocation_id or "missing", attempt_id=envelope.attempt_id or "missing")
        if envelope.message_type == "tool.invoke.request":
            request, context = invoke_request_from_envelope(envelope)
            arguments = dict(request.arguments)
            if self.semantics == "query":
                try:
                    outcome: Any = ToolResult(status="succeeded", outputs=self.handler(arguments, threading.Event(), lambda _: None))
                except ProviderFailure as exc:
                    outcome = ToolError(code=exc.code, message=str(exc), retryable=False, details=exc.details)
                except Exception as exc:
                    outcome = ToolError(code="REKEP_INTERNAL", message=str(exc), retryable=False)
                self._emit(make_invoke_response_envelope(outcome, envelope))
                return
            with self.lock:
                busy = any(item.phase in {"accepted", "running"} for item in self.records.values())
                if not busy:
                    record = _Record(context=context)
                    self.records[key] = record
            if busy:
                self._emit(make_invoke_response_envelope(ToolError(code="REKEP_BUSY", message="endpoint already has an active invocation"), envelope))
                return
            self._emit(make_invoke_response_envelope(ToolAccepted(details={"accepted": True}), envelope))
            threading.Thread(target=self._run_action, args=(record, arguments), daemon=True).start()
            return
        with self.lock:
            record = self.records.get(key)
        if envelope.message_type == "tool.status.request":
            status = (
                ToolExecutionStatus(phase=record.phase, error=record.error)
                if record
                else ToolExecutionStatus(phase="unknown", error=ToolError(code="REKEP_NOT_FOUND", message="invocation is not retained"))
            )
            self._emit(make_status_response_envelope(status, envelope))
        elif envelope.message_type == "tool.result.request":
            response = ToolResultResponse(status="not_found" if record is None else ("pending" if record.result is None else "available"), result=record.result if record and record.result else None)
            self._emit(make_result_response_envelope(response, envelope))
        elif envelope.message_type == "tool.control.request":
            command, _ = control_request_from_payload(envelope.payload)
            if record is None:
                response = ToolControlResponse(command=command, status="rejected", error=ToolError(code="REKEP_NOT_FOUND", message="invocation is not retained"))
            elif record.result is not None:
                response = ToolControlResponse(command=command, status="terminal")
            else:
                record.cancel.set()
                response = ToolControlResponse(command=command, status="accepted")
            self._emit(make_control_response_envelope(response, envelope))

    def _drain(self) -> None:
        while True:
            try:
                output_id, batch, metadata = self.outputs.get_nowait()
            except queue.Empty:
                return
            if metadata is None:
                self.node.send_output(output_id, batch)
            else:
                self.node.send_output(output_id, batch, metadata=metadata)

    def run(self) -> None:
        self._register()
        last_register = time.monotonic()
        for event in self.node:
            if event.get("type") == "STOP":
                break
            if event.get("type") == "INPUT":
                input_id = str(event.get("id"))
                if input_id == "tool_in":
                    self._handle_tool(event["value"])
                elif input_id == "tick" and time.monotonic() - last_register >= 1.0:
                    self._register()
                    last_register = time.monotonic()
                elif self.input_handler is not None:
                    self.input_handler(input_id, event["value"], event.get("metadata", {}))
            self._drain()
        self._drain()
