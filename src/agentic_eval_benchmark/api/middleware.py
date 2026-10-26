"""Pure-ASGI request context: request IDs and a hard request-body size limit."""

import re
import uuid
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from agentic_eval_benchmark.api.problems import make_problem, problem_response

_REQUEST_ID_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _BodyTooLargeError(Exception):
    pass


class RequestContextMiddleware:
    """Assign ``X-Request-ID`` (echoing a well-formed client ID) and cap body size.

    The limit is enforced on ``Content-Length`` up front and on the bytes actually
    received, so chunked uploads cannot bypass it.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = False

        async def send_with_id(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode()),
                ]
            await send(message)

        declared = headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._too_large(scope, receive, send_with_id, request_id)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send_with_id)
        except _BodyTooLargeError:
            if started:
                raise
            await self._too_large(scope, receive, send_with_id, request_id)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send, request_id: str) -> None:
        problem = make_problem(
            413,
            "payload-too-large",
            "Content Too Large",
            f"Request bodies are limited to {self.max_body_bytes} bytes.",
            request_id=request_id,
        )
        await problem_response(problem)(scope, receive, send)
