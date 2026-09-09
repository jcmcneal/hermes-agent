---
title: Persistent plugin sessions
---

Trusted local plugins running inside `hermes serve` or the dashboard can use
`ctx.sessions` to execute turns in persistent profile sessions. This uses the existing
backend's session runtime; it starts no CLI process or separately supervised worker.
The service is installed before dashboard router lifespans start and stops with the
backend. It is unavailable in an ordinary standalone CLI/plugin process.

```python
sessions = ctx.sessions
binding = sessions.ensure_session(
    principal_id="account:alice", profile="reviewer", conversation_key="task:123",
    title="Review task 123",
)
receipt = sessions.submit(
    principal_id="account:alice", profile="reviewer", conversation_key="task:123",
    operation_key="task:123:revision:2:review", text="Review revision 2.",
    max_turns=12,
    session_env={"MY_PLUGIN_TASK_ID": "123"},
)
receipt = sessions.status(
    principal_id="account:alice", operation_key="task:123:revision:2:review",
)
```

These methods are synchronous admission/storage operations. Model execution happens in
the backend's turn thread. Async consumers should call them through `asyncio.to_thread`.
The facade is also available through
`tui_gateway.plugin_sessions.get_session_service(plugin_id)` for dashboard adapters
without a retained `PluginContext`.

## Identity and authorization

The plugin namespace, principal ID, profile and conversation key bind one durable
session identity. Reusing that tuple resumes the saved session; never use a display title
as an identity. Native compression may rotate the live session to a successor while the
binding retains its durable lineage anchor. The service verifies that anchor in the
requested profile's store before native resume, preventing title fallback and legacy
cross-profile adoption.

This is an in-process API for trusted plugin code, **not an authentication boundary**.
The plugin must authorize the requesting principal, validate target profile access and
check enabled configuration before calling it. A principal ID is supplied by that trusted
adapter, not accepted blindly from an HTTP body. The facade does not enable a plugin's
tools on a profile where they are disabled. Backend administrators can still discover
hidden sessions and inspect their history through the native session APIs: facade
principal isolation is not per-user transcript secrecy against a backend administrator.
Ordinary session RPC mutations targeting these sessions are fenced; cold resume preserves
the durable plugin source even if a client supplies a different source. The execution
journal is created with owner-only file permissions on POSIX hosts.

## Delivery and recovery

`submit` durably journals its operation key before admitting a native prompt. Repeating
an identical key and request returns the same receipt. Changing the profile, conversation,
text, turn limit or session environment under that key raises `SessionServiceConflict`.
The journal contains execution receipts, not the plugin's business queue or workflow data.

Receipt states are `running`, `completed`, `failed`, `cancelled` and `indeterminate`.
Completed output is in `result["text"]`. `error` describes a failure. `not_admitted=True`
proves a synchronous rejection before a turn thread started; ordinary failed turns may
already have invoked tools. The independent `active` flag tracks whether that operation
still occupies its native runtime. A `pending_approval` payload, when present, identifies
the native approval request; approval policy is never automatically weakened.

A backend restart changes unfinished admissions to `indeterminate`. Neither the service
nor native crash recovery automatically repeats them. Status polling identifies a stopped runtime without a terminal receipt; `reconcile()`
can perform the same check across the service journal. A conversation
with indeterminate work rejects new operations until the caller explicitly cancels that
operation after deciding how to handle possible side effects.

`cancel(principal_id=..., operation_key=...)` durably fences the receipt before requesting
interruption. A late model completion cannot replace the cancellation or become a
publishable result. Cancellation cannot undo tool side effects. Wait until `active=False`
before releasing the conversation lane: a tool may still be finishing after cancellation.
Use a new operation key only for an intentional retry. Never automatically retry unknown
admission errors with a new key.

## Tool context and limits

Optional `session_env` is a dictionary of shell-safe variable names and string values.
It cannot override `HERMES_*` identity. Python tools read it with
`gateway.session_context.get_session_env`. Local terminal subprocesses inherit it;
the shared shell snapshot excludes these values so later turns do not inherit stale
routing identifiers. The service does not mutate `os.environ`. Remote terminal backends
do not currently implement this plugin context bridge.

`max_turns` temporarily changes the existing agent's iteration limit and restores the
previous limit after the turn. Profile config, secrets, history and prompt caching use the
same scopes as ordinary native session execution.

Isolated compute workers are currently rejected before admission, as with hosted room
turns. The service requires the existing in-process native runtime. It does not add an
external daemon or restart a stopped backend.
