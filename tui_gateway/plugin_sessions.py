"""Backend-owned persistent session execution for trusted local plugins.

The execution journal records admission, never business messages or workflow state. A lost
backend cannot prove whether tools ran: unfinished admissions become indeterminate and are
never replayed. Callers must retain their operation key until they consume the receipt.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class SessionServiceUnavailable(RuntimeError):
    pass


class SessionServiceConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class _PluginTurn:
    """Python-only admission proof; JSON/RPC clients cannot construct this object."""
    owner: tuple[str, str, str, str]
    operation_key: str
    on_terminal: Callable[[dict], None]
    session_env: dict[str, str]
    max_turns: int | None


def validate_plugin_turn(turn: Any, session: dict) -> bool:
    return isinstance(turn, _PluginTurn) and session.get("_plugin_session_owner") == turn.owner


# Ordinary backend clients may inspect administrator-visible transcripts, but cannot
# bypass a plugin's operation ledger to mutate an owned conversation.
_READ_SESSION_METHODS = frozenset({
    "session.list", "session.most_recent", "session.active_list", "session.resume",
    "session.history", "session.status", "session.usage", "session.context_breakdown",
    "session.events.since", "session.events.stats", "session.control.read", "session.create",
})


def plugin_session_mutation_error(server, method: str, params: dict) -> str | None:
    if not ((method.startswith("session.") and method not in _READ_SESSION_METHODS)
            or method.startswith("prompt.") or method in {"slash.exec", "command.dispatch"}):
        return None
    target = params.get("session_id")
    if not isinstance(target, str) or not target:
        return None
    with server._sessions_lock:
        record = server._sessions.get(target)
        if record is None:
            record = next((value for value in server._sessions.values()
                           if value.get("session_key") == target), None)
    if record is not None:
        owned = record.get("source") == "plugin_session" or bool(record.get("_plugin_session_owner"))
    else:
        with server._profile_db(params) as db:
            row = db.get_session(target) if db is not None else None
            owned = row is not None and row.get("source") == "plugin_session"
    if not owned:
        return None
    if method == "prompt.submit" and record is not None and validate_plugin_turn(params.get("_plugin_turn"), record):
        return None
    return "this session is managed by its plugin; use its execution API"


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


class PluginSessionService:
    """A namespace-bound facade. Methods return snapshots, never mutable session objects."""
    def __init__(self, host: _SessionHost, plugin_id: str):
        self._host, self.plugin_id = host, _required(plugin_id, "plugin_id")

    def ensure_session(self, *, principal_id: str, profile: str, conversation_key: str,
                       title: str | None = None) -> dict:
        return self._host.ensure_session(self.plugin_id, principal_id, profile, conversation_key, title)

    def submit(self, *, principal_id: str, profile: str, conversation_key: str,
               operation_key: str, text: str, title: str | None = None,
               session_env: dict[str, str] | None = None, max_turns: int | None = None) -> dict:
        return self._host.submit(self.plugin_id, principal_id, profile, conversation_key,
                                 operation_key, text, title, session_env or {}, max_turns)

    def status(self, *, principal_id: str, operation_key: str) -> dict | None:
        return self._host.status(self.plugin_id, principal_id, operation_key)

    def cancel(self, *, principal_id: str, operation_key: str) -> dict | None:
        return self._host.cancel(self.plugin_id, principal_id, operation_key)

    def reconcile(self) -> None:
        self._host.reconcile()


class _SessionHost:
    def __init__(self, server, db_path: Path):
        self.server, self.db_path = server, db_path.resolve()
        self._lock = threading.RLock()
        self._closed = False
        from gateway.status import acquire_scoped_lock
        acquired, _ = acquire_scoped_lock("plugin-sessions", str(self.db_path))
        if not acquired:
            raise SessionServiceUnavailable("Another backend owns plugin session execution")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.db_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                self.db_path.chmod(0o600)
            finally:
                os.close(descriptor)
            with self._db() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS bindings (
                        plugin TEXT, principal TEXT, profile TEXT, conversation TEXT,
                        session_id TEXT NOT NULL,
                        PRIMARY KEY(plugin, principal, profile, conversation));
                    CREATE TABLE IF NOT EXISTS operations (
                        plugin TEXT, principal TEXT, operation_key TEXT,
                        profile TEXT NOT NULL, conversation TEXT NOT NULL,
                        session_id TEXT NOT NULL, runtime_id TEXT,
                        request_hash TEXT NOT NULL, status TEXT NOT NULL,
                        result TEXT, error TEXT, updated_at REAL NOT NULL,
                        PRIMARY KEY(plugin, principal, operation_key));
                """)
                db.execute("UPDATE operations SET status='indeterminate', error=?, updated_at=? "
                           "WHERE status='running'", ("Backend stopped before a durable terminal receipt", time.time()))
        except BaseException:
            self._release_lock()
            raise

    def _release_lock(self):
        from gateway.status import release_scoped_lock
        release_scoped_lock("plugin-sessions", str(self.db_path))

    def _db(self):
        if self._closed:
            raise SessionServiceUnavailable("Plugin session execution is stopping")
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        # Connection context commits but does not close: use a closing transaction helper.
        return _transaction(db)

    def _call(self, method, **params):
        response = self.server._methods[method]("plugin-session", params)
        if error := response.get("error"):
            raise SessionServiceConflict(f"{method}: {error.get('message', error)}")
        return response["result"]

    def _owner(self, plugin, principal, profile, conversation):
        return tuple(_required(v, k) for v, k in zip(
            (plugin, principal, profile, conversation), ("plugin", "principal_id", "profile", "conversation_key")))

    def _binding(self, db, owner):
        return db.execute("SELECT * FROM bindings WHERE plugin=? AND principal=? AND profile=? AND conversation=?",
                          owner).fetchone()

    def _resume(self, binding):
        profile, stored = binding["profile"], binding["session_id"]
        # Refuse title fallback or legacy cross-profile adoption in session.resume.
        with self.server._profile_db({"profile": profile}) as db:
            if db is None or db.get_session(stored) is None:
                raise SessionServiceConflict("The exact owned session is missing from its profile store")
            tip = db.get_compression_tip(stored) or stored
            tip_row = db.get_session(tip)
            if not tip_row or tip_row.get("source") != "plugin_session":
                raise SessionServiceConflict("Session continuation does not belong to the plugin runtime")
        resumed = self._call("session.resume", profile=profile, session_id=stored,
                             omit_messages=True, source="plugin_session")
        sid = resumed["session_id"]
        owner = tuple(binding[k] for k in ("plugin", "principal", "profile", "conversation"))
        record = self.server._sessions[sid]
        existing = record.get("_plugin_session_owner")
        if existing is not None and existing != owner:
            raise SessionServiceConflict("Session belongs to another plugin principal")
        record["_plugin_session_owner"] = owner
        record["close_on_disconnect"] = False
        return sid

    def ensure_session(self, plugin, principal, profile, conversation, title):
        owner = self._owner(plugin, principal, profile, conversation)
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            binding = self._binding(db, owner)
            if binding is None:
                created = self._call("session.create", profile=profile, source="plugin_session", hidden=True,
                                     title=title or f"{plugin}: {conversation}",
                                     close_on_disconnect=False, follow_profile_config=True)
                sid, stored = created["session_id"], created["stored_session_id"]
                record = self.server._sessions[sid]
                record["_plugin_session_owner"] = owner
                # Use the installed helper: sibling functions bind server globals at registration.
                if self.server._ensure_session_db_row(record) is False:
                    raise SessionServiceUnavailable("Session storage unavailable")
                with self.server._profile_db({"profile": profile}) as sessions:
                    if sessions is None or sessions.get_session(stored) is None:
                        raise SessionServiceUnavailable("Session identity was not durably saved")
                db.execute("INSERT INTO bindings VALUES (?,?,?,?,?)", (*owner, stored))
                binding = self._binding(db, owner)
            else:
                self._resume(binding)
            return dict(profile=profile, conversation_key=conversation, session_id=binding["session_id"])

    def submit(self, plugin, principal, profile, conversation, operation, text, title, session_env, max_turns):
        owner = self._owner(plugin, principal, profile, conversation)
        _required(operation, "operation_key")
        _required(text, "text")
        if max_turns is not None and (type(max_turns) is not int or max_turns <= 0):
            raise ValueError("max_turns must be a positive integer")
        if not isinstance(session_env, dict) or any(
            not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or k.startswith("HERMES_") or
            not isinstance(v, str) for k, v in session_env.items()):
            raise ValueError("session_env must contain string values and cannot override HERMES_* identity")
        digest = hashlib.sha256(json.dumps([profile, conversation, text, session_env, max_turns],
                                          sort_keys=True).encode()).hexdigest()
        with self._lock:
            existing = self.status(plugin, principal, operation)
            if existing is not None:
                with self._db() as db:
                    row = self._operation(db, plugin, principal, operation)
                    if row["request_hash"] != digest:
                        raise SessionServiceConflict("Operation key was already used for a different request")
                return existing
            self.ensure_session(plugin, principal, profile, conversation, title)
            with self._db() as db:
                db.execute("BEGIN IMMEDIATE")
                binding = self._binding(db, owner)
                pending = db.execute("SELECT operation_key FROM operations WHERE plugin=? AND principal=? "
                                     "AND profile=? AND conversation=? AND status IN ('running','indeterminate')",
                                     owner).fetchone()
                if pending:
                    raise SessionServiceConflict("Conversation has unfinished or indeterminate work")
                sid = self._resume(binding)
                if self.server._sessions[sid].get("running"):
                    raise SessionServiceConflict("The prior session turn is still active")
                db.execute("INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (plugin, principal, operation, profile, conversation, binding["session_id"], sid,
                            digest, "running", None, None, time.time()))
            turn = _PluginTurn(owner, operation,
                               lambda result: self._complete(plugin, principal, operation, result),
                               dict(session_env), max_turns)
            try:
                self._call("prompt.submit", profile=profile, session_id=sid, text=text, _plugin_turn=turn)
            except SessionServiceConflict as exc:
                # An error envelope proves synchronous rejection before the turn thread was started.
                self._complete(plugin, principal, operation, {"status": "failed", "error": str(exc), "not_admitted": True})
            except BaseException:
                with self._db() as db:
                    db.execute("UPDATE operations SET status='indeterminate', error=?, updated_at=? "
                               "WHERE plugin=? AND principal=? AND operation_key=? AND status='running'",
                               ("Admission outcome unknown", time.time(), plugin, principal, operation))
                raise
            return self.status(plugin, principal, operation)

    def _operation(self, db, plugin, principal, operation):
        return db.execute("SELECT * FROM operations WHERE plugin=? AND principal=? AND operation_key=?",
                          (plugin, principal, operation)).fetchone()

    def _runtime_active(self, row):
        record = self.server._sessions.get(row["runtime_id"])
        turn = record.get("_plugin_turn") if record else None
        thread = record.get("_run_thread") if record else None
        return bool(record and thread is not None and thread.is_alive()
                    and turn is not None and turn.operation_key == row["operation_key"]
                    and turn.owner == (row["plugin"], row["principal"], row["profile"], row["conversation"]))

    def _snapshot(self, row):
        if row is None:
            return None
        record = self.server._sessions.get(row["runtime_id"])
        active = self._runtime_active(row)
        result = json.loads(row["result"]) if row["result"] else None
        pending = None
        if active and callable(reader := getattr(self.server, "_pending_approval_request_payload", None)):
            pending = reader(str(record.get("session_key") or ""))
        return {"pending_approval": pending, "not_admitted": bool(result and result.get("not_admitted")),
                "active": active, "operation_key": row["operation_key"], "profile": row["profile"],
                "conversation_key": row["conversation"], "session_id": row["session_id"],
                "status": row["status"], "result": result,
                "error": row["error"]}

    def status(self, plugin, principal, operation):
        with self._lock, self._db() as db:
            row = self._operation(db, plugin, principal, operation)
            if row is not None:
                self._reconcile_row(db, row)
            return self._snapshot(self._operation(db, plugin, principal, operation))

    def _complete(self, plugin, principal, operation, result):
        status = {"settled": "completed", "completed": "completed", "failed": "failed",
                  "cancelled": "cancelled"}.get(result.get("status"), "failed")
        with self._lock, self._db() as db:
            db.execute("UPDATE operations SET status=?, result=?, error=?, updated_at=? "
                       "WHERE plugin=? AND principal=? AND operation_key=? AND status='running'",
                       (status, json.dumps(dict(result)), result.get("error"), time.time(),
                        plugin, principal, operation))

    def cancel(self, plugin, principal, operation):
        with self._lock:
            with self._db() as db:
                row = self._operation(db, plugin, principal, operation)
                if row is None or row["status"] not in {"running", "indeterminate"}:
                    return self._snapshot(row)
                db.execute("UPDATE operations SET status='cancelled', result=NULL, updated_at=? "
                           "WHERE plugin=? AND principal=? AND operation_key=?",
                           (time.time(), plugin, principal, operation))
            # Fence is committed BEFORE interruption, including when a tool cannot stop promptly.
            record = self.server._sessions.get(row["runtime_id"])
            if record is not None:
                self._call("session.interrupt", session_id=row["runtime_id"], profile=row["profile"],
                           _expected_plugin_turn=(plugin, principal, row["profile"], row["conversation"], operation))
            return self.status(plugin, principal, operation)

    def _reconcile_row(self, db, row):
        if row["status"] == "running" and not self._runtime_active(row):
            db.execute("UPDATE operations SET status='indeterminate', error=?, updated_at=? "
                       "WHERE plugin=? AND principal=? AND operation_key=? AND status='running'",
                       ("Run ended without a durable terminal receipt", time.time(),
                        row["plugin"], row["principal"], row["operation_key"]))

    def reconcile(self):
        with self._lock, self._db() as db:
            for row in db.execute("SELECT * FROM operations WHERE status='running'").fetchall():
                self._reconcile_row(db, row)

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                with self._db() as db:
                    rows = db.execute("SELECT * FROM operations WHERE status='running'").fetchall()
                for row in rows:
                    try:
                        self.cancel(row["plugin"], row["principal"], row["operation_key"])
                    except Exception:
                        logging.getLogger(__name__).exception("Plugin turn could not be interrupted during shutdown")
            finally:
                self._closed = True
                self._release_lock()


@contextmanager
def _transaction(db):
    try:
        with db:
            yield db
    finally:
        db.close()


_host: _SessionHost | None = None
_host_lock = threading.RLock()


def install_session_service(server, *, db_path: Path | str | None = None):
    global _host
    from hermes_constants import get_hermes_home
    with _host_lock:
        if _host is None:
            _host = _SessionHost(server, Path(db_path or get_hermes_home() / "plugin-sessions.db"))
        return _host


def get_session_service(plugin_id: str) -> PluginSessionService:
    with _host_lock:
        if _host is None or _host._closed:
            raise SessionServiceUnavailable("Persistent plugin sessions require the running Hermes backend")
        return PluginSessionService(_host, plugin_id)


def stop_session_service():
    global _host
    with _host_lock:
        if _host is not None:
            try:
                _host.close()
            finally:
                _host = None
