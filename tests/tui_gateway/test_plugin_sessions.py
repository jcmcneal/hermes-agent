"""Persistent plugin sessions drive the installed handlers and real profile SQLite stores."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tui_gateway import plugin_sessions, server
from tests.tui_gateway.test_auto_continue import _InlineThread

_REAL_THREAD = threading.Thread


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(server, '_hermes_home', home)
    monkeypatch.setattr('gateway.status._get_lock_dir', lambda: home / 'locks')
    db = SessionDB(db_path=home / 'state.db')
    monkeypatch.setattr(server, '_db', db)
    monkeypatch.setattr(server, '_db_error', None)
    monkeypatch.setattr(server, '_sessions', {})
    monkeypatch.setattr(server, '_load_cfg', lambda: {})
    monkeypatch.setattr(server, '_load_dashboard_process_isolation_config', lambda: {})
    monkeypatch.setattr(server, '_schedule_agent_build', lambda *a, **kw: None)
    monkeypatch.setattr(server, '_schedule_session_cap_enforcement', lambda: None)
    monkeypatch.setattr(server, '_ensure_active_session_slot', lambda *a: None)
    monkeypatch.setattr(server, '_enable_gateway_prompts', lambda: None)
    monkeypatch.setattr(server, '_emit', lambda *a, **kw: None)
    monkeypatch.setattr(server, '_wire_callbacks', lambda sid: None)
    monkeypatch.setattr(server, '_sync_agent_model_with_config', lambda *a: None)
    monkeypatch.setattr(server, '_sync_agent_compression_with_config', lambda *a: None)
    monkeypatch.setattr(server, '_sync_session_key_after_compress', lambda *a, **kw: None)
    monkeypatch.setattr(server, '_tts_stream_begin', lambda: None)
    monkeypatch.setattr(server, '_get_usage', lambda agent: {})
    monkeypatch.setattr(server.threading, 'Thread', _InlineThread)
    calls = []
    def build(sid, record):
        if record.get('agent') is not None:
            return
        def run(text, **kwargs):
            from gateway.session_context import get_session_env
            from hermes_constants import get_hermes_home
            calls.append(dict(text=text, session_id=record['session_key'],
                              history=list(kwargs.get('conversation_history', [])),
                              profile_home=str(get_hermes_home()),
                              peer=get_session_env('BOT_COMS_PEER_ID'),
                              principal=get_session_env('HERMES_SESSION_USER_ID'),
                              profile=get_session_env('HERMES_SESSION_PROFILE'),
                              max_turns=record['agent'].max_iterations))
            messages = [*kwargs.get('conversation_history', []),
                        {'role': 'user', 'content': text},
                        {'role': 'assistant', 'content': 'reply:' + text}]
            with server._profile_db({'profile': 'work' if record.get('profile_home') else 'default'}) as store:
                store.append_message(record['session_key'], 'user', text)
                store.append_message(record['session_key'], 'assistant', 'reply:' + text)
            return {'final_response': 'reply:' + text, 'messages': messages}
        record['agent'] = SimpleNamespace(session_id=record['session_key'], run_conversation=run,
                                        clear_interrupt=lambda: None, model='test', provider='test',
                                        max_iterations=99)
        record['agent_ready'].set()
    monkeypatch.setattr(server, '_start_agent_build', build)
    host = plugin_sessions.install_session_service(server, db_path=home / 'operations.db')
    yield plugin_sessions.get_session_service('example'), host, calls, home
    plugin_sessions.stop_session_service()
    server._sessions.clear()
    db.close()


def request(**extra):
    return dict(principal_id='alice', profile='default', conversation_key='dm',
                operation_key='op-1', text='hello', **extra)


def test_exact_session_reuse_and_durable_duplicate_receipts(runtime):
    service, host, calls, home = runtime
    first = service.submit(**request(session_env={'BOT_COMS_PEER_ID': 'reviewer'}, max_turns=12))
    assert first['status'] == 'completed', first
    assert first['result']['text'] == 'reply:hello'
    again = service.submit(**request(session_env={'BOT_COMS_PEER_ID': 'reviewer'}, max_turns=12))
    assert again == first
    assert len(calls) == 1
    second = service.submit(**{**request(), 'operation_key': 'op-2', 'text': 'follow up'})
    assert second['status'] == 'completed', second
    assert second['session_id'] == first['session_id']
    assert calls[1]['history'][-1]['content'] == 'reply:hello'
    assert calls[0]['peer'] == 'reviewer' and calls[1]['peer'] == ''
    assert calls[0]['max_turns'] == 12 and calls[1]['max_turns'] == 99
    with pytest.raises(plugin_sessions.SessionServiceConflict):
        service.submit(**{**request(), 'text': 'different'})
    assert service.status(principal_id='bob', operation_key='op-1') is None
    assert plugin_sessions.get_session_service('another').status(principal_id='alice', operation_key='op-1') is None
    plugin_sessions.stop_session_service()
    plugin_sessions.install_session_service(server, db_path=home / 'operations.db')
    restored = plugin_sessions.get_session_service('example')
    assert restored.status(principal_id='alice', operation_key='op-1') == first


def test_profile_and_principal_bindings_are_isolated(runtime):
    service, host, calls, home = runtime
    work = home / 'profiles' / 'work'
    work.mkdir(parents=True)
    one = service.submit(**request())
    two = service.submit(**{**request(), 'profile': 'work', 'operation_key': 'work-op'})
    three = service.submit(**{**request(), 'principal_id': 'bob'})
    assert all(r['status'] == 'completed' for r in (one, two, three)), (one,two,three)
    assert len({r['session_id'] for r in (one, two, three)}) == 3
    assert calls[1]['profile_home'] == str(work)
    assert calls[1]['profile'] == 'work' and calls[1]['principal'] == 'alice'
    assert calls[2]['principal'] == 'bob'
    with server._profile_db({'profile': 'default'}) as default:
        assert default.get_session(two['session_id']) is None
    with server._profile_db({'profile': 'work'}) as scoped:
        assert scoped.get_session(two['session_id']) is not None
        assert scoped.get_session(one['session_id']) is None


def test_cancel_fences_late_completion_and_waits_for_runtime_drain(runtime, monkeypatch):
    service, host, calls, home = runtime
    pending = []
    def hold(rid, sid, record, text, display_kind, callback):
        pending.append((record, callback))
        record["_run_thread"] = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(server, '_run_after_agent_ready', hold)
    result = service.submit(**request())
    assert result['status'] == 'running'
    record, callback = pending[0]
    monkeypatch.setattr(server, '_interrupt_session_turn', lambda *a, **k: None)
    cancelled = service.cancel(principal_id='alice', operation_key='op-1')
    assert cancelled['status'] == 'cancelled' and cancelled['active']
    callback({'status': 'settled', 'text': 'late result'})
    assert service.status(principal_id='alice', operation_key='op-1')['result'] is None
    with pytest.raises(plugin_sessions.SessionServiceConflict):
        service.submit(**{**request(), 'operation_key': 'op-2'})
    record['running'] = False
    record['_run_thread'] = SimpleNamespace(is_alive=lambda: False)
    assert not service.status(principal_id='alice', operation_key='op-1')['active']


def test_recovery_never_replays_ambiguous_admission(runtime, monkeypatch):
    service, host, calls, home = runtime
    def hold(rid, sid, record, *args):
        record['_run_thread'] = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(server, '_run_after_agent_ready', hold)
    running = service.submit(**request())
    assert running['status'] == 'running'
    # Simulate process loss: release the host lock without graceful cancellation.
    host._closed = True
    host._release_lock()
    plugin_sessions._host = None
    server._sessions.clear()
    plugin_sessions.install_session_service(server, db_path=home / 'operations.db')
    recovered = plugin_sessions.get_session_service('example')
    receipt = recovered.submit(**request())
    assert receipt['status'] == 'indeterminate'
    assert not calls
    with pytest.raises(plugin_sessions.SessionServiceConflict):
        recovered.submit(**{**request(), 'operation_key': 'op-2'})


def test_json_cannot_forge_internal_turn(runtime):
    service, host, calls, home = runtime
    binding = service.ensure_session(principal_id='alice', profile='default', conversation_key='dm')
    sid = next(iter(server._sessions))
    response = server._methods['prompt.submit']('forged', {
        'session_id': sid, 'text': 'wrong', '_plugin_turn': {'owner': ['example','alice','default','dm']}})
    assert 'error' in response
    assert not calls


def test_agent_build_failure_persists_terminal_receipt(runtime, monkeypatch):
    service, host, calls, home = runtime
    monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: {'error': {'message': 'build failed'}})
    outcome = service.submit(**request(max_turns=12))
    assert outcome['status'] == 'failed'
    assert 'build failed' in outcome['error']
    assert not calls


def test_plugin_context_reaches_local_children_without_snapshot_leak(tmp_path):
    from gateway.session_context import plugin_session_env
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    try:
        env.init_session()
        for peer in ('first', 'second'):
            with plugin_session_env({'BOT_COMS_PEER_ID': peer}):
                result = env.execute('printf "peer:%s\\n" "$BOT_COMS_PEER_ID"')
            assert 'peer:' + peer in result['output']
        assert 'BOT_COMS_PEER_ID' not in Path(env._snapshot_path).read_text()
        result = env.execute('printf "peer:%s\\n" "$BOT_COMS_PEER_ID"')
        assert 'peer:first' not in result['output'] and 'peer:second' not in result['output']
    finally:
        env.cleanup()


@pytest.mark.parametrize('compress', [False, True])
def test_only_proven_compression_successors_are_resumed(runtime, compress):
    service, host, calls, home = runtime
    first = service.submit(**request())
    anchor = first['session_id']
    with server._profile_db({'profile': 'default'}) as db:
        if compress:
            db.end_session(anchor, 'compression')
        db.create_session('successor', source='plugin_session', parent_session_id=anchor)
        db.append_message('successor', 'user', 'child question')
        db.append_message('successor', 'assistant', 'child reply')
    server._sessions.clear()  # cold resume, as after backend restart
    second = service.submit(**{**request(), 'operation_key': 'op-2', 'text': 'next'})
    assert second['status'] == 'completed', second
    assert second['session_id'] == anchor
    assert calls[-1]['session_id'] == ('successor' if compress else anchor)
    assert calls[-1]['history'][-1]['content'] == ('child reply' if compress else 'reply:hello')


def test_live_callback_blocked_on_admission_lock_is_not_indeterminate(runtime, monkeypatch):
    service, host, calls, home = runtime
    entered = threading.Event()
    original_complete = host._complete
    original_call = host._call
    def complete(*args):
        entered.set()
        original_complete(*args)
    def call(method, **params):
        result = original_call(method, **params)
        if method == 'prompt.submit':
            assert entered.wait(3), 'terminal callback never reached admission lock'
        return result
    monkeypatch.setattr(host, '_complete', complete)
    monkeypatch.setattr(host, '_call', call)
    monkeypatch.setattr(server.threading, 'Thread', _REAL_THREAD)
    monkeypatch.setattr(server, '_wait_agent_for_prompt', lambda *a: {'error': {'message': 'build failed'}})
    outcome = service.submit(**request())
    # The native wrapper cleared running, but its callback is still a live thread waiting
    # for submit's lock. Only that terminal callback can settle the receipt.
    assert outcome['status'] == 'running', outcome
    record = next(iter(server._sessions.values()))
    record['_run_thread'].join(timeout=3)
    assert service.status(principal_id='alice', operation_key='op-1')['status'] == 'failed'


def test_shutdown_fences_all_operations_even_when_interrupt_raises(runtime, monkeypatch):
    service, host, calls, home = runtime
    def hold(rid, sid, record, *args):
        record['_run_thread'] = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(server, '_run_after_agent_ready', hold)
    for principal in ('alice', 'bob'):
        service.submit(**{**request(), 'principal_id': principal})
    original_call = host._call
    def call(method, **params):
        if method == 'session.interrupt':
            raise RuntimeError('interrupt unavailable')
        return original_call(method, **params)
    monkeypatch.setattr(host, '_call', call)
    plugin_sessions.stop_session_service()
    assert host._closed
    plugin_sessions.install_session_service(server, db_path=home / 'operations.db')
    restored = plugin_sessions.get_session_service('example')
    for principal in ('alice', 'bob'):
        assert restored.status(principal_id=principal, operation_key='op-1')['status'] == 'cancelled'


def test_cold_resume_cannot_remove_plugin_ownership_with_source_override(runtime):
    service, host, calls, home = runtime
    binding = service.ensure_session(principal_id='alice', profile='default', conversation_key='dm')
    server._sessions.clear()
    resumed = server._methods['session.resume']('spoof', {
        'session_id': binding['session_id'], 'profile': 'default', 'source': 'desktop', 'omit_messages': True})
    sid = resumed['result']['session_id']
    assert server._sessions[sid]['source'] == 'plugin_session'
    response = server._methods['prompt.submit']('bypass', {'session_id': sid, 'text': 'bypass journal'})
    assert 'error' in response
    assert not calls
    normal = service.submit(**request())
    assert normal['status'] == 'completed', normal


@pytest.mark.parametrize('method', ['session.interrupt', 'session.undo', 'session.compress', 'session.close'])
def test_external_rpc_cannot_mutate_plugin_owned_session(runtime, method):
    service, host, calls, home = runtime
    binding = service.ensure_session(principal_id='alice', profile='default', conversation_key='dm')
    sid = next(iter(server._sessions))
    denied = server.handle_request({'id': 'untracked', 'method': method, 'params': {'session_id': sid}})
    assert denied['error']['code'] == 4122
    assert not calls


def test_cancel_does_not_wait_for_an_agent_to_build(runtime, monkeypatch):
    service, host, calls, home = runtime
    def hold(rid, sid, record, *args):
        record['_run_thread'] = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(server, '_start_agent_build', lambda *args: None)
    monkeypatch.setattr(server, '_run_after_agent_ready', hold)
    service.submit(**request())
    def unexpected_wait(*args):
        raise AssertionError('cancel must not wait for the agent')
    monkeypatch.setattr(server, '_sess', unexpected_wait)
    cancelled = service.cancel(principal_id='alice', operation_key='op-1')
    assert cancelled['status'] == 'cancelled'
    assert next(iter(server._sessions.values()))['_turn_cancel_requested']


def test_execution_journal_is_private(runtime):
    service, host, calls, home = runtime
    assert host.db_path.stat().st_mode & 0o777 == 0o600


def test_authorized_native_approval_response_still_reaches_existing_handler(runtime, monkeypatch):
    service, host, calls, home = runtime
    service.submit(**request())
    sid = next(iter(server._sessions))
    resolved = []
    monkeypatch.setattr('tools.approval.resolve_gateway_approval',
                        lambda session_id, choice, **kw: resolved.append((session_id, choice, kw)) or 1)
    response = server.handle_request({'id': 'approval', 'method': 'approval.respond', 'params': {
        'session_id': sid, 'request_id': 'exact-request', 'choice': 'once', 'all': False}})
    assert response['result']['resolved'] == 1
    assert resolved == [(server._sessions[sid]['session_key'], 'once',
                         {'resolve_all': False, 'request_id': 'exact-request'})]
