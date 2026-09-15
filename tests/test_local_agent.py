"""Offline, model-free regression suite for local_agent.py.

No network/API calls anywhere in this file. Every test exercises a pure
function or a filesystem-only tool against a tmp_path workspace. This suite
exists because the mission brief's claimed "186/186 passing" controller
tests were not present in this checkout (verified: no tests/ directory
existed prior to this file) - this is the real, honest test count for the
generic controller mechanisms, built from scratch.

Deliberately does NOT hardcode any public-task name, answer, or verifier
value: fixtures below use synthetic instructions/deliverables shaped like
the public tasks' *contracts* (JSON audit report, key=value forensics
report, fix+pytest, CTF flag) without referencing find-sqli-login,
fix-sqli-login, etc. or any of their real content.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_agent as la  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Contract extraction and routing
# --------------------------------------------------------------------------- #

class TestClassifyMechanical:
    def test_fix_task_with_pytest_mention(self):
        assert la.classify_mechanical(
            "You are working in /app. Run pytest tests/ to verify. "
            "Analyse the code and fix the most critical security issues."
        ) == "fix"

    def test_audit_task_read_only_bug_bounty(self):
        assert la.classify_mechanical(
            "Perform a security audit as if preparing a bug bounty report. "
            "Do not modify application code. Write findings to security_report.json."
        ) == "audit"

    def test_forensics_task_by_deliverable_name(self):
        assert la.classify_mechanical(
            "Reconcile the incident and record your conclusion in incident_report.txt "
            "using key=value lines."
        ) == "forensics"

    def test_ctf_task_by_flag_mention(self):
        assert la.classify_mechanical(
            "Find the hidden flag{...} and write it to /app/flag.txt"
        ) == "ctf"

    def test_negated_fix_language_does_not_route_to_fix(self):
        # "no remediation is required" contains "remediat" but must not be
        # read as a real fix instruction.
        result = la.classify_mechanical(
            "Identify the vulnerability and document it. No remediation is required. "
            "Do not modify the code."
        )
        assert result != "fix"

    def test_do_not_modify_code_routes_to_audit_not_fix(self):
        assert la.classify_mechanical(
            "Investigate the login endpoint for SQL injection. Do not modify the code; "
            "just report your findings."
        ) == "audit"

    def test_bare_investigate_without_vuln_mention_is_generic(self):
        # Weak audit signal alone (no vulnerability mentioned) must not force
        # a confident "audit" classification.
        assert la.classify_mechanical("Investigate the service.") == "generic"


class TestExplicitDeliverablePath:
    def test_extracts_backtick_quoted_app_path(self):
        assert la.extract_explicit_deliverable_path(
            "Write your report to `/app/custom_output.json`."
        ) == "/app/custom_output.json"

    def test_extracts_non_app_relative_path(self):
        assert la.extract_explicit_deliverable_path(
            "Write the report to `output/report.json`."
        ) == "output/report.json"

    def test_no_explicit_path_returns_empty(self):
        assert la.extract_explicit_deliverable_path(
            "Just fix the bug, no report needed."
        ) == ""

    def test_explicit_path_takes_precedence_over_kind_default(self, tmp_path):
        # guess_deliverable must prefer an explicit path over the emergency
        # per-kind default even when the kind implies a different default name.
        instr = "Perform an audit. Write findings to `/app/custom/out.json`."
        result = la.guess_deliverable(instr, "audit", tmp_path)
        assert result == "/app/custom/out.json"

    def test_no_explicit_path_falls_back_to_kind_default(self, tmp_path):
        result = la.guess_deliverable("Perform a security audit.", "audit", tmp_path)
        assert result == str(tmp_path / "security_report.json")


class TestExplicitDeliverableFormat:
    def test_detects_json_format(self):
        assert la.extract_explicit_deliverable_format(
            "Write a JSON report with a findings array."
        ) == "json"

    def test_no_format_stated_returns_empty(self):
        assert la.extract_explicit_deliverable_format("Fix the bug.") == ""


class TestTaskContractMechanical:
    def test_audit_contract_has_no_modifies_files(self):
        c = la.extract_task_contract_mechanical(
            "Perform a security audit. Do not modify application code. "
            "Write findings to /app/security_report.json."
        )
        assert c.family == "audit"
        assert c.modifies_files is False
        assert c.deliverable_path == "/app/security_report.json"

    def test_fix_contract_modifies_files_true(self):
        c = la.extract_task_contract_mechanical(
            "Fix the SQL injection vulnerability. Run pytest tests/ afterwards."
        )
        assert c.family == "fix"
        assert c.modifies_files is True

    def test_ambiguous_instruction_marked_ambiguous(self):
        c = la.extract_task_contract_mechanical("Look at this and do something useful.")
        assert c.source == "ambiguous"


# --------------------------------------------------------------------------- #
# Workdir correctness
# --------------------------------------------------------------------------- #

class TestWorkdirCorrectness:
    def test_run_sync_quiet_uses_explicit_cwd_for_relative_command(self, tmp_path):
        (tmp_path / "marker.txt").write_text("hello", encoding="utf-8")
        code, out = la._run_sync_quiet("cat marker.txt", cwd=tmp_path)
        assert code == 0
        assert "hello" in out

    def test_run_sync_quiet_without_cwd_does_not_see_tmp_relative_file(self, tmp_path):
        (tmp_path / "marker.txt").write_text("hello", encoding="utf-8")
        code, out = la._run_sync_quiet("cat marker.txt", cwd=None)
        # Without an explicit cwd, a relative path scoped to tmp_path must
        # NOT resolve — this is the exact regression class fixed by making
        # every fix-kind pytest invocation pass cwd=workdir explicitly.
        assert code != 0

    def test_fix_kind_with_no_tests_directory_is_not_falsely_green(self, tmp_path):
        # No tests/ dir at all: pytest reports "no tests ran" / an error, not
        # "N passed" - must not be misread as success.
        ok, reason = la.verify_deliverable("fix", "", tmp_path)
        assert ok is False

    def test_verify_deliverable_fix_kind_runs_pytest_in_workdir(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )
        ok, reason = la.verify_deliverable("fix", "", tmp_path)
        assert ok is True
        assert "passed" in reason


# --------------------------------------------------------------------------- #
# Test-result parsing / false-positive success detection
# --------------------------------------------------------------------------- #

class TestPytestResultInterpretation:
    def test_all_passed_is_success(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")
        ok, _ = la.verify_deliverable("fix", "", tmp_path)
        assert ok is True

    def test_mixed_pass_fail_is_not_success(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_mixed.py").write_text(
            "def test_a():\n    assert 1 == 1\n\ndef test_b():\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        ok, reason = la.verify_deliverable("fix", "", tmp_path)
        assert ok is False
        assert "failed" in reason.lower() or "1 failed" in reason

    def test_irrelevant_passed_text_in_report_does_not_fool_audit_check(self, tmp_path):
        # A JSON audit report whose findings mention the word "passed" (e.g.
        # describing that some other check passed) must still be judged on
        # actual JSON validity + non-empty findings, never on substring
        # matching "passed" anywhere in the file.
        report = tmp_path / "security_report.json"
        report.write_text(json.dumps({
            "findings": [{"title": "auth check passed for admin route", "severity": "informational"}]
        }), encoding="utf-8")
        ok, _ = la.verify_deliverable("audit", str(report), tmp_path)
        assert ok is True  # valid structurally; the point is it isn't a keyword sniff
        # Now prove a structurally-empty findings array is rejected even
        # though the raw text contains "passed".
        report.write_text(json.dumps({"findings": [], "note": "all checks passed"}), encoding="utf-8")
        ok2, reason2 = la.verify_deliverable("audit", str(report), tmp_path)
        assert ok2 is False

    def test_parse_pytest_feedback_flags_failed_status(self):
        raw = (
            "collected 2 items\n"
            "test_a.py::test_a PASSED\n"
            "FAILED test_a.py::test_b - AssertionError: boom\n"
            "===================== 1 failed, 1 passed in 0.10s ====================="
        )
        typed = json.loads(la.parse_pytest_feedback(raw).split("[typed feedback] ", 1)[1].split("\n[raw tail]")[0])
        assert typed["status"] == "failed"
        assert typed["n_failed_named"] == 1

    def test_parse_pytest_feedback_flags_passed_status(self):
        raw = "collected 1 item\ntest_a.py::test_a PASSED\n===================== 1 passed in 0.05s ====================="
        typed = json.loads(la.parse_pytest_feedback(raw).split("[typed feedback] ", 1)[1].split("\n[raw tail]")[0])
        assert typed["status"] == "passed"


class TestFalsePositiveSuccessDetection:
    def test_ctf_arbitrary_nonempty_file_is_rejected(self, tmp_path):
        flag = tmp_path / "flag.txt"
        flag.write_text("this is just some random garbage text, not a flag", encoding="utf-8")
        ok, reason = la.verify_deliverable("ctf", str(flag), tmp_path)
        assert ok is False

    def test_ctf_proper_flag_format_is_accepted(self, tmp_path):
        flag = tmp_path / "flag.txt"
        flag.write_text("flag{abc123}", encoding="utf-8")
        ok, _ = la.verify_deliverable("ctf", str(flag), tmp_path)
        assert ok is True

    def test_ctf_declared_json_format_rejects_bailout_placeholder(self, tmp_path):
        flag = tmp_path / "answer.json"
        flag.write_text(json.dumps({"flag": "not found"}), encoding="utf-8")
        ok, reason = la.verify_deliverable("ctf", str(flag), tmp_path, expected_format="json")
        assert ok is False
        assert "bail-out" in reason or "bailout" in reason

    def test_ctf_declared_kv_format_rejects_bailout_value(self, tmp_path):
        flag = tmp_path / "answer.txt"
        flag.write_text("answer=unknown", encoding="utf-8")
        ok, _ = la.verify_deliverable("ctf", str(flag), tmp_path, expected_format="kv")
        assert ok is False

    def test_ctf_structural_sha256_format_validated(self, tmp_path):
        flag = tmp_path / "answer.txt"
        flag.write_text("a" * 64, encoding="utf-8")
        ok, _ = la.verify_deliverable("ctf", str(flag), tmp_path, expected_format="sha256")
        assert ok is True
        flag.write_text("not-a-hash", encoding="utf-8")
        ok2, _ = la.verify_deliverable("ctf", str(flag), tmp_path, expected_format="sha256")
        assert ok2 is False

    def test_bailout_placeholder_detector_catches_repeated_char(self):
        assert la._looks_like_bailout_placeholder("xxxxxxxxxx") is True
        assert la._looks_like_bailout_placeholder("unable to determine") is True
        assert la._looks_like_bailout_placeholder("a real specific answer") is False


class TestForensicsDeliverableFormat:
    def test_valid_four_line_kv_report_accepted(self, tmp_path):
        report = tmp_path / "incident_report.txt"
        report.write_text(
            "attacker_ip=203.0.113.50\n"
            "compromised_user=svc\n"
            "exfil_bytes=1024\n"
            "first_malicious_event_utc=2026-05-01T00:00:00.000Z\n",
            encoding="utf-8",
        )
        ok, _ = la.verify_deliverable("forensics", str(report), tmp_path)
        assert ok is True

    def test_spaces_around_equals_rejected(self, tmp_path):
        report = tmp_path / "incident_report.txt"
        report.write_text("attacker_ip = 203.0.113.50\ncompromised_user=svc\n", encoding="utf-8")
        ok, reason = la.verify_deliverable("forensics", str(report), tmp_path)
        assert ok is False
        assert "format" in reason.lower() or "space" in reason.lower()

    def test_duplicate_keys_rejected(self, tmp_path):
        report = tmp_path / "incident_report.txt"
        report.write_text("k=a\nk=b\n", encoding="utf-8")
        ok, reason = la.verify_deliverable("forensics", str(report), tmp_path)
        assert ok is False
        assert "duplicate" in reason.lower()

    def test_prose_commentary_line_rejected(self, tmp_path):
        report = tmp_path / "incident_report.txt"
        report.write_text("attacker_ip=203.0.113.50\nThis is my analysis of the incident.\n", encoding="utf-8")
        ok, reason = la.verify_deliverable("forensics", str(report), tmp_path)
        assert ok is False


# --------------------------------------------------------------------------- #
# Write/edit success tracking + mechanical-success termination
# --------------------------------------------------------------------------- #

class TestMechanicalSuccessTracking:
    def _state(self, tmp_path):
        return la.AgentState(workdir=tmp_path)

    def test_write_tool_sets_workspace_modified_flag(self, tmp_path):
        state = self._state(tmp_path)
        result = run(la._execute_tool(state, "write_file", {"path": "out.txt", "content": "hi"}))
        assert result.startswith("OK:")
        assert state.workspace_modified is True

    def test_failed_write_does_not_set_workspace_modified(self, tmp_path):
        state = self._state(tmp_path)
        # Writing a protected test file must be refused and must not count
        # as a real workspace modification.
        (tmp_path / "tests").mkdir()
        result = run(la._execute_tool(state, "write_file", {"path": "tests/test_x.py", "content": "x"}))
        assert result.startswith("ERROR")
        assert state.workspace_modified is False

    def test_run_pytest_sets_all_green_on_pass(self, tmp_path):
        state = self._state(tmp_path)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
        run(la._execute_tool(state, "run_pytest", {}))
        assert state.pytest_all_green is True

    def test_run_pytest_unsets_green_after_later_regression(self, tmp_path):
        state = self._state(tmp_path)
        (tmp_path / "tests").mkdir()
        test_file = tmp_path / "tests" / "test_a.py"
        test_file.write_text("def test_a():\n    assert True\n", encoding="utf-8")
        run(la._execute_tool(state, "run_pytest", {}))
        assert state.pytest_all_green is True
        test_file.write_text("def test_a():\n    assert False\n", encoding="utf-8")
        run(la._execute_tool(state, "run_pytest", {}))
        assert state.pytest_all_green is False

    def test_mechanical_success_requires_both_signals(self, tmp_path):
        state = self._state(tmp_path)
        assert la._mechanical_success("fix", state) is False
        state.workspace_modified = True
        assert la._mechanical_success("fix", state) is False  # pytest not confirmed green yet
        state.pytest_all_green = True
        assert la._mechanical_success("fix", state) is True

    def test_mechanical_success_only_applies_to_fix_kind(self, tmp_path):
        state = self._state(tmp_path)
        state.workspace_modified = True
        state.pytest_all_green = True
        assert la._mechanical_success("audit", state) is False
        assert la._mechanical_success("forensics", state) is False


# --------------------------------------------------------------------------- #
# Protection against modifying tests/verifiers/harness files
# --------------------------------------------------------------------------- #

class TestProtectedFiles:
    def test_tests_directory_file_is_protected(self, tmp_path):
        p = tmp_path / "tests" / "test_api.py"
        assert la._is_protected_file(p) is True

    def test_conftest_is_protected(self, tmp_path):
        assert la._is_protected_file(tmp_path / "conftest.py") is True

    def test_agent_harness_files_are_protected(self, tmp_path):
        assert la._is_protected_file(tmp_path / "local_agent.py") is True
        assert la._is_protected_file(tmp_path / "run.sh") is True

    def test_application_file_is_not_protected(self, tmp_path):
        assert la._is_protected_file(tmp_path / "app" / "main.py") is False

    def test_check_editable_blocks_test_file_write(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        err = la._check_editable(tmp_path / "tests" / "test_api.py", state)
        assert err is not None
        assert "FORBIDDEN" in err

    def test_check_editable_allows_application_file(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        err = la._check_editable(tmp_path / "app" / "main.py", state)
        assert err is None

    def test_bash_command_writing_to_tests_is_blocked(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        (tmp_path / "tests").mkdir()
        result = la._bash_touches_protected(f"echo pwned > {tmp_path}/tests/test_api.py", state)
        assert result is not None
        assert "BLOCKED" in result

    def test_bash_command_touching_app_file_is_allowed(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        result = la._bash_touches_protected(f"echo hi > {tmp_path}/app/main.py", state)
        assert result is None

    def test_tampered_test_file_is_restored_from_snapshot(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_api.py"
        original = "def test_api():\n    assert True\n"
        test_file.write_text(original, encoding="utf-8")
        state = la.AgentState(workdir=tmp_path)
        state.tests_snapshot = la._snapshot_tests(state)
        # Simulate tampering that slipped past the tool-level guard (e.g. a
        # raw filesystem write via some other path).
        test_file.write_text("def test_api():\n    assert False  # tampered\n", encoding="utf-8")
        restored = la._restore_tests(state.tests_snapshot)
        assert restored >= 1
        assert test_file.read_text(encoding="utf-8") == original


# --------------------------------------------------------------------------- #
# Bounded retry behavior (transient-service-error classification)
# --------------------------------------------------------------------------- #

class TestErrorClassification:
    def test_transient_5xx_is_retryable(self):
        exc = Exception(
            "ModelHTTPError(\"status_code: 503, model_name: x, "
            "body: {'code': 'SERVICE_UNAVAILABLE', 'message': 'temporarily unavailable'}\")"
        )
        assert la._is_transient_service_error(exc) is True

    def test_balance_exhausted_is_not_retryable_as_transient(self):
        exc = Exception(
            "APIStatusError(\"Error code: 402 - {'error': {'code': 'INSUFFICIENT_BALANCE', "
            "'message': 'limit reached'}}\")"
        )
        assert la._is_balance_exhausted(exc) is True
        assert la._is_transient_service_error(exc) is False

    def test_daily_cap_is_distinct_from_transient_and_balance(self):
        exc = Exception("RateLimitError: requests per day limit exceeded")
        assert la._is_daily_cap(exc) is True
        assert la._is_balance_exhausted(exc) is False

    def test_ordinary_value_error_is_neither(self):
        exc = ValueError("bad json")
        assert la._is_transient_service_error(exc) is False
        assert la._is_balance_exhausted(exc) is False
        assert la._is_daily_cap(exc) is False

    def test_bounded_backoff_returns_none_when_time_nearly_exhausted(self, monkeypatch):
        monkeypatch.setattr(la, "time_left", lambda: 5.0)
        assert la._bounded_backoff(10.0, reserve=60.0) is None

    def test_bounded_backoff_caps_to_available_time(self, monkeypatch):
        monkeypatch.setattr(la, "time_left", lambda: 100.0)
        wait = la._bounded_backoff(500.0, reserve=60.0)
        assert wait is not None
        assert wait <= 40.0 + 1e-6


# --------------------------------------------------------------------------- #
# Repetition guard (loop protection independent of budget)
# --------------------------------------------------------------------------- #

class TestRepetitionGuard:
    def test_identical_repeated_calls_trigger_guard_message(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        last = ""
        for _ in range(6):
            last = la.repetition_guard(state, "read_file", "path=app/main.py", "same output every time")
        assert "loop" in last.lower() or "repeat" in last.lower() or last != ""


# --------------------------------------------------------------------------- #
# json_closer (mechanical truncation repair, 0 LLM requests)
# --------------------------------------------------------------------------- #

class TestJsonCloser:
    def test_closes_truncated_findings_array(self):
        truncated = '{"findings": [{"title": "a", "severity": "high"'
        closed = la.json_closer(truncated)
        data = json.loads(closed)
        assert isinstance(data, dict)

    def test_well_formed_json_is_unchanged_semantically(self):
        good = json.dumps({"findings": [{"title": "a"}]})
        closed = la.json_closer(good)
        assert json.loads(closed) == json.loads(good)
