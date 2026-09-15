"""Offline, zero-inference adversarial/synthetic hardening suite.

Follow-up to tests/test_local_agent.py, added during a pre-release hardening
pass triggered by a 5/6 Flash run whose one failure (an incompletely-patched
SQL injection variant) raised a general architecture question: does the
repair loop feed useful, specific failure evidence back to the model, or
does it risk declaring success prematurely / exhausting itself blindly?

Every "model" in this file is a deterministic Python stub, not a live
endpoint - zero network calls, zero cost. Where a test needs to simulate the
LLM loop (run_pyai_loop), it monkeypatches it with a scripted stand-in that
asserts on the PROMPT it was given (proving what evidence actually reached
it) and performs a scripted, deterministic action (proving what the
controller does with the result) - this tests the CONTROLLER's mechanism,
never model intelligence.

No public task name, filename, or verifier string is referenced anywhere
below; all fixtures are synthetic and disposable.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import local_agent as la  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 3. Repair loop feeds bounded, specific failure evidence (not vague, not
#    premature success, not silent exhaustion)
# --------------------------------------------------------------------------- #

class TestRepairFeedbackLoop:
    def test_repair_prompt_contains_the_specific_failing_test_name(self, tmp_path, monkeypatch):
        """A fix-kind task where the baseline leaves ONE named test failing:
        the repair instruction handed to the model loop must name that exact
        test, not a generic 'something is broken' message."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_thing.py").write_text(
            "def test_alpha():\n    assert True\n\n"
            "def test_named_regression_case():\n    assert 1 == 2\n",
            encoding="utf-8",
        )
        state = la.AgentState(workdir=tmp_path)
        seen_instruction = {}

        async def fake_run_pyai_loop(instruction, kind, state_, request_budget, baseline, compactor_kwargs=None, deliverable=""):
            seen_instruction["text"] = instruction
            return "done"

        monkeypatch.setattr(la, "run_pyai_loop", fake_run_pyai_loop)
        run(la.ensure_deliverable("Fix the bug. Run pytest tests/.", "fix", state, "", budget_left=5))
        assert "test_named_regression_case" in seen_instruction["text"]

    def test_repair_that_actually_fixes_the_remaining_failure_is_accepted(self, tmp_path, monkeypatch):
        """Simulates a capable model: reads the fed-back failure reason and
        performs the exact edit needed. ensure_deliverable must re-verify
        and report success - not declare success before checking, not
        stay stuck reporting failure once the fix is genuinely in place."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        target = tmp_path / "app.py"
        target.write_text("def is_safe(x):\n    return True  # bug: always true\n", encoding="utf-8")
        (tests_dir / "test_thing.py").write_text(
            "import sys; sys.path.insert(0, '.')\n"
            "from app import is_safe\n"
            "def test_rejects_bad_input():\n    assert is_safe('bad') is False\n",
            encoding="utf-8",
        )
        state = la.AgentState(workdir=tmp_path)

        async def fake_run_pyai_loop(instruction, kind, state_, request_budget, baseline, compactor_kwargs=None, deliverable=""):
            assert "test_rejects_bad_input" in instruction
            target.write_text("def is_safe(x):\n    return x != 'bad'\n", encoding="utf-8")
            return "fixed"

        monkeypatch.setattr(la, "run_pyai_loop", fake_run_pyai_loop)
        ok = run(la.ensure_deliverable("Fix the bug. Run pytest tests/.", "fix", state, "", budget_left=5))
        assert ok is True

    def test_repair_that_declares_done_without_fixing_is_not_accepted(self, tmp_path, monkeypatch):
        """A model that replies with a confident-sounding final answer but
        never actually touches the failing behavior must NOT be trusted -
        ensure_deliverable's return value comes only from re-running the
        real verification, never from the model's own claim."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_thing.py").write_text(
            "def test_still_broken():\n    assert 1 == 2\n", encoding="utf-8"
        )
        state = la.AgentState(workdir=tmp_path)

        async def fake_run_pyai_loop(instruction, kind, state_, request_budget, baseline, compactor_kwargs=None, deliverable=""):
            return "I have fixed the issue and everything passes now."

        monkeypatch.setattr(la, "run_pyai_loop", fake_run_pyai_loop)
        ok = run(la.ensure_deliverable("Fix the bug. Run pytest tests/.", "fix", state, "", budget_left=5))
        assert ok is False

    def test_repair_regression_on_file_deliverable_is_reverted(self, tmp_path, monkeypatch):
        """audit/forensics/ctf kinds snapshot the deliverable file's health
        before repair; if the repair attempt makes it WORSE, the pre-repair
        (healthier) version must be restored rather than kept."""
        report = tmp_path / "security_report.json"
        good = json.dumps({"findings": [{"title": "a real finding", "severity": "high"}]})
        report.write_text(good, encoding="utf-8")
        state = la.AgentState(workdir=tmp_path)

        # Force a "repair" to trigger: patch verify_deliverable so the first
        # check (before repair) reports NOT ok despite being the healthier
        # version, forcing ensure_deliverable into its repair branch; the
        # repair stub then overwrites the file with something worse.
        calls = {"n": 0}
        real_verify = la.verify_deliverable

        def flaky_verify(kind, deliverable, workdir, expected_format=""):
            calls["n"] += 1
            if calls["n"] == 1:
                return False, "forced repair trigger"
            return real_verify(kind, deliverable, workdir, expected_format)

        async def worsening_repair(instruction, kind, state_, request_budget, baseline, compactor_kwargs=None, deliverable=""):
            Path(deliverable).write_text(json.dumps({"findings": []}), encoding="utf-8")
            return "done"

        monkeypatch.setattr(la, "verify_deliverable", flaky_verify)
        monkeypatch.setattr(la, "run_pyai_loop", worsening_repair)
        run(la.ensure_deliverable("Audit the app.", "audit", state, str(report), budget_left=5))
        # The worse version must have been reverted back to the original.
        assert json.loads(report.read_text())["findings"], "expected revert to the healthier pre-repair content"

    def test_repair_not_attempted_when_budget_left_is_zero(self, tmp_path, monkeypatch):
        """No budget for repair: must not attempt one (and must not crash),
        just report the current (failing) verification state."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_thing.py").write_text("def test_x():\n    assert 1 == 2\n", encoding="utf-8")
        state = la.AgentState(workdir=tmp_path)
        called = {"n": 0}

        async def should_not_run(*a, **kw):
            called["n"] += 1
            return "done"

        monkeypatch.setattr(la, "run_pyai_loop", should_not_run)
        ok = run(la.ensure_deliverable("Fix the bug.", "fix", state, "", budget_left=0))
        assert ok is False
        assert called["n"] == 0


# --------------------------------------------------------------------------- #
# 5. Contract / routing adversarial matrix (paraphrased, action-first)
# --------------------------------------------------------------------------- #

class TestRoutingMatrix:
    def test_a_find_vulnerability_no_modification(self):
        c = la.extract_task_contract_mechanical(
            "Please look over this API for weaknesses and write up what you find "
            "in a report. Leave the implementation untouched."
        )
        assert c.family == "audit"
        assert c.modifies_files is False

    def test_b_fix_vulnerability_modification_required(self):
        c = la.extract_task_contract_mechanical(
            "There's a security hole in this service somewhere - track it down and "
            "patch it so the existing checks in tests/ stay green."
        )
        assert c.family == "fix"
        assert c.modifies_files is True

    def test_c_security_wording_but_report_only(self):
        c = la.extract_task_contract_mechanical(
            "We suspect there may be an injection issue in this codebase. Write a "
            "bug-bounty-style writeup of what you find. Do not modify the code."
        )
        assert c.family == "audit"
        assert c.modifies_files is False

    def test_c2_known_edge_case_third_party_remediation_phrasing_not_recognized(self):
        # KNOWN, DOCUMENTED LIMITATION (not fixed - no trace evidence backs
        # a fix, only this synthetic phrasing): "remediation will be handled
        # by another team" contains the bare word "remediat" with no local
        # negation cue the current vocabulary recognizes, so it currently
        # routes to "fix" even though the sentence means "not your job".
        # Left as-is per the mission's "don't patch imaginable problems"
        # rule; recorded here so it isn't silently reintroduced as an
        # assumption elsewhere.
        c = la.extract_task_contract_mechanical(
            "We suspect there may be an injection issue in this codebase. Write a "
            "bug-bounty-style writeup of what you find; remediation will be handled "
            "by another team."
        )
        assert c.family == "fix"  # documents current (imperfect) behavior

    def test_d_forensics_explicit_deliverable(self):
        c = la.extract_task_contract_mechanical(
            "Reconstruct what happened from the logs in /app/incident/ and record "
            "your attributed conclusion as key=value lines in `/app/incident_report.txt`."
        )
        assert c.family == "forensics"
        assert c.deliverable_path == "/app/incident_report.txt"

    def test_e_ctf_explicit_output_path(self):
        c = la.extract_task_contract_mechanical(
            "Solve the challenge and write the flag to `/app/solution/answer.txt`."
        )
        # No literal "flag{"/"ctf" keyword here and no fix/audit signal - this
        # is intentionally ambiguous at the mechanical layer (would fall to
        # the LLM contract pass in the real pipeline). What must hold
        # regardless of family is the explicit path.
        assert la.extract_explicit_deliverable_path(
            "Solve the challenge and write the flag to `/app/solution/answer.txt`."
        ) == "/app/solution/answer.txt"

    def test_f_generic_repository_modification(self):
        c = la.extract_task_contract_mechanical(
            "Refactor the pagination helper so it stops duplicating the last row "
            "on the final page."
        )
        assert c.family in ("generic", "fix")

    def test_g_explicit_path_overrides_kind_default(self, tmp_path):
        result = la.guess_deliverable(
            "Do a review of the service and put your findings at `/app/out/custom.json`.",
            "audit", tmp_path,
        )
        assert result == "/app/out/custom.json"

    def test_h_no_explicit_path_uses_kind_default(self, tmp_path):
        result = la.guess_deliverable("Do a review of the service.", "audit", tmp_path)
        assert result == str(tmp_path / "security_report.json")

    def test_i_nested_workdir_resolves_relative_paths_correctly(self, tmp_path):
        nested = tmp_path / "workspace" / "project"
        nested.mkdir(parents=True)
        state = la.AgentState(workdir=nested)
        resolved = la._resolve_path("src/app.py", state)
        assert resolved == (nested / "src" / "app.py").resolve()

    def test_j_misleading_vulnerability_keywords_with_explicit_report_only_action(self):
        # Contains "SQL injection", "exploit", "attack" but the actual
        # requested action is unambiguously read-only reporting.
        c = la.extract_task_contract_mechanical(
            "This legacy service has a documented history of SQL injection and "
            "exploit attempts in its changelog. Summarize that history in a report; "
            "do not modify the code."
        )
        assert c.modifies_files is False


# --------------------------------------------------------------------------- #
# 6. Success-detection adversarial matrix (extra cases beyond test_local_agent.py)
# --------------------------------------------------------------------------- #

class TestSuccessDetectionMatrix:
    def test_partial_test_execution_is_not_success(self, tmp_path):
        # pytest output that stopped partway (collection error) must not be
        # misread as "N passed".
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_broken_import.py").write_text(
            "import this_module_does_not_exist\n\ndef test_x():\n    assert True\n",
            encoding="utf-8",
        )
        ok, reason = la.verify_deliverable("fix", "", tmp_path)
        assert ok is False

    def test_write_success_without_validation_is_not_conflated_with_mechanical_success(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        run(la._execute_tool(state, "write_file", {"path": "notes.txt", "content": "wip"}))
        assert state.workspace_modified is True
        assert state.pytest_all_green is False
        assert la._mechanical_success("fix", state) is False

    def test_stale_green_is_invalidated_by_later_source_modification(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_a.py"
        test_file.write_text("def test_a():\n    assert True\n", encoding="utf-8")
        run(la._execute_tool(state, "run_pytest", {}))
        assert state.pytest_all_green is True
        # Model keeps poking at the app after a green run - a write alone
        # does not re-confirm green; only a subsequent run_pytest call
        # updates the flag, and _mechanical_success needs the LATEST run to
        # still say green, not a stale one from before further edits.
        run(la._execute_tool(state, "write_file", {"path": "app.py", "content": "raise RuntimeError()"}))
        test_file.write_text("def test_a():\n    import app\n    assert False\n", encoding="utf-8")
        run(la._execute_tool(state, "run_pytest", {}))
        assert state.pytest_all_green is False
        assert la._mechanical_success("fix", state) is False

    def test_successful_irrelevant_command_does_not_look_like_deliverable_success(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        result = run(la._execute_tool(state, "bash", {"command": "echo 'all checks passed'"}))
        assert "OK" in result or result  # bash tool ran fine
        # An irrelevant successful echo must not itself flip any mechanical
        # success flag - only real write/pytest tool outcomes do.
        assert state.workspace_modified is False
        assert state.pytest_all_green is False

    def test_ctf_kv_format_all_bailout_placeholders_rejected_even_if_syntactically_fine(self, tmp_path):
        f = tmp_path / "answer.txt"
        f.write_text("result=n/a", encoding="utf-8")
        ok, _ = la.verify_deliverable("ctf", str(f), tmp_path, expected_format="kv")
        assert ok is False


# --------------------------------------------------------------------------- #
# 7. Test-runner generality audit (no production change unless evidenced)
# --------------------------------------------------------------------------- #

class TestRunnerGenerality:
    def test_bash_tool_can_run_arbitrary_non_pytest_commands(self, tmp_path):
        """The agent is not hard-locked to pytest: tool_bash executes any
        shell command, so a JS/Go/shell-script repo's own test command
        (npm test, go test, ./test.sh) is reachable through the SAME tool a
        model would already reach for. run_pytest is an additional
        convenience tool, not the only path to running tests."""
        state = la.AgentState(workdir=tmp_path)
        (tmp_path / "check.sh").write_text("#!/bin/sh\necho custom-check-ran\nexit 0\n", encoding="utf-8")
        result = run(la.tool_bash(state, "sh check.sh"))
        assert "custom-check-ran" in result

    def test_system_prompt_for_fix_kind_does_not_force_pytest_as_only_option(self):
        prompt = la.build_system_prompt("fix", Path("/app"))
        # Not asserting a specific sentence (would be brittle); asserting the
        # prompt at minimum mentions verifying/testing in general terms
        # rather than being silent about verification altogether.
        assert "test" in prompt.lower() or "verify" in prompt.lower()


# --------------------------------------------------------------------------- #
# 8. Forensics / audit / CTF generalization
# --------------------------------------------------------------------------- #

class TestGeneralizationByFamily:
    def test_audit_family_never_flags_modifies_files_from_report_language_alone(self):
        c = la.extract_task_contract_mechanical(
            "Identify and document every insecure endpoint. Findings only - "
            "no code changes."
        )
        assert c.modifies_files is False

    def test_audit_structural_check_does_not_claim_semantic_truth(self, tmp_path):
        # A structurally valid, but obviously content-free / low-effort
        # report must still pass the STRUCTURAL check (that's all
        # verify_deliverable ever claims to prove) - semantic correctness is
        # the verifier's job, not this mechanical check's.
        report = tmp_path / "security_report.json"
        report.write_text(json.dumps({"findings": [{"title": "x"}]}), encoding="utf-8")
        ok, reason = la.verify_deliverable("audit", str(report), tmp_path)
        assert ok is True
        assert "ok" == reason  # no semantic claim embedded in the reason string

    def test_forensics_missing_artifact_is_not_fabricated_into_success(self, tmp_path):
        # No report written at all: must fail, never invent a plausible-
        # looking pass.
        ok, reason = la.verify_deliverable("forensics", str(tmp_path / "incident_report.txt"), tmp_path)
        assert ok is False
        assert "missing" in reason.lower()

    def test_ctf_no_flag_convention_required_when_task_declares_other_format(self, tmp_path):
        f = tmp_path / "answer.txt"
        f.write_text("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", encoding="utf-8")
        ok, _ = la.verify_deliverable("ctf", str(f), tmp_path, expected_format="sha256")
        assert ok is True


# --------------------------------------------------------------------------- #
# 9. Synthetic mini-benchmark: controller mechanisms end-to-end, mocked model
# --------------------------------------------------------------------------- #

class TestSyntheticMiniBenchmark:
    def test_simple_deterministic_file_task_fast_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(la, "_resolve_workdir", lambda: tmp_path)
        result = la.detect_trivial("Create a file at hello.txt whose entire content is exactly hithere")
        assert result is not None
        path, content = result
        assert content == "hithere"

    def test_incomplete_first_repair_then_second_evidence_cycle_is_faithful(self, tmp_path, monkeypatch):
        """Simulates the EXACT class of failure seen in the real 5/6 run: an
        initial patch closes most but not all of a vulnerability class. The
        stub model on its FIRST repair call only partially fixes it; a
        second ensure_deliverable pass (as would happen if the outer loop
        had more budget) with a fresh, correctly-updated failure reason
        finishes the job. Proves the reason string reflects the CURRENT
        state each time, not a stale one."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        target = tmp_path / "app.py"
        target.write_text("def check(x):\n    return True\n", encoding="utf-8")
        (tests_dir / "test_thing.py").write_text(
            "import sys; sys.path.insert(0, '.')\n"
            "from app import check\n"
            "def test_rejects_a():\n    assert check('a') is False\n"
            "def test_rejects_b():\n    assert check('b') is False\n",
            encoding="utf-8",
        )
        state = la.AgentState(workdir=tmp_path)

        call_log = []

        async def partial_then_full_fix(instruction, kind, state_, request_budget, baseline, compactor_kwargs=None, deliverable=""):
            call_log.append(instruction)
            if len(call_log) == 1:
                target.write_text("def check(x):\n    return x != 'a'\n", encoding="utf-8")  # fixes only 'a'
            else:
                target.write_text("def check(x):\n    return x not in ('a', 'b')\n", encoding="utf-8")  # fixes both
            return "done"

        monkeypatch.setattr(la, "run_pyai_loop", partial_then_full_fix)
        ok1 = run(la.ensure_deliverable("Fix the bug.", "fix", state, "", budget_left=5))
        assert ok1 is False  # test_rejects_b still fails after the first (partial) repair
        assert "test_rejects_b" in call_log[0]

        ok2 = run(la.ensure_deliverable("Fix the bug.", "fix", state, "", budget_left=5))
        assert ok2 is True
        assert "test_rejects_b" in call_log[1]  # second call's reason still names the real remaining gap

    def test_non_python_validation_via_bash_tool(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        (tmp_path / "run_checks.sh").write_text(
            "#!/bin/sh\n[ -f marker.txt ] && echo PASS || echo FAIL\n", encoding="utf-8"
        )
        result = run(la.tool_bash(state, "sh run_checks.sh"))
        assert "FAIL" in result
        run(la.tool_write_file(state, "marker.txt", "present"))
        result2 = run(la.tool_bash(state, "sh run_checks.sh"))
        assert "PASS" in result2

    def test_misleading_decoy_files_do_not_prevent_reading_the_real_target(self, tmp_path):
        state = la.AgentState(workdir=tmp_path)
        (tmp_path / "readme_security.txt").write_text("this file is a decoy, ignore", encoding="utf-8")
        (tmp_path / "real_target.py").write_text("SECRET = 'the real content'", encoding="utf-8")
        result = run(la.tool_read_file(state, "real_target.py"))
        assert "the real content" in result

    def test_success_then_further_tool_use_does_not_unset_mechanical_success_flags(self, tmp_path):
        """Once mechanical success is real, a model that (against the nudge)
        keeps poking around with READ-ONLY tools afterward must not have the
        success signal spuriously cleared - only a genuine regression
        (another run_pytest reporting red) should do that."""
        state = la.AgentState(workdir=tmp_path)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
        run(la._execute_tool(state, "write_file", {"path": "app.py", "content": "x = 1"}))
        run(la._execute_tool(state, "run_pytest", {}))
        assert la._mechanical_success("fix", state) is True
        run(la._execute_tool(state, "read_file", {"path": "app.py"}))
        assert la._mechanical_success("fix", state) is True


# --------------------------------------------------------------------------- #
# 10. Anti-overfitting scan: no public task names/answers baked into production code
# --------------------------------------------------------------------------- #

class TestAntiOverfitScan:
    PUBLIC_TASK_TOKENS = [
        "fix-sqli-login", "find-sqli-login", "fix-sqli-search",
        "incident-log-forensics", "hello-file", "bye-file",
        "203.0.113.50", "deploysvc", "2457600",
    ]

    def test_no_public_task_names_or_answers_in_production_source(self):
        src = Path(la.__file__).read_text(encoding="utf-8")
        low = src.lower()
        hits = [tok for tok in self.PUBLIC_TASK_TOKENS if tok.lower() in low]
        assert not hits, f"public-benchmark-specific tokens found in production code: {hits}"

    def test_no_hardcoded_local_task_directory_reference(self):
        src = Path(la.__file__).read_text(encoding="utf-8")
        assert "local_task" not in src
