"""Tests for bayesbench.benchmark and bayesbench.decorators."""

import pytest

from bayesbench import BayesianBenchmark, benchmark, suite
from bayesbench.benchmark import BenchmarkReport, TaskResult
from bayesbench.decision import DecisionStatus
from bayesbench.posteriors import NormalPosterior

# ---------------------------------------------------------------------------
# Deterministic mock models
# ---------------------------------------------------------------------------

PROBLEMS = [{"q": str(i), "a": str(i)} for i in range(100)]


def perfect_model(problem):
    """Always answers correctly."""
    return problem["a"]


def random_model(problem):
    """Always answers incorrectly."""
    return "WRONG"


def score(problem, response):
    return response == problem["a"]


# ---------------------------------------------------------------------------
# BayesianBenchmark.compare
# ---------------------------------------------------------------------------


class TestBayesianBenchmarkCompare:
    def test_perfect_vs_wrong_declares_winner_early(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(
            model_a=perfect_model,
            model_b=random_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="test",
        )
        assert result.winner == "model_a"
        assert result.problems_tested < len(PROBLEMS), "Should stop before exhausting dataset"

    def test_equal_models_may_be_non_discriminating(self):
        bench = BayesianBenchmark(confidence=0.95, skip_threshold=0.85, min_samples=3)
        result = bench.compare(
            model_a=perfect_model,
            model_b=perfect_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="test",
        )
        assert result.skipped or result.winner is None

    def test_skip_threshold_one_disables_skipping(self):
        bench = BayesianBenchmark(confidence=0.95, skip_threshold=1.0, min_samples=3)
        result = bench.compare(
            model_a=perfect_model,
            model_b=perfect_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="no_skip",
        )

        assert not result.skipped
        assert result.problems_tested == len(PROBLEMS)

    def test_result_type(self):
        bench = BayesianBenchmark()
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert isinstance(result, TaskResult)

    def test_efficiency_gt_zero_when_stopped_early(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.efficiency > 0.0

    def test_posterior_a_mean_gt_b_when_a_wins(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.posterior_a.mean > result.posterior_b.mean

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            BayesianBenchmark(confidence=0.3)

    def test_invalid_skip_threshold_raises(self):
        with pytest.raises(ValueError):
            BayesianBenchmark(skip_threshold=0.1)

    def test_winner_uses_configured_confidence(self):
        bench = BayesianBenchmark(confidence=0.9999, skip_threshold=0.999, min_samples=3)
        result = bench.compare(
            model_a=perfect_model,
            model_b=random_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="custom_confidence",
        )

        assert result.confidence == 0.9999
        assert result.p_a_beats_b < result.confidence
        assert result.winner is None
        assert result.to_dict()["confidence"] == 0.9999


# ---------------------------------------------------------------------------
# Decision statuses, safe defaults, paired mode, max_samples, CS rule
# ---------------------------------------------------------------------------


class TestDecisionSemantics:
    def test_decision_status_winner(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.WINNER_A
        assert result.winner == "model_a"

    def test_decision_status_inconclusive_on_exhaustion(self):
        bench = BayesianBenchmark(confidence=0.9999, min_samples=3)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS[:20])
        assert result.decision is DecisionStatus.INCONCLUSIVE
        assert result.problems_tested == 20

    def test_legacy_skip_reports_equivalent(self):
        bench = BayesianBenchmark(confidence=0.95, skip_threshold=0.85, min_samples=3)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.EQUIVALENT
        assert result.skipped is True  # deprecated alias
        assert result.winner is None

    def test_rope_equivalence_via_benchmark(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=10, equivalence_margin=0.05)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.EQUIVALENT

    def test_rope_not_declared_for_real_gap(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=10, equivalence_margin=0.01)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.WINNER_A

    def test_max_samples_caps_run(self):
        bench = BayesianBenchmark(confidence=0.9999, min_samples=3, max_samples=7)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS)
        assert result.problems_tested == 7
        assert result.decision is DecisionStatus.INCONCLUSIVE

    def test_to_dict_includes_decision(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        d = result.to_dict()
        assert d["decision"] == "winner_a"
        assert d["terminal_reason"] == ""
        assert d["paired"] is False


class TestTraces:
    def test_trace_length_matches_problems_tested(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert len(result.trace) == result.problems_tested
        assert result.problems_tested < len(PROBLEMS)

    def test_trace_records_scores_and_evidence(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        first = result.trace[0]
        assert first.step == 1
        assert first.score_a is True and first.score_b is False
        last = result.trace[-1]
        assert last.is_terminal is True
        assert last.status is DecisionStatus.WINNER_A

    def test_trace_terminates_with_decision_status(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        terminal = [t for t in result.trace if t.is_terminal]
        assert len(terminal) == 1
        assert terminal[0].step == result.problems_tested

    def test_on_step_callback_fires_per_problem(self):
        seen = []
        bench = BayesianBenchmark(
            confidence=0.95, min_samples=3, on_step=lambda t: seen.append(t.step)
        )
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert seen == list(range(1, result.problems_tested + 1))

    def test_iter_compare_streams_updates(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        updates = list(bench.iter_compare(perfect_model, random_model, score, PROBLEMS))
        assert updates[-1].result is not None
        assert updates[-1].result.decision is DecisionStatus.WINNER_A
        assert all(u.result is None for u in updates[:-1])
        assert len(updates) == updates[-1].result.problems_tested

    def test_iter_compare_lazy_invocations(self):
        calls = {"a": 0, "b": 0}

        def model_a(p):
            calls["a"] += 1
            return p["a"]

        def model_b(p):
            calls["b"] += 1
            return "WRONG"

        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        updates = list(bench.iter_compare(model_a, model_b, score, PROBLEMS))
        result = updates[-1].result
        assert result is not None
        assert calls["a"] == result.problems_tested
        assert calls["b"] == result.problems_tested

    def test_trace_present_for_exhausted_runs(self):
        bench = BayesianBenchmark(confidence=0.9999, min_samples=3)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS[:15])
        assert len(result.trace) == 15
        assert result.trace[-1].status is DecisionStatus.INCONCLUSIVE


class TestConfidenceSequenceBenchmark:
    def test_cs_rule_end_to_end(self):
        bench = BayesianBenchmark(decision_rule="confidence_sequence", alpha=0.05)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.WINNER_A
        assert result.winner == "model_a"
        assert result.problems_tested < len(PROBLEMS)

    def test_cs_rule_rejects_continuous(self):
        from bayesbench.posteriors import NormalPosterior

        bench = BayesianBenchmark(
            decision_rule="confidence_sequence", posterior_factory=NormalPosterior
        )

        def score_float(problem, response):
            return 0.5

        with pytest.raises(TypeError):
            bench.compare(perfect_model, random_model, score_float, PROBLEMS[:5])

    def test_invalid_decision_rule_raises(self):
        with pytest.raises(ValueError):
            BayesianBenchmark(decision_rule="nope")

    def test_cs_rule_exhausted_dataset_reports_inconclusive(self):
        """Equal models never separate under the CS rule; the exhausted path
        must build a result without touching missing posteriors."""
        bench = BayesianBenchmark(decision_rule="confidence_sequence", alpha=0.05)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS[:20])
        assert result.decision is DecisionStatus.INCONCLUSIVE
        assert result.problems_tested == 20
        assert result.posterior_a.n == 20.0
        assert 0.0 <= result.p_a_beats_b <= 1.0


class TestPairedBenchmark:
    def test_paired_binary(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3, paired=True)
        result = bench.compare(perfect_model, random_model, score, PROBLEMS)
        assert result.paired is True
        assert result.decision is DecisionStatus.WINNER_A
        assert result.problems_tested < len(PROBLEMS)

    def test_paired_continuous(self):
        def score_diff(problem, response):
            return 0.9 if response == problem["a"] else 0.1

        bench = BayesianBenchmark(
            confidence=0.95,
            min_samples=5,
            paired=True,
            posterior_factory=NormalPosterior,
        )
        result = bench.compare(perfect_model, random_model, score_diff, PROBLEMS)
        assert result.paired is True
        assert result.decision is DecisionStatus.WINNER_A

    def test_paired_ties_stay_inconclusive(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3, paired=True, max_samples=50)
        result = bench.compare(perfect_model, perfect_model, score, PROBLEMS)
        assert result.decision is DecisionStatus.INCONCLUSIVE


# ---------------------------------------------------------------------------
# BayesianBenchmark.task decorator + run()
# ---------------------------------------------------------------------------


class TestBayesianBenchmarkTask:
    def test_task_decorator_registers_task(self):
        bench = BayesianBenchmark()

        @bench.task(dataset=PROBLEMS)
        def my_task(problem):
            return True, False

        assert len(bench._tasks) == 1
        assert bench._tasks[0]["name"] == "my_task"

    def test_run_returns_report(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS)
        def compare_task(problem):
            a = perfect_model(problem) == problem["a"]
            b = random_model(problem) == problem["a"]
            return a, b

        report = bench.run()
        assert isinstance(report, BenchmarkReport)
        assert len(report.task_results) == 1

    def test_multiple_tasks(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS, name="task1")
        def t1(problem):
            return True, False

        @bench.task(dataset=PROBLEMS, name="task2")
        def t2(problem):
            return True, False

        report = bench.run()
        assert len(report.task_results) == 2

    def test_task_no_dataset_raises(self):
        bench = BayesianBenchmark()

        @bench.task()
        def no_data(problem):
            return True, True

        with pytest.raises(ValueError, match="no dataset"):
            bench.run()

    def test_overall_efficiency(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS)
        def easy(problem):
            return True, False

        report = bench.run()
        assert report.overall_efficiency > 0.0


# ---------------------------------------------------------------------------
# Async compare
# ---------------------------------------------------------------------------


class TestAsyncCompare:
    @pytest.mark.asyncio
    async def test_async_compare_sync_callables(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = await bench.compare_async(
            model_a=perfect_model,
            model_b=random_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="async_test",
        )
        assert result.winner == "model_a"

    @pytest.mark.asyncio
    async def test_async_compare_async_callables(self):
        async def async_perfect(problem):
            return problem["a"]

        async def async_wrong(problem):
            return "WRONG"

        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        result = await bench.compare_async(
            model_a=async_perfect,
            model_b=async_wrong,
            score_fn=score,
            dataset=PROBLEMS,
            name="async_models",
        )
        assert result.winner == "model_a"


# ---------------------------------------------------------------------------
# @benchmark decorator
# ---------------------------------------------------------------------------


class TestBenchmarkDecorator:
    def test_run_method_added(self):
        @benchmark(
            model_a=perfect_model,
            model_b=random_model,
            dataset=PROBLEMS,
        )
        def exact(problem, response):
            return response == problem["a"]

        assert callable(exact.run)

    def test_run_returns_task_result(self):
        @benchmark(
            model_a=perfect_model,
            model_b=random_model,
            dataset=PROBLEMS,
            confidence=0.95,
            min_samples=3,
        )
        def exact(problem, response):
            return response == problem["a"]

        result = exact.run()
        assert isinstance(result, TaskResult)
        assert result.winner == "model_a"

    def test_original_function_still_works(self):
        @benchmark(
            model_a=perfect_model,
            model_b=random_model,
            dataset=PROBLEMS,
        )
        def exact(problem, response):
            return response == problem["a"]

        assert exact({"a": "hello"}, "hello") is True
        assert exact({"a": "hello"}, "bye") is False

    def test_custom_name(self):
        @benchmark(
            model_a=perfect_model,
            model_b=random_model,
            dataset=PROBLEMS,
            name="custom_name",
        )
        def exact(problem, response):
            return response == problem["a"]

        result = exact.run()
        assert result.name == "custom_name"


# ---------------------------------------------------------------------------
# @suite decorator
# ---------------------------------------------------------------------------


class TestSuiteDecorator:
    def test_suite_run_returns_report(self):
        @suite(confidence=0.95, min_samples=3)
        class MyBench:
            dataset = PROBLEMS

            @staticmethod
            def task_easy(problem):
                return True, False

        report = MyBench.run()
        assert isinstance(report, BenchmarkReport)
        assert len(report.task_results) == 1
        assert report.task_results[0].name == "easy"

    def test_suite_multiple_tasks(self):
        @suite(confidence=0.95, min_samples=3)
        class Multi:
            dataset = PROBLEMS

            @staticmethod
            def task_first(problem):
                return True, False

            @staticmethod
            def task_second(problem):
                return True, False

        report = Multi.run()
        assert len(report.task_results) == 2

    def test_suite_per_task_dataset(self):
        alt_problems = PROBLEMS[:20]

        @suite(confidence=0.95, min_samples=3)
        class PerTask:
            dataset = PROBLEMS
            dataset_small = alt_problems

            @staticmethod
            def task_small(problem):
                return True, False

        report = PerTask.run()
        assert report.task_results[0].total_problems == len(alt_problems)

    def test_non_task_methods_ignored(self):
        @suite()
        class Clean:
            dataset = PROBLEMS

            @staticmethod
            def helper():
                pass

            @staticmethod
            def task_only(problem):
                return True, True

        report = Clean.run()
        assert len(report.task_results) == 1


# ---------------------------------------------------------------------------
# BenchmarkReport helpers
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_summary_string(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS)
        def t(problem):
            return True, False

        report = bench.run()
        summary = report.summary()
        assert "Bayesian Benchmark Report" in summary
        assert "cost reduction" in summary

    def test_winners_dict(self):
        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS, name="win_task")
        def t(problem):
            return True, False

        report = bench.run()
        assert "win_task" in report.winners
