"""Tests for baysbench.benchmark and baysbench.decorators."""

import pytest

from baysbench import BayesianBenchmark, benchmark, suite
from baysbench.benchmark import BenchmarkReport, TaskResult

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
