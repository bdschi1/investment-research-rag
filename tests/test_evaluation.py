"""Tests for evaluation harness — metrics, grading, runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.evaluation.answer_grader import AnswerGrader
from rag.evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from rag.evaluation.runner import EvalRunner
from rag.evaluation.schemas import (
    EvalResult,
    EvalScenario,
    FinancialRubric,
    LikertScore,
)

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_precision_at_k_perfect(self):
        assert precision_at_k({"a", "b"}, ["a", "b", "c"], k=2) == 1.0

    def test_precision_at_k_half(self):
        assert precision_at_k({"a"}, ["a", "b"], k=2) == 0.5

    def test_precision_at_k_zero(self):
        assert precision_at_k({"a"}, ["b", "c"], k=2) == 0.0

    def test_precision_at_k_empty(self):
        assert precision_at_k(set(), [], k=5) == 0.0

    def test_recall_at_k_perfect(self):
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=3) == 1.0

    def test_recall_at_k_half(self):
        assert recall_at_k({"a", "b"}, ["a", "c"], k=2) == 0.5

    def test_recall_at_k_zero(self):
        assert recall_at_k({"a"}, ["b", "c"], k=2) == 0.0

    def test_recall_at_k_empty_relevant(self):
        assert recall_at_k(set(), ["a"], k=1) == 0.0

    def test_mrr_first_position(self):
        assert mrr({"a"}, ["a", "b", "c"]) == 1.0

    def test_mrr_second_position(self):
        assert mrr({"b"}, ["a", "b", "c"]) == 0.5

    def test_mrr_third_position(self):
        assert mrr({"c"}, ["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_mrr_not_found(self):
        assert mrr({"d"}, ["a", "b", "c"]) == 0.0

    def test_ndcg_perfect(self):
        # Perfect ranking: [3, 2, 1]
        assert ndcg_at_k([3.0, 2.0, 1.0], k=3) == pytest.approx(1.0)

    def test_ndcg_reversed(self):
        # Worst ranking: [1, 2, 3] vs ideal [3, 2, 1]
        score = ndcg_at_k([1.0, 2.0, 3.0], k=3)
        assert 0 < score < 1

    def test_ndcg_empty(self):
        assert ndcg_at_k([], k=5) == 0.0

    def test_ndcg_all_zero(self):
        assert ndcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0


# ---------------------------------------------------------------------------
# Answer grader
# ---------------------------------------------------------------------------


class TestAnswerGrader:
    @pytest.fixture
    def grader(self) -> AnswerGrader:
        return AnswerGrader()

    def test_grade_with_expected(self, grader: AnswerGrader):
        scenario = EvalScenario(
            id="test-1",
            question="What was Apple's revenue?",
            expected_answer="Apple reported revenue of $94.9 billion in Q4 2024.",
        )
        result = grader.grade(
            scenario=scenario,
            generated_answer="Revenue was $94.9 billion, up 6% [1].",
            retrieval_count=5,
            citation_count=1,
        )
        assert isinstance(result, EvalResult)
        assert result.scenario_id == "test-1"
        assert result.relevance.value >= 2
        assert result.citation_quality.value >= 2

    def test_grade_empty_answer(self, grader: AnswerGrader):
        scenario = EvalScenario(id="test-2", question="What?")
        result = grader.grade(scenario, "")
        assert result.relevance == LikertScore.VERY_POOR
        assert result.completeness == LikertScore.VERY_POOR

    def test_grade_long_cited_answer(self, grader: AnswerGrader):
        scenario = EvalScenario(id="test-3", question="What was revenue?")
        answer = (
            "Revenue was $94.9 billion for the fiscal year [1]. Services grew "
            "to a record $25 billion, driven by advertising and App Store [2]. "
            "Gross margin expanded 120 basis points to 46.2% [3]. iPhone demand "
            "was strong with the iPhone 16 launch exceeding expectations. "
            "Management highlighted the strength of the installed base and "
            "wearables showed signs of stabilization after prior declines."
        )
        result = grader.grade(scenario, answer)
        assert result.citation_quality.value >= 4  # 3+ citations
        assert result.completeness.value >= 3  # Adequate length (30+ words)

    def test_grade_without_expected(self, grader: AnswerGrader):
        scenario = EvalScenario(id="test-4", question="Describe the outlook.")
        result = grader.grade(scenario, "The outlook is positive based on [1].")
        # Without expected answer, accuracy defaults to ADEQUATE
        assert result.accuracy == LikertScore.ADEQUATE


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


class TestEvalRunner:
    @pytest.fixture
    def runner(self) -> EvalRunner:
        return EvalRunner()

    def test_load_scenarios_yaml(self, runner: EvalRunner, tmp_path: Path):
        yaml_content = """
scenarios:
  - id: "s1"
    question: "What was revenue?"
    expected_answer: "Revenue was $94.9B"
    ticker: "AAPL"
    tags: ["revenue", "quarterly"]
  - id: "s2"
    question: "What were risk factors?"
"""
        p = tmp_path / "scenarios.yaml"
        p.write_text(yaml_content)

        scenarios = runner.load_scenarios(p)
        assert len(scenarios) == 2
        assert scenarios[0].id == "s1"
        assert scenarios[0].ticker == "AAPL"
        assert "revenue" in scenarios[0].tags

    def test_load_scenarios_directory(self, runner: EvalRunner, tmp_path: Path):
        for i in range(3):
            p = tmp_path / f"scenario_{i}.yaml"
            p.write_text(f'- id: "s{i}"\n  question: "Q{i}?"')

        scenarios = runner.load_scenarios(tmp_path)
        assert len(scenarios) == 3

    def test_run_grading(self, runner: EvalRunner):
        scenarios = [
            EvalScenario(id="s1", question="What was revenue?", expected_answer="$94.9B"),
            EvalScenario(id="s2", question="Risk factors?"),
        ]
        answers = {
            "s1": "Revenue was $94.9 billion [1].",
            "s2": "Key risks include FX and supply chain [1,2].",
        }
        results = runner.run(scenarios, answers)
        assert len(results) == 2

    def test_summary(self, runner: EvalRunner):
        scenarios = [
            EvalScenario(id="s1", question="Q?"),
        ]
        answers = {"s1": "Answer with citations [1][2][3]. More detail here."}
        results = runner.run(scenarios, answers)
        summary = runner.summary(results)
        assert "avg_relevance" in summary
        assert "total_scenarios" in summary
        assert summary["total_scenarios"] == 1

    def test_summary_empty(self):
        assert EvalRunner.summary([]) == {}

    def test_load_missing_path(self, runner: EvalRunner):
        with pytest.raises(FileNotFoundError):
            runner.load_scenarios("/nonexistent/path")


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestEvalSchemas:
    def test_likert_values(self):
        assert LikertScore.VERY_POOR.value == 1
        assert LikertScore.EXCELLENT.value == 5

    def test_financial_rubric_default(self):
        rubric = FinancialRubric()
        assert len(rubric.dimensions) == 5
        assert abs(sum(rubric.weights.values()) - 1.0) < 0.01

    def test_rubric_weighted_score(self):
        rubric = FinancialRubric()
        scores = {
            "factual_accuracy": 0.9,
            "quantitative_precision": 0.8,
            "source_attribution": 0.7,
            "analytical_depth": 0.6,
            "completeness": 0.5,
        }
        ws = rubric.weighted_score(scores)
        assert 0 < ws < 1
        # Manual: 0.9*0.3 + 0.8*0.25 + 0.7*0.2 + 0.6*0.15 + 0.5*0.1 = 0.75
        assert abs(ws - 0.75) < 0.01

    def test_eval_scenario(self):
        s = EvalScenario(
            id="test",
            question="Q?",
            expected_answer="A",
            tags=["tag1"],
        )
        assert s.id == "test"
        assert s.tags == ["tag1"]
