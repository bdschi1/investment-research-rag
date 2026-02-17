"""Evaluation harness — run scenarios, collect metrics, report results."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from rag.evaluation.answer_grader import AnswerGrader
from rag.evaluation.schemas import EvalResult, EvalScenario

logger = logging.getLogger(__name__)


class EvalRunner:
    """Run evaluation scenarios and aggregate results."""

    def __init__(self, grader: AnswerGrader | None = None):
        self.grader = grader or AnswerGrader()

    def load_scenarios(self, path: str | Path) -> list[EvalScenario]:
        """Load scenarios from a YAML file or directory.

        Supports both single YAML files and directories of YAML files.
        """
        p = Path(path)
        scenarios: list[EvalScenario] = []

        if p.is_file():
            scenarios.extend(self._parse_yaml(p))
        elif p.is_dir():
            for yaml_file in sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml")):
                scenarios.extend(self._parse_yaml(yaml_file))
        else:
            raise FileNotFoundError(f"Scenario path not found: {path}")

        logger.info("Loaded %d evaluation scenarios from %s", len(scenarios), path)
        return scenarios

    @staticmethod
    def _parse_yaml(path: Path) -> list[EvalScenario]:
        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            return []

        raw_scenarios = data if isinstance(data, list) else data.get("scenarios", [data])

        scenarios = []
        for i, item in enumerate(raw_scenarios):
            scenarios.append(EvalScenario(
                id=item.get("id", f"{path.stem}_{i}"),
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                expected_citations=item.get("expected_citations", []),
                ticker=item.get("ticker"),
                doc_type=item.get("doc_type"),
                tags=item.get("tags", []),
            ))

        return scenarios

    def run(
        self,
        scenarios: list[EvalScenario],
        answers: dict[str, str],
        retrieval_counts: dict[str, int] | None = None,
        citation_counts: dict[str, int] | None = None,
    ) -> list[EvalResult]:
        """Grade answers for a list of scenarios.

        Args:
            scenarios: Eval scenarios to grade.
            answers: Map of scenario_id → generated answer.
            retrieval_counts: Map of scenario_id → retrieval count.
            citation_counts: Map of scenario_id → citation count.

        Returns:
            List of ``EvalResult`` for each scenario.
        """
        results: list[EvalResult] = []
        retrieval_counts = retrieval_counts or {}
        citation_counts = citation_counts or {}

        for scenario in scenarios:
            answer = answers.get(scenario.id, "")
            result = self.grader.grade(
                scenario=scenario,
                generated_answer=answer,
                retrieval_count=retrieval_counts.get(scenario.id, 0),
                citation_count=citation_counts.get(scenario.id, 0),
            )
            results.append(result)

        return results

    @staticmethod
    def summary(results: list[EvalResult]) -> dict[str, float]:
        """Compute aggregate metrics across results.

        Returns:
            Dict with average scores for each quality dimension.
        """
        if not results:
            return {}

        n = len(results)
        return {
            "avg_relevance": sum(r.relevance.value for r in results) / n,
            "avg_accuracy": sum(r.accuracy.value for r in results) / n,
            "avg_completeness": sum(r.completeness.value for r in results) / n,
            "avg_citation_quality": sum(r.citation_quality.value for r in results) / n,
            "avg_retrieval_count": sum(r.retrieval_count for r in results) / n,
            "avg_citation_count": sum(r.citation_count for r in results) / n,
            "total_scenarios": n,
        }
