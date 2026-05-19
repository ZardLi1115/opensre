from __future__ import annotations

from click.testing import CliRunner

from app.cli.__main__ import cli
from app.cli.commands.bench import _compare_payloads


def test_bench_list_includes_cloudopsbench_adapter() -> None:
    result = CliRunner().invoke(cli, ["bench", "list"])

    assert result.exit_code == 0
    assert "cloudopsbench" in result.output


def test_bench_validate_accepts_checked_in_cloudopsbench_config() -> None:
    config_path = "tests/benchmarks/configs/claude-vs-paper.yml"

    result = CliRunner().invoke(cli, ["bench", "validate", config_path])

    assert result.exit_code == 0
    assert f"OK: {config_path}" in result.output


def test_compare_payloads_includes_score_metric_deltas() -> None:
    left = {
        "benchmark": "fake",
        "cost_usd": 0.1,
        "results": [
            {
                "mode": "opensre+llm",
                "llm": "gpt-5",
                "score": {"metrics": {"a1": 0.5, "tcr": 0.25}},
            }
        ],
    }
    right = {
        "benchmark": "fake",
        "cost_usd": 0.2,
        "results": [
            {
                "mode": "opensre+llm",
                "llm": "gpt-5",
                "score": {"metrics": {"a1": 0.75, "tcr": 0.5}},
            }
        ],
    }

    comparison = _compare_payloads(left, right)

    assert comparison["left"]["metrics"]["opensre+llm/gpt-5"]["a1"] == 0.5
    assert comparison["right"]["metrics"]["opensre+llm/gpt-5"]["a1"] == 0.75
    assert comparison["delta"]["metrics"]["opensre+llm/gpt-5"]["a1"] == 0.25
    assert comparison["delta"]["metrics"]["opensre+llm/gpt-5"]["tcr"] == 0.25
