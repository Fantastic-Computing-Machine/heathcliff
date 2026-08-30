# ABOUTME: Regression tests for the bounded evidence-backed research workflow
# ABOUTME: Verifies semantic depth, source independence, extraction, and partial reports

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.subagents.info import agent, evidence


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, args, config=None):
        self.calls.append((args, config))
        return self.result


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages, config=None):
        self.messages = messages
        return self.result


class EchoExtractTool(FakeTool):
    def __init__(self):
        super().__init__(None)

    def invoke(self, args, config=None):
        self.calls.append((args, config))
        return {
            "ok": True,
            "results": [
                {"url": url, "raw_content": f"Evidence from {url}"}
                for url in args["urls"]
            ],
        }


def research_plan():
    return evidence.ResearchPlan(
        mode="research",
        depth="brief",
        research_questions=["What changed?", "What evidence supports it?"],
        search_queries=["topic primary evidence", "topic independent analysis"],
        source_types=["primary source", "independent analysis"],
        required_topics=["core evidence"],
        perspectives=["evidence", "limitations"],
        output_constraints=["concise report"],
    )


def deep_plan():
    return evidence.ResearchPlan(
        mode="research",
        depth="deep",
        research_questions=["What is happening?", "What should be done?"],
        search_queries=[f"deep research angle {number}" for number in range(5)],
        source_types=["primary", "independent", "critical"],
        required_topics=[
            "technical feasibility",
            "lifecycle economics",
            "community impacts",
            "policy options",
            "risks",
        ],
        perspectives=["technical", "economic", "social", "risk"],
        report_outline=["Executive summary", "Evidence", "Options", "Recommendation"],
        requires_options=True,
        requires_recommendation=True,
        requires_implementation_plan=True,
        separate_facts_and_assumptions=True,
        requires_counterarguments=True,
    )


def test_depth_tiers_enforce_scaled_planning_requirements():
    with pytest.raises(ValueError, match="deep research requires at least 5"):
        evidence.ResearchPlan(
            mode="research",
            depth="deep",
            search_queries=["one", "two"],
            required_topics=["technical feasibility"],
            perspectives=["risk"],
        )

    assert deep_plan().depth == "deep"
    assert evidence._BUDGETS["deep"].initial_sources == 7
    assert evidence._BUDGETS["deep"].max_sources == 10


def test_structured_planning_retries_one_invalid_response():
    valid = research_plan()
    runnable = Mock()
    runnable.invoke.side_effect = [ValueError("invalid schema"), valid]
    model = Mock()
    model.with_structured_output.return_value = runnable

    result = evidence.plan_request(model, "research request", [])

    assert result == valid
    assert runnable.invoke.call_count == 2


def test_collect_evidence_reads_distinct_non_wikipedia_sources(monkeypatch):
    monkeypatch.setattr(evidence.Config, "TAVILY_API_KEY", "configured")
    search = FakeTool(
        {
            "ok": True,
            "results": [
                {"url": "https://one.example/report", "title": "One"},
                {"url": "https://en.wikipedia.org/wiki/Topic", "title": "Wiki"},
                {"url": "https://two.example/analysis", "title": "Two"},
            ],
        }
    )
    extract = FakeTool(
        {
            "ok": True,
            "results": [
                {"url": "https://one.example/report", "raw_content": "Evidence one"},
                {"url": "https://two.example/analysis", "raw_content": "Evidence two"},
            ],
        }
    )
    monkeypatch.setattr(evidence, "tavily_search", search)
    monkeypatch.setattr(evidence, "tavily_extract", extract)

    sources, limitations = evidence.collect_evidence(research_plan(), [])

    assert {source.url for source in sources} == {
        "https://one.example/report",
        "https://two.example/analysis",
    }
    assert not limitations
    assert len(extract.calls) == 1


def test_deep_collection_reserves_capacity_for_gap_sources(monkeypatch):
    monkeypatch.setattr(evidence.Config, "TAVILY_API_KEY", "configured")
    results = [
        {"url": f"https://source{number}.example/report", "title": f"Source {number}"}
        for number in range(12)
    ]
    monkeypatch.setattr(evidence, "tavily_search", FakeTool({"ok": True, "results": results}))
    extract = EchoExtractTool()
    monkeypatch.setattr(evidence, "tavily_extract", extract)

    initial, _ = evidence.collect_evidence(deep_plan(), [])
    follow_up = deep_plan().model_copy(
        update={"search_queries": [f"gap query {number}" for number in range(4)]}
    )
    added, _ = evidence.collect_evidence(follow_up, [], existing_sources=initial)

    assert len(initial) == 7
    assert len(added) == 3
    assert [len(call[0]["urls"]) for call in extract.calls] == [7, 3]


def test_failed_extraction_returns_transparent_partial_report(monkeypatch):
    monkeypatch.setattr(evidence.Config, "TAVILY_API_KEY", "configured")
    monkeypatch.setattr(
        evidence,
        "tavily_search",
        FakeTool(
            {
                "ok": True,
                "results": [{"url": "https://one.example/report", "title": "One"}],
            }
        ),
    )
    monkeypatch.setattr(
        evidence, "tavily_extract", FakeTool({"ok": False, "error": "blocked"})
    )
    monkeypatch.setattr(evidence, "read_website", FakeTool("Error fetching website"))

    sources, limitations = evidence.collect_evidence(research_plan(), [])
    result = evidence.synthesize_report(
        Mock(), "request", research_plan(), sources, limitations, []
    )

    assert not sources
    assert result.startswith("## Research incomplete")
    assert "requires at least 2 independent sources" in result
    assert "Could not extract source" in result


def test_candidate_selection_represents_complementary_queries():
    selected = evidence._select_candidates(
        [
            {"url": "https://one.example/a", "query": "official"},
            {"url": "https://two.example/b", "query": "official"},
            {"url": "https://three.example/c", "query": "independent"},
            {"url": "https://four.example/d", "query": "limitations"},
        ],
        max_sources=4,
    )

    assert {item["query"] for item in selected} == {
        "official",
        "independent",
        "limitations",
    }


def test_synthesis_preserves_conflicts_limitations_and_source_links():
    source = evidence.EvidenceSource(
        url="https://one.example/report",
        title="One",
        content="Evidence one",
        provider="tavily",
    )
    model = FakeStructuredModel(
        evidence.ResearchReport(
            answer="The evidence supports the finding.",
            conflicts=["Source estimates differ."],
            limitations=["Only preliminary figures are available."],
        )
    )

    result = evidence.synthesize_report(
        model, "request", research_plan(), [source], [], []
    )

    assert "## Conflicting evidence" in result
    assert "## Limitations" in result
    assert "https://one.example/report" in result


def test_deep_report_renders_every_planned_deliverable():
    plan = deep_plan()
    sources = [
        evidence.EvidenceSource(
            url=f"https://source{number}.example/report",
            title=f"Source {number}",
            content="Evidence",
            provider="tavily",
        )
        for number in range(6)
    ]
    report = evidence.ResearchReport(
        title="Fleet electrification decision",
        answer="Executive finding.",
        perspective_findings=[
            evidence.PerspectiveFinding(
                perspective=perspective,
                analysis=f"{perspective} analysis",
                source_urls=[sources[0].url],
            )
            for perspective in plan.perspectives
        ],
        topic_findings=[
            evidence.TopicFinding(
                topic=topic,
                analysis=f"{topic} analysis",
                source_urls=[sources[0].url],
            )
            for topic in plan.required_topics
        ],
        established_facts=["A measured fact."],
        assumptions=["A planning assumption."],
        options=[
            evidence.ResearchOption(
                name="Phased transition",
                advantages=["Lower transition risk"],
                disadvantages=["Slower benefits"],
            )
        ],
        recommendation="Proceed in phases.",
        implementation_steps=["Pilot", "Measure", "Scale"],
        counterarguments=["Upfront cost may outweigh near-term benefits."],
    )

    result = evidence.synthesize_report(
        FakeStructuredModel(report), "request", plan, sources, [], []
    )

    assert result.startswith("# Fleet electrification decision")
    for heading in [
        "## Executive summary",
        "## Analysis by perspective",
        "## Findings by requested topic",
        "## Established facts",
        "## Assumptions",
        "## Options and trade-offs",
        "## Recommendation",
        "## Implementation plan",
        "## Counterarguments",
        "## Sources",
    ]:
        assert heading in result
    assert "## Research incomplete" not in result


def test_deep_workflow_runs_one_gap_filling_round(monkeypatch):
    initial = [
        evidence.EvidenceSource(
            url=f"https://source{number}.example/report",
            content=f"Evidence {number}",
            provider="tavily",
        )
        for number in range(6)
    ]
    supplemental = [
        evidence.EvidenceSource(
            url="https://supplemental.example/report",
            content="Supplemental evidence",
            provider="tavily",
        )
    ]
    collect = Mock(side_effect=[(initial, []), (supplemental, [])])
    assess = Mock(
        side_effect=[
            evidence.EvidenceAssessment(
                covered_perspectives=["technical", "economic"],
                gaps=["Affected communities are not represented"],
                follow_up_queries=["affected communities primary evidence"],
                ready_to_synthesize=False,
            ),
            evidence.EvidenceAssessment(
                covered_perspectives=["technical", "economic", "social", "risk"],
                gaps=[],
                follow_up_queries=[],
                ready_to_synthesize=True,
            ),
        ]
    )
    synthesize = Mock(return_value="Deep report")
    monkeypatch.setattr(evidence, "collect_evidence", collect)
    monkeypatch.setattr(evidence, "assess_evidence", assess)
    monkeypatch.setattr(evidence, "synthesize_report", synthesize)

    result = evidence.run_research_workflow(Mock(), "request", deep_plan(), [])

    assert result == "Deep report"
    assert collect.call_count == 2
    assert assess.call_count == 2
    assert len(synthesize.call_args.args[3]) == 7
    assert synthesize.call_args.args[6].ready_to_synthesize is True


def test_brief_workflow_skips_gap_assessment(monkeypatch):
    sources = [
        evidence.EvidenceSource(
            url=f"https://source{number}.example/report",
            content="Evidence",
            provider="website",
        )
        for number in range(2)
    ]
    monkeypatch.setattr(evidence, "collect_evidence", lambda *args, **kwargs: (sources, []))
    assess = Mock(side_effect=AssertionError("brief research must not gap-assess"))
    monkeypatch.setattr(evidence, "assess_evidence", assess)
    monkeypatch.setattr(evidence, "synthesize_report", lambda *args: "Brief report")

    assert evidence.run_research_workflow(Mock(), "request", research_plan(), []) == "Brief report"
    assess.assert_not_called()


def test_synthesis_failure_returns_collected_evidence(monkeypatch):
    sources = [
        evidence.EvidenceSource(
            url="https://source.example/report",
            title="Collected source",
            content="Collected factual evidence.",
            provider="website",
        )
    ]
    monkeypatch.setattr(evidence, "collect_evidence", lambda *args, **kwargs: (sources, []))
    monkeypatch.setattr(
        evidence,
        "synthesize_report",
        Mock(side_effect=ValueError("invalid report schema")),
    )

    result = evidence.run_research_workflow(Mock(), "request", research_plan(), [])

    assert result.startswith("## Research incomplete")
    assert "Collected factual evidence" in result
    assert "Final structured synthesis failed" in result


def test_direct_plan_keeps_existing_fast_agent_path(monkeypatch):
    direct_agent = Mock()
    direct_agent.invoke.return_value = {
        "messages": [SimpleNamespace(content="Direct answer")]
    }
    monkeypatch.setattr(agent, "_agent", None)
    monkeypatch.setattr(agent, "_agents", {})
    monkeypatch.setattr(agent, "_build", lambda *_: direct_agent)
    monkeypatch.setattr(agent, "_model", lambda _: Mock())
    monkeypatch.setattr(
        agent,
        "plan_request",
        lambda *args: evidence.ResearchPlan(
            mode="direct", direct_tools=["wikipedia_search"]
        ),
    )
    workflow = Mock(side_effect=AssertionError("direct lookup must not research"))
    monkeypatch.setattr(agent, "run_research_workflow", workflow)

    result = agent.info_agent_tool.invoke({"request": "What is the capital of France?"})

    assert result == "Direct answer"
    workflow.assert_not_called()


def test_research_plan_uses_enforced_evidence_path(monkeypatch):
    monkeypatch.setattr(agent, "_agent", None)
    monkeypatch.setattr(agent, "_agents", {})
    monkeypatch.setattr(agent, "_build", lambda *_: Mock())
    monkeypatch.setattr(agent, "_model", lambda _: Mock())
    monkeypatch.setattr(agent, "plan_request", lambda *args: research_plan())
    workflow = Mock(return_value="Evidence-backed report")
    monkeypatch.setattr(agent, "run_research_workflow", workflow)

    result = agent.info_agent_tool.invoke({"request": "Compare the available evidence"})

    assert result == "Evidence-backed report"
    workflow.assert_called_once()
