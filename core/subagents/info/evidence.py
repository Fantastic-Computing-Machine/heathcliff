# ABOUTME: Semantic research planning and bounded multi-source evidence collection
# ABOUTME: Enforces source reading and transparent partial reports before synthesis

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import Config
from core.subagents._runner import record_tool_invocation
from core.subagents.info.tools import (
    read_website,
    search_web,
    tavily_extract,
    tavily_search,
)

_URL_RE = re.compile(r"https?://[^\s|)>\]}]+")


@dataclass(frozen=True)
class ResearchBudget:
    """Hard bounds for each semantically selected research depth."""

    min_queries: int
    min_sources: int
    initial_sources: int
    max_sources: int
    source_chars: int
    gap_rounds: int


_BUDGETS = {
    "brief": ResearchBudget(2, 2, 3, 3, 3500, 0),
    "standard": ResearchBudget(3, 3, 5, 5, 4500, 0),
    "deep": ResearchBudget(5, 6, 7, 10, 5000, 1),
}


class ResearchPlan(BaseModel):
    """The model's semantic decision and bounded retrieval plan."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["direct", "research"]
    depth: Literal["brief", "standard", "deep"] = "standard"
    direct_tools: list[
        Literal[
            "get_weather",
            "get_news",
            "search_web",
            "tavily_search",
            "tavily_extract",
            "wikipedia_search",
            "wikidata_search",
            "stackexchange_search",
            "nasa_media_search",
            "nasa_media_manifest",
            "nasa_media_metadata",
            "nasa_video_captions",
            "read_website",
            "yahoo_finance_news",
            "youtube_search",
        ]
    ] = Field(default_factory=list, max_length=4)
    research_questions: list[str] = Field(default_factory=list, max_length=10)
    search_queries: list[str] = Field(default_factory=list, max_length=8)
    source_types: list[str] = Field(default_factory=list, max_length=8)
    required_topics: list[str] = Field(default_factory=list, max_length=12)
    perspectives: list[str] = Field(default_factory=list, max_length=8)
    report_outline: list[str] = Field(default_factory=list, max_length=10)
    output_constraints: list[str] = Field(default_factory=list, max_length=8)
    requires_options: bool = False
    requires_recommendation: bool = False
    requires_implementation_plan: bool = False
    separate_facts_and_assumptions: bool = False
    requires_counterarguments: bool = False

    @model_validator(mode="after")
    def validate_research_queries(self) -> "ResearchPlan":
        if self.mode == "research":
            budget = _BUDGETS[self.depth]
            if len(self.search_queries) < budget.min_queries:
                raise ValueError(
                    f"{self.depth} research requires at least "
                    f"{budget.min_queries} search queries"
                )
            if not self.perspectives:
                raise ValueError("research mode requires at least one perspective")
            if not self.required_topics:
                raise ValueError("research mode requires at least one required topic")
        if self.mode == "direct" and not self.direct_tools:
            raise ValueError("direct mode requires at least one selected tool")
        return self


class EvidenceSource(BaseModel):
    """One successfully read, non-Wikipedia source."""

    url: str
    title: str = ""
    content: str
    provider: str
    query: str = ""
    published_date: str = ""


class PerspectiveFinding(BaseModel):
    """Evidence-backed analysis for one planned perspective."""

    perspective: str
    analysis: str
    source_urls: list[str] = Field(default_factory=list)


class TopicFinding(BaseModel):
    """Evidence-backed coverage of one explicitly requested topic."""

    topic: str
    analysis: str
    source_urls: list[str] = Field(default_factory=list)


class ResearchOption(BaseModel):
    """One compared option with explicit trade-offs."""

    name: str
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Structured synthesis prevents requested analysis from being discarded."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Research report"
    answer: str
    topic_findings: list[TopicFinding] = Field(default_factory=list)
    perspective_findings: list[PerspectiveFinding] = Field(default_factory=list)
    established_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    options: list[ResearchOption] = Field(default_factory=list)
    recommendation: str = ""
    implementation_steps: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    """One bounded assessment of evidence coverage before deep synthesis."""

    model_config = ConfigDict(extra="forbid")

    covered_perspectives: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=4)
    ready_to_synthesize: bool


_PLAN_PROMPT = """\
You decide how an information specialist should handle a request based on its
meaning, never keyword matching. Use mode "direct" for a single factual lookup,
weather lookup, news lookup, URL read, or another narrow answer. Use mode
"research" for analysis, comparison, recommendation, planning, investigation,
or a report.

For research mode, choose depth from the requested scope and intended output:
- brief: a focused explanation supported by 2-3 sources and two searches.
- standard: a substantive analysis supported by 3-5 sources and 3-5 searches.
- deep: a research document or broad situation analysis supported by 6-10
  independent sources, 5-8 searches, and a post-retrieval gap assessment.

Produce complementary searches, explicit questions, source types, analytical
perspectives, and a useful report outline. Copy every explicitly requested
subject or analytical dimension into required_topics without merging or omitting
items. Perspectives are viewpoints such as affected stakeholders; required_topics
are subjects that must be covered, such as technical feasibility or lifecycle
economics. Include every user constraint. Cover requested evidence classes such
as official or primary evidence, independent analysis, criticism, affected
stakeholders, alternatives, risks, and uncertainty when relevant. For direct
mode, select only the smallest relevant tool set in direct_tools. Set the
requirement booleans from the requested deliverable. Do not choose URLs.
"""

_SYNTHESIS_PROMPT = """\
You are Heathcliff's evidence-bound research writer. Answer only from the
provided successfully extracted sources. Cite factual claims with direct source
URLs in Markdown. Compare the sources, explicitly record material disagreement
in conflicts, and record evidence gaps in limitations. Honour the requested
format and length. Never claim that research was comprehensive when fewer than
the plan's required number of independent sources were successfully read. For
planning and situation analysis, separate facts, assumptions, options,
trade-offs, risks, and recommendations. Follow the plan's report outline and
address every listed perspective. For every plan perspective, emit one
perspective_findings item using the exact perspective name and cite its source
URLs. For every required topic, emit one topic_findings item using the exact
topic name. Populate options, recommendation, implementation_steps,
established_facts, assumptions, and counterarguments whenever the corresponding
plan requirement is true.
"""

_ASSESSMENT_PROMPT = """\
Assess whether the extracted evidence covers the research plan's questions,
source types, and perspectives. Identify concrete gaps and produce only the
smallest set of complementary follow-up searches needed. Do not repeat existing
queries. Mark ready_to_synthesize only when the evidence is broad enough for the
requested depth; source count alone is not sufficient.
"""


def plan_request(model: Any, request: str, callbacks: Sequence[Any]) -> ResearchPlan:
    """Ask the model for a typed semantic depth decision."""
    messages = [SystemMessage(_PLAN_PROMPT), HumanMessage(request)]
    return _invoke_structured(model, ResearchPlan, messages, callbacks)


def _invoke_structured(
    model: Any,
    schema: type[BaseModel],
    messages: list[Any],
    callbacks: Sequence[Any],
) -> Any:
    """Invoke strict structured output once, then repair one invalid response."""
    runnable = model.with_structured_output(schema)
    config = {"callbacks": list(callbacks)} if callbacks else None
    try:
        return runnable.invoke(messages, config) if config else runnable.invoke(messages)
    except Exception:
        repaired = [
            *messages,
            HumanMessage(
                "The previous result failed schema validation. Return a complete "
                "result satisfying every field constraint in the requested schema."
            ),
        ]
        return runnable.invoke(repaired, config) if config else runnable.invoke(repaired)


def _invoke_tool(tool: Any, args: dict[str, Any], callbacks: Sequence[Any]) -> Any:
    config = {"callbacks": list(callbacks)} if callbacks else None
    result = tool.invoke(args, config) if config else tool.invoke(args)
    record_tool_invocation("info_agent", getattr(tool, "name", "tool"), args, result)
    return result


def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _is_independent_url(url: str) -> bool:
    domain = _domain(url)
    return bool(domain) and not (
        domain == "wikipedia.org" or domain.endswith(".wikipedia.org")
    )


def _candidates_from_tavily(raw: Any, query: str) -> list[dict[str, str]]:
    if not isinstance(raw, dict) or not raw.get("ok"):
        return []
    candidates = []
    for item in raw.get("results", []):
        if not isinstance(item, dict) or not item.get("url"):
            continue
        candidates.append(
            {
                "url": str(item["url"]),
                "title": str(item.get("title", "")),
                "query": query,
                "published_date": str(item.get("published_date", "")),
            }
        )
    return candidates


def _candidates_from_web(raw: str, query: str) -> list[dict[str, str]]:
    if raw.startswith("Web search failed"):
        return []
    candidates = []
    for line in raw.splitlines():
        match = _URL_RE.search(line)
        if not match:
            continue
        candidates.append(
            {
                "url": match.group(0).rstrip(".,;"),
                "title": line.split("|")[0].strip(" -"),
                "query": query,
                "published_date": "",
            }
        )
    return candidates


def _select_candidates(
    candidates: list[dict[str, str]],
    max_sources: int,
    excluded_domains: set[str] | None = None,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    domains = set(excluded_domains or ())
    buckets: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        buckets.setdefault(candidate["query"], []).append(candidate)

    while buckets and len(selected) < max_sources:
        for query in list(buckets):
            bucket = buckets[query]
            while bucket:
                candidate = bucket.pop(0)
                url = candidate["url"]
                domain = _domain(url)
                if _is_independent_url(url) and domain not in domains:
                    selected.append(candidate)
                    domains.add(domain)
                    break
            if not bucket:
                buckets.pop(query)
            if len(selected) == max_sources:
                break
    return selected


def collect_evidence(
    plan: ResearchPlan,
    callbacks: Sequence[Any],
    existing_sources: Sequence[EvidenceSource] = (),
) -> tuple[list[EvidenceSource], list[str]]:
    """Search, select independent domains, and read the selected pages."""
    budget = _BUDGETS[plan.depth]
    candidates: list[dict[str, str]] = []
    limitations: list[str] = []
    tavily_enabled = bool(Config.TAVILY_API_KEY)

    for query in plan.search_queries:
        query_candidates: list[dict[str, str]] = []
        if tavily_enabled:
            raw = _invoke_tool(tavily_search, {"query": query}, callbacks)
            query_candidates = _candidates_from_tavily(raw, query)
        if not query_candidates:
            raw = _invoke_tool(search_web, {"query": query}, callbacks)
            query_candidates = _candidates_from_web(str(raw), query)
        if not query_candidates:
            limitations.append(f"No usable search results for: {query}")
        candidates.extend(query_candidates)

    source_ceiling = budget.max_sources if existing_sources else budget.initial_sources
    remaining_capacity = max(0, source_ceiling - len(existing_sources))
    selected = _select_candidates(
        candidates,
        remaining_capacity,
        {_domain(source.url) for source in existing_sources},
    )
    if not selected:
        return [], limitations

    extracted: dict[str, str] = {}
    urls = [candidate["url"] for candidate in selected]
    if tavily_enabled:
        raw = _invoke_tool(
            tavily_extract,
            {"urls": urls, "query": " ".join(plan.research_questions)},
            callbacks,
        )
        if isinstance(raw, dict) and raw.get("ok"):
            for item in raw.get("results", []):
                if isinstance(item, dict) and item.get("url") and item.get("raw_content"):
                    extracted[str(item["url"])] = str(item["raw_content"])

    sources: list[EvidenceSource] = []
    for candidate in selected:
        url = candidate["url"]
        content = extracted.get(url, "")
        provider = "tavily"
        if not content:
            content = str(_invoke_tool(read_website, {"url": url}, callbacks))
            provider = "website"
        if content.startswith("Error") or not content.strip():
            limitations.append(f"Could not extract source: {url}")
            continue
        sources.append(
            EvidenceSource(
                **candidate,
                content=content[: budget.source_chars],
                provider=provider,
            )
        )

    return sources, list(dict.fromkeys(limitations))


def assess_evidence(
    model: Any,
    plan: ResearchPlan,
    sources: Sequence[EvidenceSource],
    callbacks: Sequence[Any],
) -> EvidenceAssessment:
    """Find unanswered perspectives before a deep report is written."""
    payload = {
        "plan": plan.model_dump(),
        "sources": [
            {
                "url": source.url,
                "title": source.title,
                "query": source.query,
                "content": source.content[:2500],
            }
            for source in sources
        ],
    }
    messages = [
        SystemMessage(_ASSESSMENT_PROMPT),
        HumanMessage(json.dumps(payload, ensure_ascii=False)),
    ]
    return _invoke_structured(model, EvidenceAssessment, messages, callbacks)


def run_research_workflow(
    model: Any,
    request: str,
    plan: ResearchPlan,
    callbacks: Sequence[Any],
) -> str:
    """Collect, assess, optionally fill gaps, and synthesize one research request."""
    sources, limitations = collect_evidence(plan, callbacks)
    assessment: EvidenceAssessment | None = None

    if plan.depth == "deep" and sources:
        for _ in range(_BUDGETS[plan.depth].gap_rounds):
            try:
                assessment = assess_evidence(model, plan, sources, callbacks)
            except Exception as exc:
                limitations.append(f"Evidence gap assessment failed: {exc}")
                break
            if assessment.ready_to_synthesize or not assessment.follow_up_queries:
                break
            follow_up_plan = plan.model_copy(
                update={"search_queries": assessment.follow_up_queries}
            )
            added, added_limitations = collect_evidence(
                follow_up_plan, callbacks, existing_sources=sources
            )
            sources.extend(added)
            limitations.extend(added_limitations)
            if not added:
                break

        if assessment is not None and assessment.follow_up_queries:
            try:
                assessment = assess_evidence(model, plan, sources, callbacks)
            except Exception as exc:
                limitations.append(f"Final evidence assessment failed: {exc}")

    if assessment is not None and not assessment.ready_to_synthesize:
        limitations.extend(f"Unresolved evidence gap: {gap}" for gap in assessment.gaps)

    limitations = list(dict.fromkeys(limitations))
    try:
        return synthesize_report(
            model,
            request,
            plan,
            sources,
            limitations,
            callbacks,
            assessment,
        )
    except Exception as exc:
        return _partial_report(
            plan,
            sources,
            [*limitations, f"Final structured synthesis failed: {exc}"],
        )


def _partial_report(
    plan: ResearchPlan,
    sources: Sequence[EvidenceSource],
    limitations: Sequence[str],
) -> str:
    """Return collected evidence without inventing a synthesis."""
    budget = _BUDGETS[plan.depth]
    sections = ["## Research incomplete"]
    if sources:
        sections.append(
            "## Retrieved evidence\n\n"
            + "\n\n".join(
                f"### {source.title or _domain(source.url)}\n\n"
                f"{source.content[:600].strip()}\n\n[Source]({source.url})"
                for source in sources
            )
        )
    all_limitations = list(limitations)
    if len(sources) < budget.min_sources:
        all_limitations.append(
            f"Only {len(sources)} of at least {budget.min_sources} required "
            "independent sources were read."
        )
    sections.append(
        "## Limitations\n\n" + "\n".join(f"- {item}" for item in all_limitations)
    )
    return "\n\n".join(sections)


def synthesize_report(
    model: Any,
    request: str,
    plan: ResearchPlan,
    sources: list[EvidenceSource],
    limitations: list[str],
    callbacks: Sequence[Any],
    assessment: EvidenceAssessment | None = None,
) -> str:
    """Produce a cited report or a deterministic evidence-bound partial result."""
    budget = _BUDGETS[plan.depth]
    if not sources:
        limitations = [
            *limitations,
            f"No sources were read; {plan.depth} research requires at least "
            f"{budget.min_sources} independent sources.",
        ]
        details = "\n".join(f"- {item}" for item in limitations)
        return f"## Research incomplete\n\nNo usable sources were retrieved.\n\n{details}"

    if len(sources) < budget.min_sources:
        limitations = [
            *limitations,
            f"Only {len(sources)} of at least {budget.min_sources} independent "
            f"sources required for {plan.depth} research were successfully read.",
        ]

    payload = {
        "request": request,
        "plan": plan.model_dump(),
        "sources": [source.model_dump() for source in sources],
        "known_limitations": limitations,
        "evidence_assessment": assessment.model_dump() if assessment else None,
    }
    messages = [
        SystemMessage(_SYNTHESIS_PROMPT),
        HumanMessage(json.dumps(payload, ensure_ascii=False)),
    ]
    report = _invoke_structured(model, ResearchReport, messages, callbacks)

    missing = _missing_report_requirements(report, plan)
    if missing:
        report = _invoke_structured(
            model,
            ResearchReport,
            [
                *messages,
                HumanMessage(
                    "Rewrite the report. It omitted these mandatory deliverables: "
                    + "; ".join(missing)
                ),
            ],
            callbacks,
        )
        missing = _missing_report_requirements(report, plan)

    sections = [f"# {report.title.strip()}"] if plan.depth == "deep" else []
    if missing:
        sections.append("## Research incomplete")
        limitations.extend(f"Missing required deliverable: {item}" for item in missing)
    if plan.depth == "deep":
        sections.append(f"## Executive summary\n\n{report.answer.strip()}")
    else:
        sections.append(report.answer.strip())
    if report.topic_findings:
        sections.append(
            "## Findings by requested topic\n\n"
            + "\n\n".join(
                f"### {finding.topic}\n\n{finding.analysis}"
                for finding in report.topic_findings
            )
        )
    if report.perspective_findings:
        sections.append(
            "## Analysis by perspective\n\n"
            + "\n\n".join(
                f"### {finding.perspective}\n\n{finding.analysis}"
                for finding in report.perspective_findings
            )
        )
    if report.established_facts:
        sections.append(
            "## Established facts\n\n"
            + "\n".join(f"- {item}" for item in report.established_facts)
        )
    if report.assumptions:
        sections.append(
            "## Assumptions\n\n"
            + "\n".join(f"- {item}" for item in report.assumptions)
        )
    if report.options:
        sections.append(
            "## Options and trade-offs\n\n"
            + "\n\n".join(
                f"### {option.name}\n\n"
                + "**Advantages**\n"
                + "\n".join(f"- {item}" for item in option.advantages)
                + "\n\n**Disadvantages**\n"
                + "\n".join(f"- {item}" for item in option.disadvantages)
                for option in report.options
            )
        )
    if report.recommendation:
        sections.append(f"## Recommendation\n\n{report.recommendation}")
    if report.implementation_steps:
        sections.append(
            "## Implementation plan\n\n"
            + "\n".join(
                f"{number}. {item}"
                for number, item in enumerate(report.implementation_steps, 1)
            )
        )
    if report.counterarguments:
        sections.append(
            "## Counterarguments\n\n"
            + "\n".join(f"- {item}" for item in report.counterarguments)
        )
    if report.conflicts:
        sections.append("## Conflicting evidence\n\n" + "\n".join(
            f"- {item}" for item in report.conflicts
        ))
    all_limitations = list(dict.fromkeys([*limitations, *report.limitations]))
    if all_limitations:
        sections.append("## Limitations\n\n" + "\n".join(
            f"- {item}" for item in all_limitations
        ))

    sections.append("## Sources\n\n" + "\n".join(
            f"- [{source.title or _domain(source.url)}]({source.url})"
            for source in sources
        ))
    return "\n\n".join(section for section in sections if section)


def _missing_report_requirements(
    report: ResearchReport, plan: ResearchPlan
) -> list[str]:
    """Check semantic plan requirements against typed report sections."""
    missing = []
    topics = {item.topic.strip().casefold() for item in report.topic_findings}
    missing_topics = [
        topic for topic in plan.required_topics if topic.strip().casefold() not in topics
    ]
    if missing_topics:
        missing.append("required topics: " + ", ".join(missing_topics))
    findings = {item.perspective.strip().casefold() for item in report.perspective_findings}
    missing_perspectives = [
        perspective
        for perspective in plan.perspectives
        if perspective.strip().casefold() not in findings
    ]
    if missing_perspectives:
        missing.append("perspectives: " + ", ".join(missing_perspectives))
    if plan.requires_options and not report.options:
        missing.append("options and trade-offs")
    if plan.requires_recommendation and not report.recommendation.strip():
        missing.append("recommendation")
    if plan.requires_implementation_plan and not report.implementation_steps:
        missing.append("implementation plan")
    if plan.requires_counterarguments and not report.counterarguments:
        missing.append("counterarguments")
    if plan.separate_facts_and_assumptions:
        if not report.established_facts:
            missing.append("established facts")
        if not report.assumptions:
            missing.append("assumptions")
    return missing
