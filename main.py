"""Daniel, Arthur, and Lex - a research-to-sales messaging pipeline.

Daniel uses DuckDuckGo's public HTML search page and Python's standard library.
It does not require an API key or third-party package.  Example:

    python main.py "How does regenerative agriculture affect soil health?"
    python main.py --sources 8 --json "The history of open source software"

After researching a topic, Daniel passes the complete report directly to
Arthur, the sales copywriter agent. Arthur creates a professional marketing
message from the retrieved findings without inventing unsupported statistics
or claims. Lex, the legal and compliance agent, reviews Arthur's output before
it is presented as ready for publication.
"""

from __future__ import annotations

import argparse
import html
import ipaddress
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from textwrap import shorten
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


AGENT_NAME = "Daniel"
US_JURISDICTION = "United States federal and state law"
DANIEL_SYSTEM_PROMPT = """
You are Daniel, a research agent focused strictly on United States law.
Research U.S. federal and state authorities, including the U.S. Copyright Act,
Federal Trade Commission (FTC) guidance and rules, federal regulations and
statutes, federal case law, California CCPA/CPRA, Illinois BIPA, and other
state privacy, biometric, consumer-protection, and AI regulations and case
law. Identify the jurisdiction for every material authority and distinguish
binding law from agency guidance, proposed rules, commentary, and unresolved
legal questions. Exclude foreign and international law unless it is mentioned
only to explain that it is outside scope. Do not give legal advice.
""".strip()
LEX_SYSTEM_PROMPT = """
You are Lex, a legal and compliance review agent focused strictly on United
States law. Review marketing copy under U.S. federal and state requirements,
especially Section 5 of the FTC Act and related FTC advertising guidance, the
U.S. Copyright Act, California CCPA/CPRA, Illinois BIPA, and other applicable
state privacy, biometric, consumer-protection, and AI laws. Check for
deceptive, misleading, unsubstantiated, copyright, privacy, biometric,
consumer-protection, and unsupported legal claims. Identify potentially
applicable jurisdictions and flag issues for qualified U.S. counsel. Exclude
foreign and international law from the review. Do not give a legal opinion or
determine that copy is legally compliant.
""".strip()
SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
USER_AGENT = (
    "DanielResearchAgent/1.0 "
    "(educational research assistant; contact unavailable)"
)
STOP_WORDS = {
    "about",
    "after",
    "also",
    "because",
    "been",
    "being",
    "could",
    "from",
    "have",
    "into",
    "more",
    "other",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "were",
    "which",
    "with",
    "would",
}


@dataclass(frozen=True)
class SearchResult:
    """A result returned by a search engine."""

    title: str
    url: str
    snippet: str = ""


@dataclass
class Source:
    """A downloaded source and Daniel's small extracted summary."""

    index: int
    title: str
    url: str
    domain: str
    summary: str
    relevance: float
    word_count: int
    error: str | None = None


class SearchResultParser(HTMLParser):
    """Extract DuckDuckGo result links without requiring BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current: dict[str, str] | None = None
        self._in_title = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if tag == "a" and "result__a" in classes:
            self._current = {
                "title": "",
                "url": attributes.get("href") or "",
                "snippet": "",
            }
            self._title_parts = []
            self._in_title = True
        elif self._current and "result__snippet" in classes:
            self._snippet_parts = []
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._current and self._in_title:
            self._title_parts.append(data)
        if self._current and self._in_snippet:
            self._snippet_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current and self._in_title:
            self._current["title"] = clean_text(" ".join(self._title_parts))
            self._in_title = False
        elif self._current and self._in_snippet and tag in {"a", "div"}:
            self._current["snippet"] = clean_text(" ".join(self._snippet_parts))
            self._in_snippet = False
            if self._current["title"] and self._current["url"]:
                self.results.append(SearchResult(**self._current))
                self._current = None


class PageParser(HTMLParser):
    """Turn the readable parts of an HTML page into plain text."""

    ignored_tags = {"script", "style", "noscript", "svg", "template", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "h1", "h2", "h3", "li", "br"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = clean_text(data)
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        self.text_parts.append(value)


@dataclass
class FetchedPage:
    title: str
    text: str
    status: int


def clean_text(value: str) -> str:
    """Normalize HTML text and whitespace."""
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def words(value: str) -> set[str]:
    """Return useful lowercase terms for simple relevance scoring."""
    return {
        word
        for word in re.findall(r"[a-z0-9]{3,}", value.lower())
        if word not in STOP_WORDS
    }


def domain_from_url(url: str) -> str:
    return (urlparse(url).netloc or "unknown").lower().removeprefix("www.")


def unwrap_search_url(url: str) -> str:
    """Resolve DuckDuckGo's redirect links to the actual destination."""
    parsed = urlparse(html.unescape(url))
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else url


def is_public_http_url(url: str) -> bool:
    """Avoid fetching local files or obvious private network addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def request_bytes(url: str, timeout: float = 12.0) -> tuple[int, bytes]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read(2_000_000)


def request_form_bytes(
    url: str, form: dict[str, str], timeout: float = 12.0
) -> tuple[int, bytes]:
    """Send a browser-like form request to a search endpoint."""
    body = urlencode(form).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read(2_000_000)


def search_web(query: str, limit: int) -> list[SearchResult]:
    """Search the public web and return unique, public HTTP results."""
    url = SEARCH_URL.format(query=quote_plus(query))
    _, body = request_bytes(url)
    parser = SearchResultParser()
    parser.feed(body.decode("utf-8", errors="replace"))

    # Some network environments receive DuckDuckGo's anti-automation page for
    # GET requests, but its normal HTML results remain available via POST.
    if not parser.results:
        _, body = request_form_bytes(
            "https://html.duckduckgo.com/html/",
            {"q": query},
        )
        parser = SearchResultParser()
        parser.feed(body.decode("utf-8", errors="replace"))

    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in parser.results:
        destination = unwrap_search_url(result.url)
        if destination in seen or not is_public_http_url(destination):
            continue
        seen.add(destination)
        unique.append(SearchResult(result.title, destination, result.snippet))
        if len(unique) >= limit:
            break
    return unique


def fetch_page(url: str) -> FetchedPage:
    """Download one page and extract its readable text."""
    status, body = request_bytes(url)
    parser = PageParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    text = clean_text(" ".join(parser.text_parts))
    title = clean_text(" ".join(parser.title_parts)) or domain_from_url(url)
    return FetchedPage(title=title, text=text, status=status)


def sentence_list(text: str) -> list[str]:
    text = clean_text(text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.split()) >= 8
    ]


def best_sentences(text: str, topic: str, count: int = 2) -> list[str]:
    """Choose topic-relevant sentences without pretending to be an LLM."""
    topic_terms = words(topic)
    candidates = sentence_list(text)
    scored: list[tuple[float, int, str]] = []
    for position, sentence in enumerate(candidates):
        sentence_terms = words(sentence)
        overlap = len(topic_terms & sentence_terms)
        length_penalty = max(0, len(sentence.split()) - 42) / 42
        score = overlap * 3 - length_penalty - position / 10_000
        scored.append((score, position, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = sorted(scored[:count], key=lambda item: item[1])
    return [sentence for _, _, sentence in selected]


def source_relevance(topic: str, result: SearchResult, page: FetchedPage) -> float:
    topic_terms = words(topic)
    haystack = words(f"{result.title} {result.snippet} {page.title} {page.text[:12000]}")
    return len(topic_terms & haystack) / max(len(topic_terms), 1)


class DanielResearchAgent:
    """A transparent research workflow with no hidden model or API dependency."""

    def __init__(self, max_sources: int = 5, pause_seconds: float = 0.4) -> None:
        self.max_sources = max(2, min(max_sources, 20))
        self.pause_seconds = max(0.0, pause_seconds)

    def research(self, topic: str) -> dict[str, object]:
        topic = clean_text(topic)
        if not topic:
            raise ValueError("A research topic is required.")

        us_legal_topic = (
            f"{topic} United States federal and state law U.S. Copyright Act "
            "FTC California CCPA CPRA Illinois BIPA state privacy AI regulation"
        )
        queries = [
            us_legal_topic,
            f"{topic} U.S. federal statute FTC guidance state law",
            f"{topic} California CCPA CPRA Illinois BIPA state AI law",
        ]
        results: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            try:
                for result in search_web(query, self.max_sources):
                    if result.url not in seen:
                        results.append(result)
                        seen.add(result.url)
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                if not results:
                    raise RuntimeError(f"Search failed: {error}") from error
            if len(results) >= self.max_sources * 2:
                break
            time.sleep(self.pause_seconds)

        sources: list[Source] = []
        for result in results[: self.max_sources * 2]:
            try:
                page = fetch_page(result.url)
                if len(page.text.split()) < 40:
                    continue
                sources.append(
                    Source(
                        index=0,
                        title=page.title or result.title,
                        url=result.url,
                        domain=domain_from_url(result.url),
                        summary=" ".join(best_sentences(page.text, topic)) or result.snippet,
                        relevance=round(source_relevance(topic, result, page), 3),
                        word_count=len(page.text.split()),
                    )
                )
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                continue
            time.sleep(self.pause_seconds)

        sources = sorted(
            sources,
            key=lambda source: source.relevance,
            reverse=True,
        )[: self.max_sources]
        for index, source in enumerate(sources, start=1):
            source.index = index

        return {
            "agent": AGENT_NAME,
            "jurisdiction": US_JURISDICTION,
            "scope": (
                "U.S. federal and state statutes, regulations, agency guidance, "
                "and case law only."
            ),
            "topic": topic,
            "summary": self._summary(sources),
            "findings": [
                {
                    "claim": source.summary,
                    "citation": f"[{source.index}]",
                    "source": source.domain,
                }
                for source in sources
                if source.summary
            ],
            "sources": [asdict(source) for source in sources],
            "limitations": [
                "This report is extractive and does not independently verify every claim.",
                "Search rankings and page availability can change over time.",
                "Read the linked primary sources before making high-stakes decisions.",
            ],
        }

    @staticmethod
    def _summary(sources: Iterable[Source]) -> str:
        summaries = [source.summary for source in sources if source.summary]
        if not summaries:
            return "Daniel could not retrieve enough readable sources to form a summary."
        return shorten(summaries[0], width=420, placeholder="…")


class ArthurSalesCopywriterAgent:
    """Turn Daniel's evidence into a professional, citation-aware sales message.

    This is deliberately implemented with deterministic Python rather than a
    hidden model call. The handoff is the report dictionary itself, so Arthur
    can only use claims and sources Daniel actually returned.
    """

    def __init__(self, audience: str = "growing sales teams") -> None:
        self.audience = audience.strip() or "growing sales teams"

    def write(self, research_report: dict[str, object]) -> dict[str, object]:
        """Generate marketing copy directly from Daniel's research results."""
        topic = str(research_report.get("topic", "sales growth")).strip()
        findings = self._findings(research_report)
        sources = self._sources(research_report)

        lead_claim = self._claim_text(findings[0]) if findings else (
            "Turn repetitive sales work into more time for meaningful customer conversations."
        )
        supporting_claims = [
            self._claim_text(finding)
            for finding in findings[1:3]
            if self._claim_text(finding)
        ]
        proof_points = [
            {
                "text": claim,
                "citation": str(finding.get("citation", "")),
            }
            for finding, claim in zip(findings[1:3], supporting_claims)
        ]

        headline = self._headline(topic)
        body = self._body(lead_claim, supporting_claims)
        cta = (
            f"Start building a smarter {topic.lower()} workflow today — "
            "and give your team more time to sell."
        )

        return {
            "agent": "Arthur",
            "role": "Sales copywriter",
            "audience": self.audience,
            "headline": headline,
            "body": body,
            "proof_points": proof_points,
            "call_to_action": cta,
            "source_count": len(sources),
            "source_domains": [
                str(source.get("domain", ""))
                for source in sources
                if source.get("domain")
            ],
            "disclosure": (
                "Professional sales copy written by Arthur from Daniel's extracted research; "
                "review source claims before publishing."
            ),
        }

    @staticmethod
    def _headline(topic: str) -> str:
        subject = topic.strip().rstrip(".!?")
        return f"Make {subject.lower()} your next growth advantage"

    @staticmethod
    def _body(lead_claim: str, supporting_claims: list[str]) -> str:
        paragraphs = [
            (
                f"Your team should spend less time on repetitive work and more time "
                f"creating revenue. Research on this opportunity shows that {lead_claim}"
            )
        ]
        if supporting_claims:
            paragraphs.append(
                "The strongest teams are pairing automation with better pipeline "
                "visibility, sharper prioritization, and more relevant customer "
                f"engagement. {supporting_claims[0]}"
            )
        paragraphs.append(
            "Use automation to support your people, not replace their judgment. "
            "Start with one high-friction workflow, measure the outcome, and scale "
            "what genuinely improves the customer experience."
        )
        return "\n\n".join(paragraphs)

    @staticmethod
    def _findings(report: dict[str, object]) -> list[dict[str, object]]:
        findings = report.get("findings", [])
        return [item for item in findings if isinstance(item, dict)]

    @staticmethod
    def _sources(report: dict[str, object]) -> list[dict[str, object]]:
        sources = report.get("sources", [])
        return [item for item in sources if isinstance(item, dict)]

    @staticmethod
    def _claim_text(finding: dict[str, object]) -> str:
        return clean_text(str(finding.get("claim", ""))).rstrip(".")


class LexLegalComplianceAgent:
    """Review Arthur's sales message under U.S. federal and state requirements.

    Lex is a practical pre-publication check, not a substitute for counsel.
    The review receives both Arthur's message and Daniel's report so it can
    identify unsupported federal- and state-law claims and preserve the
    research-to-copy trail. Foreign and international law are out of scope.
    """

    risky_patterns = (
        (
            re.compile(
                r"\b(?:guarantee(?:s|d)?|guaranteed|risk[- ]free|no risk|"
                r"zero risk|100%|always|never|number one|#1|best)\b",
                re.IGNORECASE,
            ),
            "Avoid absolute, superiority, or risk-free claims unless they are legally substantiated.",
        ),
        (
            re.compile(
                r"\b(?:will|must|cure|eliminate|eradicate|prevent)\b",
                re.IGNORECASE,
            ),
            "Review predictive or outcome claims; qualify them unless the evidence and offer terms support them.",
        ),
    )
    jurisdiction_terms = {
        "california": "California CCPA/CPRA",
        "ccpa": "California CCPA/CPRA",
        "cpra": "California CCPA/CPRA",
        "illinois": "Illinois BIPA",
        "bipa": "Illinois BIPA",
        "biometric": "state biometric privacy laws",
        "voiceprint": "state biometric privacy laws",
        "facial recognition": "state biometric and privacy laws",
        "privacy": "state privacy laws",
        "consumer data": "state privacy laws",
        "artificial intelligence": "state AI laws",
        "ai regulation": "state AI laws",
    }

    def review(
        self,
        research_report: dict[str, object],
        sales_message: dict[str, object],
    ) -> dict[str, object]:
        """Review Arthur's message directly against Daniel's research."""
        copy_text = self._copy_text(sales_message)
        findings = self._research_claims(research_report)
        issues: list[dict[str, str]] = []

        for pattern, guidance in self.risky_patterns:
            for match in pattern.finditer(copy_text):
                issues.append(
                    {
                        "type": "potentially_risky_claim",
                        "text": match.group(0),
                        "guidance": guidance,
                    }
                )

        unsupported = self._unsupported_phrases(copy_text, findings)
        issues.extend(
            {
                "type": "unsupported_claim",
                "text": phrase,
                "guidance": (
                    "Tie this claim to a source, add qualifying language, "
                    "or remove it before publication."
                ),
            }
            for phrase in unsupported
        )

        if not sales_message.get("disclosure"):
            issues.append(
                {
                    "type": "missing_disclosure",
                    "text": "No research disclosure was supplied.",
                    "guidance": "Add a clear disclosure and review citation requirements for the intended channel.",
                }
            )

        applicable_jurisdictions = self._applicable_jurisdictions(copy_text)
        if applicable_jurisdictions:
            issues.append(
                {
                    "type": "jurisdiction_review",
                    "text": ", ".join(applicable_jurisdictions),
                    "guidance": (
                        "Confirm the target states, covered data, thresholds, "
                        "notice/consent duties, exemptions, and required disclosures "
                        "with qualified U.S. counsel."
                    ),
                }
            )

        # Keep one issue per category/phrase so a repeated word does not
        # overwhelm the review.
        issues = self._unique_issues(issues)
        status = "needs_review" if issues else "approved_for_editorial_review"
        return {
            "agent": "Lex",
            "role": "Legal and compliance reviewer",
            "jurisdiction": US_JURISDICTION,
            "scope": (
                "U.S. Copyright Act, FTC Act and FTC advertising guidance, "
                "federal regulations, state privacy and biometric laws, state "
                "AI and consumer-protection laws, and related U.S. case law."
            ),
            "status": status,
            "issue_count": len(issues),
            "issues": issues,
            "publishing_guidance": (
                "Lex's automated screen found items that need human legal review "
                "under applicable U.S. federal and state requirements before publication."
                if issues
                else "No common automated red flags were detected under the "
                "configured U.S. federal-and-state-law screen; obtain final "
                "human legal approval before publication."
            ),
            "reviewed_message_agent": sales_message.get("agent", "Arthur"),
            "reviewed_source_count": len(self._sources(research_report)),
            "disclaimer": (
                "This is an automated U.S. federal-and-state-law pre-publication "
                "screen, not legal advice or a determination of compliance. "
                "Foreign and international law were not reviewed."
            ),
        }

    @staticmethod
    def _copy_text(sales_message: dict[str, object]) -> str:
        proof_points = sales_message.get("proof_points", [])
        proof_text = " ".join(
            str(point.get("text", ""))
            for point in proof_points
            if isinstance(point, dict)
        )
        return clean_text(
            " ".join(
                [
                    str(sales_message.get("headline", "")),
                    str(sales_message.get("body", "")),
                    str(sales_message.get("call_to_action", "")),
                    proof_text,
                ]
            )
        )

    @staticmethod
    def _research_claims(report: dict[str, object]) -> list[str]:
        findings = report.get("findings", [])
        return [
            clean_text(str(finding.get("claim", ""))).lower()
            for finding in findings
            if isinstance(finding, dict) and finding.get("claim")
        ]

    @classmethod
    def _applicable_jurisdictions(cls, copy_text: str) -> list[str]:
        lowered = copy_text.lower()
        return list(
            dict.fromkeys(
                jurisdiction
                for term, jurisdiction in cls.jurisdiction_terms.items()
                if term in lowered
            )
        )

    @staticmethod
    def _sources(report: dict[str, object]) -> list[dict[str, object]]:
        sources = report.get("sources", [])
        return [source for source in sources if isinstance(source, dict)]

    @staticmethod
    def _unsupported_phrases(copy_text: str, findings: list[str]) -> list[str]:
        """Find strong-looking sentences that have no research term overlap."""
        copy_sentences = sentence_list(copy_text)
        research_terms = words(" ".join(findings))
        unsupported: list[str] = []
        for sentence in copy_sentences:
            terms = words(sentence)
            if len(terms) >= 8 and len(terms & research_terms) < 2:
                unsupported.append(shorten(sentence, width=180, placeholder="…"))
        return unsupported[:3]

    @staticmethod
    def _unique_issues(
        issues: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        unique: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for issue in issues:
            key = (issue["type"], issue["text"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        return unique


def render_report(report: dict[str, object]) -> str:
    """Render a report for a human reader."""
    lines = [
        f"# {AGENT_NAME}'s Research Report",
        f"**Topic:** {report['topic']}",
        "",
        "## Executive summary",
        str(report["summary"]),
        "",
        "## Findings",
    ]
    findings = report["findings"]
    if findings:
        for finding in findings:  # type: ignore[union-attr]
            lines.append(
                f"- {finding['claim']} **{finding['citation']}** "
                f"({finding['source']})"
            )
    else:
        lines.append("- No findings were available.")

    lines.extend(["", "## Sources"])
    for source in report["sources"]:  # type: ignore[union-attr]
        if source["error"]:
            lines.append(f"- [{source['index']}] {source['url']} — {source['error']}")
        else:
            lines.append(f"- [{source['index']}] [{source['title']}]({source['url']})")

    marketing = report.get("marketing_message")
    if isinstance(marketing, dict):
        lines.extend(
            [
                "",
                "## Final marketing message",
                f"### {marketing['headline']}",
                "",
                str(marketing["body"]),
                "",
                f"**Call to action:** {marketing['call_to_action']}",
                "",
                "**Research-backed proof points:**",
            ]
        )
        proof_points = marketing.get("proof_points", [])
        if proof_points:
            lines.extend(
                f"- {point['text']} {point['citation']}"
                for point in proof_points  # type: ignore[union-attr]
            )
        else:
            lines.append("- No supporting proof points were available.")
        lines.extend(["", f"*{marketing['disclosure']}*"])

    compliance = report.get("compliance_review")
    if isinstance(compliance, dict):
        lines.extend(
            [
                "",
                "## Lex compliance review",
                f"**Status:** `{compliance['status']}`",
                f"**Issues found:** {compliance['issue_count']}",
                "",
                str(compliance["publishing_guidance"]),
            ]
        )
        issues = compliance.get("issues", [])
        if issues:
            lines.extend(
                [
                    "",
                    "**Items to review:**",
                ]
            )
            lines.extend(
                f"- **{issue['type']}:** {issue['text']} — {issue['guidance']}"
                for issue in issues  # type: ignore[union-attr]
            )
        lines.extend(["", f"*{compliance['disclaimer']}*"])

    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])  # type: ignore[union-attr]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daniel, a transparent web research agent."
    )
    parser.add_argument("topic", nargs="*", help="The question or topic to research")
    parser.add_argument(
        "--sources",
        type=int,
        default=5,
        help="Number of readable sources to include (2-20, default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of Markdown",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topic = " ".join(args.topic).strip()
    if not topic:
        topic = input("Daniel: What should I research? ").strip()
    try:
        report = DanielResearchAgent(max_sources=args.sources).research(topic)
        # Direct handoff: the copywriter receives Daniel's in-memory report,
        # rather than searching again or reading a separately rendered file.
        report["marketing_message"] = ArthurSalesCopywriterAgent().write(report)
        # Lex reviews Arthur's in-memory output against Daniel's original
        # findings before the report is rendered or serialized.
        report["compliance_review"] = LexLegalComplianceAgent().review(
            report,
            report["marketing_message"],  # type: ignore[arg-type]
        )
    except (RuntimeError, ValueError) as error:
        print(f"Daniel: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_report(report))


if __name__ == "__main__":
    main()