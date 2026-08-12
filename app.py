"""Streamlit interface for the Daniel → Arthur → Lex pipeline."""

from __future__ import annotations

from typing import Any

import streamlit as st

from main import (
    ArthurSalesCopywriterAgent,
    DanielResearchAgent,
    LexLegalComplianceAgent,
)


DEFAULT_TOPIC = "AI automation for sales and growth"


def run_pipeline(topic: str, source_count: int) -> dict[str, Any]:
    """Run Daniel's research, Arthur's copywriting, and Lex's review."""
    report = DanielResearchAgent(max_sources=source_count).research(topic)
    marketing_message = ArthurSalesCopywriterAgent().write(report)
    report["marketing_message"] = marketing_message
    report["compliance_review"] = LexLegalComplianceAgent().review(
        report,
        marketing_message,
    )
    return report


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render source links and extracted summaries."""
    if not sources:
        st.info("No readable sources were returned.")
        return

    for source in sources:
        index = source.get("index", "")
        title = source.get("title", source.get("domain", "Source"))
        url = source.get("url", "")
        with st.expander(f"[{index}] {title}"):
            if url:
                st.markdown(f"[Open source]({url})")
            st.write(f"Domain: {source.get('domain', 'unknown')}")
            st.write(f"Relevance: {source.get('relevance', 'n/a')}")
            st.write(f"Words extracted: {source.get('word_count', 'n/a')}")
            if source.get("summary"):
                st.write(source["summary"])
            if source.get("error"):
                st.warning(str(source["error"]))


def render_findings(findings: list[dict[str, Any]]) -> None:
    """Render Daniel's findings with citations."""
    if not findings:
        st.info("Daniel did not return any findings.")
        return

    for finding in findings:
        citation = finding.get("citation", "")
        source = finding.get("source", "unknown source")
        st.markdown(f"- {finding.get('claim', '')} **{citation}** ({source})")


def render_marketing_message(message: dict[str, Any]) -> None:
    """Render Arthur's final sales message."""
    st.subheader(str(message.get("headline", "Sales message")))
    st.write(str(message.get("body", "")))
    st.markdown(f"**Call to action:** {message.get('call_to_action', '')}")

    proof_points = message.get("proof_points", [])
    if proof_points:
        st.markdown("**Research-backed proof points:**")
        for point in proof_points:
            st.markdown(
                f"- {point.get('text', '')} {point.get('citation', '')}"
            )

    st.caption(str(message.get("disclosure", "")))


def render_compliance_review(review: dict[str, Any]) -> None:
    """Render Lex's compliance status and review items."""
    status = str(review.get("status", "unknown"))
    if status == "needs_review":
        st.warning(f"Lex status: {status}")
    else:
        st.success(f"Lex status: {status}")

    st.write(str(review.get("publishing_guidance", "")))
    st.write(f"Issues found: {review.get('issue_count', 0)}")

    issues = review.get("issues", [])
    if issues:
        for issue in issues:
            st.markdown(
                f"- **{issue.get('type', 'review item')}:** "
                f"{issue.get('text', '')} — {issue.get('guidance', '')}"
            )
    else:
        st.info("Lex found no common automated red flags.")

    st.caption(str(review.get("disclaimer", "")))


def main() -> None:
    st.set_page_config(
        page_title="Daniel, Arthur & Lex",
        page_icon="D",
        layout="wide",
    )
                if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
                    
    max_free_trials = 3
    remaining_trials = max_free_trials - st.session_state.usage_count

    st.title("Research to Sales Review")
    st.write(
        "Enter a topic to run Daniel's research, have Arthur write a "
        "professional sales message, and have Lex review it for U.S. "
        "federal and state legal risks."
    )

    with st.sidebar:
        st.header("Pipeline settings")
            st.info(f"💡 Free trials remaining: {max(0, remaining_trials)} of {max_free_trials}")

        source_count = st.slider(
            "Readable sources",
            min_value=2,
            max_value=20,
            value=5,
            help="Daniel will include up to this many readable sources.",
        )
        st.caption(
            "Lex provides an automated pre-publication screen, not legal advice."
        )

    topic = st.text_area(
        "Research topic",
        value=DEFAULT_TOPIC,
        height=90,
        placeholder="What should Daniel research?",
    )

        if st.button("Run research pipeline", type="primary"):
        cleaned_topic = " ".join(topic.split())
        if not cleaned_topic:
            st.error("Enter a research topic before running the pipeline.")
            return

        if st.session_state.usage_count < max_free_trials:
            st.session_state.usage_count += 1
            
            with st.spinner("Daniel is researching, Arthur is writing, and Lex is reviewing..."):
                try:
                    st.session_state["report"] = run_pipeline(
                        cleaned_topic,
                        source_count,
                    )
                except (RuntimeError, ValueError) as error:
                    st.error(f"The pipeline could not complete: {error}")
                    return
        else:
            st.error("⚠️ You have exhausted all your free trials (3/3). Please upgrade or contact support to continue.")


    report = st.session_state.get("report")
    if not report:
        st.info("Run the pipeline to see the research report and reviewed sales message.")
        return

    st.divider()
    st.header("Research report")
    st.write(f"**Topic:** {report.get('topic', '')}")
    st.write(f"**Jurisdiction:** {report.get('jurisdiction', 'Not specified')}")
    st.write(f"**Scope:** {report.get('scope', 'Not specified')}")

    summary, findings = st.columns([1, 2])
    with summary:
        st.subheader("Executive summary")
        st.write(str(report.get("summary", "")))
    with findings:
        st.subheader("Findings")
        render_findings(report.get("findings", []))

    with st.expander("Sources"):
        render_sources(report.get("sources", []))

    marketing = report.get("marketing_message")
    if isinstance(marketing, dict):
        st.divider()
        st.header("Arthur's sales message")
        render_marketing_message(marketing)

    compliance = report.get("compliance_review")
    if isinstance(compliance, dict):
        st.divider()
        st.header("Lex's legal and compliance review")
        st.write(f"**Jurisdiction:** {compliance.get('jurisdiction', '')}")
        st.write(f"**Scope:** {compliance.get('scope', '')}")
        render_compliance_review(compliance)

    limitations = report.get("limitations", [])
    if limitations:
        with st.expander("Research limitations"):
            for limitation in limitations:
                st.markdown(f"- {limitation}")


if __name__ == "__main__":
    main()
