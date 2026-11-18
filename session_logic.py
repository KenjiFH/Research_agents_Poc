"""Pure decision logic for the Streamlit upload lifecycle, kept out of app.py so it can be tested."""


def upload_action(uploaded_file_id, current_file_id, has_workflow, has_store):
    """Decide what the app must do for the current uploader state.

    Returns one of:
      "build"          - build agents for the uploaded file (nothing to clean up first)
      "replace"        - delete the previous collection, then build for the new file
      "reset"          - the file was removed: delete the collection and clear state
      "none"           - nothing to do
    """
    if uploaded_file_id is not None:
        if not has_workflow or current_file_id != uploaded_file_id:
            return "replace" if has_store else "build"
        return "none"
    if has_workflow or has_store:
        return "reset"
    return "none"


VERDICT_LABELS = {"pass": "✅ Pass", "fail": "❌ Fail", "error": "⚠️ Error"}


def verdict_label(verdict):
    """Display text for a verdict; anything unknown shows as not graded."""
    return VERDICT_LABELS.get(verdict, "– Not graded")


def verdict_warning(faithfulness, relevance):
    """A warning for the user when the answer failed a check or could not be graded, else None."""
    if faithfulness == "fail":
        return "This answer may not be supported by the document (faithfulness check failed)."
    if faithfulness == "error":
        return "This answer could not be checked against the document (the evaluator failed)."
    if relevance == "fail":
        return "This answer may not address your question (relevance check failed)."
    if relevance == "error":
        return "Relevance could not be checked (the evaluator failed)."
    return None


NO_SOURCES_NOTE = "No verified sources for this answer."


def citation_title(citation):
    """Heading for a source card: the page and the chunk it came from."""
    page = citation.get("page_no")
    return f"Page {page if page is not None else '?'} · {citation.get('chunk_id', '?')}"


def sources_note(citations):
    """A note for the UI when an answer has no verified citations, else None (AD-5, story 2.3)."""
    return None if citations else NO_SOURCES_NOTE
