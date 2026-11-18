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
