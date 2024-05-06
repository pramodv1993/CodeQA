import re
import requests

import streamlit as st


def _is_valid_url():
    repo_url = st.session_state.repo_url
    if not re.match(
        r"https?:\/\/(?:www\.)?github\.com\/[a-zA-Z0-9-]+\/[a-zA-Z0-9-]+(?:\/)?(?:\w+)?(?:\/)?(?:\w+)?",
        repo_url,
    ):
        st.write("Please enter valid repository url")
        return False
    return True


def render_data_ingestion_window():
    with st.form("my_form"):
        github_url = st.text_input(
            "Enter url of the repository you want to talk to",
            placeholder="https://github.com/zpqrtbnk/test-repo.git",
            key="repo_url",
        )
        submitted = st.form_submit_button("Upload Repository")
        if submitted and _is_valid_url():
            response = None
            with st.status("Ingesting data.."):
                response = requests.post(
                    "http://api:8001/ingest",
                    json={"repo_url": github_url, "insert_custom_embeddings": False},
                )
            if response and response.status_code == 201:
                st.markdown(
                    """Ingested to the VectorDB, you can check [Here](http://localhost:6333/dashboard#/collections)"""
                )
            else:
                st.markdown("""Encountered some error""")


# ____________CHAT___________


def _clear_chat():
    st.session_state["messages"] = []


def _update_msg_history(role, msg):
    st.session_state.messages.append((role, msg))


def _display_msg(role, msg, update_history: bool = True, stream_resp: bool = False):
    if stream_resp:
        with st.chat_message(role):
            st.write_stream(msg)
    else:
        with st.chat_message(role):
            print(msg)
            if "Referred" in msg:
                resp, references = (
                    msg[: msg.index("Referred")],
                    msg[msg.index("Referred") :],
                )
                resp = resp.rstrip()
                references = references.replace("\n", "     \n")
                print("Here", references)
                st.markdown(resp.rstrip())
                st.markdown("  ".join(references.split("\n")))
            else:
                st.markdown(msg.rstrip())
    if update_history:
        _update_msg_history(role, msg)


def _get_answer(question: int) -> str:
    response = requests.post(
        "http://api:8001/generate",
        json={"repo_url": st.session_state.repo_url, "query": question},
    )
    if response and response.status_code == 200:
        return response.content.decode("utf-8")
    return ""


def _display_chat_history():
    if st.session_state.messages:
        for role, msg in st.session_state.messages:
            _display_msg(role, msg, update_history=False)


def render_chat_window():
    st.subheader("Chat")
    _display_chat_history()
    if question := st.chat_input("Ask a question.."):
        _display_msg("user", question, update_history=True, stream_resp=False)
        answer = ""
        with st.spinner("Retrieving answers.."):
            answer = _get_answer(question)
        answer = (
            answer
            if len(answer)
            else "Sorry, Could not find the answer, Please try again."
        )
        _display_msg("assistant", answer, update_history=True, stream_resp=False)
    st.button("Clear Chat", key="clear", on_click=_clear_chat)
