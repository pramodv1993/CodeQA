import streamlit as st
from ui_elements import render_data_ingestion_window, render_chat_window

st.title("""Code QA""")
st.subheader("""Talk to any repository of your choice!""")


def init():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

#for testing github hooks - v2
init()
render_data_ingestion_window()
render_chat_window()
