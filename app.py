import streamlit as st
from upload import upload_page
from qa import question_page


st.set_page_config(page_title="PDF QA", layout="centered")
st.sidebar.title("Navigation")
st.sidebar.write("Select a page to proceed:")

if st.sidebar.button("Upload PDF"):
    st.session_state.current_page = "upload"
elif st.sidebar.button("Ask Questions"):
    st.session_state.current_page = "question"

if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"

if st.session_state.current_page == "upload":
    upload_page()
elif st.session_state.current_page == "question":
    question_page()
