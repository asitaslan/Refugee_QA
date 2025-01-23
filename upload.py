import streamlit as st
#from ST_Model_QA import PDFQAWithQdrant
from OpenAI_Model_QA import PDFQAWithQdrant

def upload_page():
    upload_file = st.file_uploader("Upload a PDF document", type='pdf')

    if upload_file is not None:
        if st.button("Process PDF"):
            st.session_state.qa_system = PDFQAWithQdrant(upload_file)
            st.success("PDF processed successfully! Go to the 'Ask Questions' page to start querying.")
    else:
        st.info("Please upload a PDF file to begin.")