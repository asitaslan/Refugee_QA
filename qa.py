import streamlit as st
#from ST_Model_QA import PDFQAWithQdrant
from OpenAI_Model_QA import PDFQAWithQdrant

def question_page():
    
    if "qa_system" not in st.session_state:
        st.warning("Please upload and process a PDF on the 'Upload PDF' page first.")
    else:
        
        query_text = st.text_input("Ask a question about the content:")

        if query_text:
            with st.spinner("Generating answer..."):
                response = st.session_state.qa_system.answer_question(query_text)
                st.success(response)