import streamlit as st
import sys
import os

# Add the parent directory to the path to import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Streamlit.backend import InformationRetrieval

def main():
    st.set_page_config(
        page_title="Oxylabs Documentation Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "urls" not in st.session_state:
        st.session_state.urls = []
    if "first_query_sent" not in st.session_state:
        st.session_state.first_query_sent = False
    
    # Show banner only when no query has been sent yet
    if not st.session_state.first_query_sent:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
            <h2>Oxylabs Documentation Assistant</h2>
            <p>Search and retrieve information from Oxylabs scraping documentation. Ask questions about web scraping APIs, proxy services, data collection methods, and implementation guides to get accurate answers with source references.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple sidebar for sources
    with st.sidebar:
        st.header("Documentation Sources")
        if st.session_state.urls:
            for i, url in enumerate(st.session_state.urls, 1):
                if url:
                    st.markdown(f"{i}. [{url}]({url})")
        else:
            st.markdown("*No sources yet*")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.urls = []
            st.session_state.first_query_sent = False
            st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.first_query_sent = True
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                ir = InformationRetrieval(prompt)
                response, urls = ir.orchestrate_retrieval()
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.urls = urls
                st.rerun()

if __name__ == "__main__":
    main()
