import os
import tempfile

import streamlit as st

from rag import (
    load_multiple_pdfs,
    chunk_pages,
    build_vector_index,
    ask,
)


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="CampusGPT",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-top: 0;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f1f8f4;
        border: 1px solid #b7dfc5;
        margin-bottom: 15px;
    }

    .source-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f7f7f7;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "document_stats" not in st.session_state:
    st.session_state.document_stats = {}

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📚 Documents")

    st.write("Upload your study material")

    uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)

if uploaded_files:
    st.write("### Selected Documents")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")
    
    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
    )

    if process_button:
     print("PROCESS BUTTON CLICKED")
     if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:

        with st.spinner("Processing documents..."):

            temp_paths = []
            temp_file_names = {}
            try:

                for uploaded_file in uploaded_files:

                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    )

                    temp_file.write(uploaded_file.getbuffer())
                    temp_file.close()

                    temp_paths.append(temp_file.name)
                    temp_file_names[os.path.basename(temp_file.name)] = uploaded_file.name

                # Extract text from all PDFs
                pages = load_multiple_pdfs(temp_paths)

                
                # Replace temporary PDF names with original names
                for page in pages:
                 if "source" in page:
                  old_name = os.path.basename(str(page["source"]))
                 page["source"] = temp_file_names.get(
                      old_name, old_name
                )

                
                if not temp_paths:
                 st.error("No PDF file was uploaded.")
                 st.stop()

                print("STREAMLIT PDF PATHS:", temp_paths)
                print("STREAMLIT PAGES:", len(pages))

                # Create chunks from all documents
                chunks = chunk_pages(pages)

                print("STREAMLIT CHUNKS:", len(chunks))

                # Build one FAISS index containing all documents
                index = build_vector_index(chunks)
                
    
                # Save knowledge base
                st.session_state.index = index
                st.session_state.chunks = chunks

                # Save document names
                st.session_state.processed_files = [
                    file.name for file in uploaded_files
                ]

                # Document statistics
                document_stats = {}

                for file in uploaded_files:
                    document_stats[file.name] = {
                        "size_kb": round(
                            file.size / 1024,
                            1
                        )
                    }

                st.session_state.document_stats = document_stats

                # Start a fresh conversation
                st.session_state.chat_history = []

                st.success(
                    f"{len(uploaded_files)} document(s) processed successfully!"
                )

            except Exception as e:
             import traceback

             print("PROCESSING ERROR:", repr(e))
             traceback.print_exc()

             st.error(f"Processing failed: {e}")
             st.exception(e)

            finally:

                for path in temp_paths:

                    if os.path.exists(path):
                        os.remove(path)

    # -----------------------------------------------------
    # KNOWLEDGE BASE STATUS
    # -----------------------------------------------------

    st.divider()

    if st.session_state.index is not None:

     st.success("🟢 Knowledge base ready")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documents",
            len(st.session_state.processed_files)
        )

    with col2:
        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    st.write("### 📚 Documents")

    for file_name in st.session_state.processed_files:

        st.write(f"📄 **{file_name}**")

        if file_name in st.session_state.document_stats:

            size = st.session_state.document_stats[
                file_name
            ]["size_kb"]

            st.caption(f"{size} KB")

    else:

     st.info(
        "🔵 Upload and process PDF documents to begin."
    )

    # -----------------------------------------------------
    
# CLEAR CHAT
if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()


    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 CampusGPT</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI Learning Assistant powered by Retrieval-Augmented Generation (RAG)</div>',
    unsafe_allow_html=True,
)


st.write(
    "Upload your study material and ask questions. "
    "CampusGPT retrieves relevant information from your documents "
    "before generating an answer."
)


# ---------------------------------------------------------
# KNOWLEDGE BASE INFORMATION
# ---------------------------------------------------------

if st.session_state.index is not None:

    st.markdown(
        """
        <div class="status-box">
        🟢 <b>Knowledge Base Ready</b><br>
        Your uploaded documents have been processed successfully.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 Sources used"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        📄 <b>{source['source']}</b><br>
                        📖 Page: {source['page']}<br>
                        🔎 Similarity: {source['score']:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask something about your uploaded documents..."
)


if question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # CHECK KNOWLEDGE BASE
    # -----------------------------------------------------

    if st.session_state.index is None:

        answer = "Please upload and process a PDF before asking questions."

        retrieved = []

    else:

        with st.chat_message("assistant"):

            with st.spinner("Searching your documents..."):

                try:

                    answer, retrieved = ask(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.chat_history,
                    )

                except Exception as e:

                    answer = (
                        "An error occurred while processing your question."
                    )

                    retrieved = []

                    st.error(str(e))


            st.markdown(answer)

            # -------------------------------------------------
            # SOURCES
            # -------------------------------------------------

            if retrieved:

                with st.expander("📚 Sources used"):

                    for source in retrieved:

                        st.markdown(
                            f"""
                            **📄 {source['source']}**  
                            **📖 Page:** {source['page']}  
                            **🔎 Similarity:** {source['score']:.2f}
                            """
                        )

    # ---------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # ---------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        }
    )