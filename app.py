import streamlit as st
from pypdf import PdfReader
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import (
    get_embeddings_model,
    get_or_create_vector_store,
    similarity_search,
    clear_cache,
)


# =============================================================================
# BUSINESS LOGIC
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings_model():
    return get_embeddings_model()


@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key,
    )


def extract_pdf_text(pdf_file) -> str:
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Chat with PDF",
        layout="centered",
    )

    # -------------------------------------------------------------------------
    # SESSION STATE (STRICT INITIALIZATION)
    # -------------------------------------------------------------------------
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------
    st.title("Chat with PDF")
    st.caption("Ask questions grounded in your document")
    st.divider()

    # -------------------------------------------------------------------------
    # SIDEBAR — CONFIGURATION
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.subheader("Configuration")

        api_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Required to generate answers",
        )

        st.divider()
        st.subheader("Document")

        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file:
            st.caption(pdf_file.name)

            if st.button("Process document", use_container_width=True):
                with st.spinner("Processing document…"):
                    text = extract_pdf_text(pdf_file)

                    if not text.strip():
                        st.error("No readable text found in the document.")
                        st.stop()

                    embeddings = load_embeddings_model()

                    st.session_state.vector_store = get_or_create_vector_store(
                        pdf_file,
                        text,
                        embeddings,
                    )

                    st.session_state.current_pdf = pdf_file.name
                    st.session_state.conversation.clear()

                    st.success("Document ready")
                    st.rerun()

        st.divider()

        if st.button("Reset session", use_container_width=True):
            clear_cache()
            st.session_state.clear()
            st.rerun()

    # -------------------------------------------------------------------------
    # READINESS CHECK
    # -------------------------------------------------------------------------
    ready = bool(api_key.strip()) and st.session_state.vector_store is not None

    if not api_key.strip():
        st.info("Enter your API key to begin.")
    elif st.session_state.vector_store is None:
        st.info("Upload and process a PDF to begin.")

    # -------------------------------------------------------------------------
    # CHAT HISTORY (WRAPPED / COLLAPSIBLE)
    # -------------------------------------------------------------------------
    conversation = st.session_state.conversation

    if conversation:
        if len(conversation) > 2:
            history = conversation[:-2]
            latest = conversation[-2:]
        else:
            history = []
            latest = conversation

        if history:
            with st.expander("Previous conversations", expanded=False):
                for role, content, timestamp in history:
                    with st.chat_message(role):
                        st.markdown(content)
                        st.caption(timestamp)

        for role, content, timestamp in latest:
            with st.chat_message(role):
                st.markdown(content)
                st.caption(timestamp)

    # -------------------------------------------------------------------------
    # CHAT INPUT
    # -------------------------------------------------------------------------
    prompt = st.chat_input(
        "Ask a question about the document",
        disabled=not ready,
    )

    if prompt and ready:
        with st.spinner("Generating answer…"):
            docs = similarity_search(
                st.session_state.vector_store,
                prompt,
                k=4,
            )

            context = "\n\n".join(d.page_content for d in docs)
            llm = load_llm(api_key)

            answer = llm.invoke(
                f"Answer the question using the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{prompt}"
            ).content

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

            st.session_state.conversation.append(
                ("user", prompt, timestamp)
            )
            st.session_state.conversation.append(
                ("assistant", answer, timestamp)
            )

            st.rerun()


if __name__ == "__main__":
    main()
