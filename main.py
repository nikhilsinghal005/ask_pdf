import streamlit as st
import numpy as np
import faiss
from utils.streamlit_utils import save_uploaded_file
from utils.embeddings_utils import EmbeddingsUtils
from utils.pdf_utils import PDFUtils
from agent.create_agent_v1 import CustomQueryAgent
import os

if __name__ == "__main__":
    st.title("Upload PDF and Query")

    # Create a container for messages
    message_container = st.container()

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")  # File uploader widget

    if uploaded_file:
        with message_container:
            st.info("Processing the uploaded file...")

        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)
        with message_container:
            st.success("File uploaded")

        # Check if the file is already processed and stored in session state
        if 'file_path' not in st.session_state or st.session_state['file_path'] != file_path:
            with message_container:
                st.info("Extracting text from the PDF file...")

            # Process the new PDF file
            pdf_class = PDFUtils.from_pdf(file_path)
            extracted_text = pdf_class.text
            chunks = pdf_class.split_text(extracted_text)
            with message_container:
                st.success("File extracted successfully")

            # Create embeddings for each chunk
            embeddings_class = EmbeddingsUtils()
            if embeddings_class.validate_chunks(chunks):
                with message_container:
                    st.info("Creating embeddings for the text chunks...")
                
                embeddings = embeddings_class.create_embeddings(chunks)
                embeddings_np = np.array(embeddings).astype('float32')
                with message_container:
                    st.success("Embeddings created successfully")
            else:
                with message_container:
                    st.error("One or more chunks exceed the token limit.")
                st.stop()

            # Initialize FAISS index
            d = embeddings_np.shape[1]  # dimension of embeddings
            index = faiss.IndexFlatL2(d)  # L2 distance index
            index.add(embeddings_np)  # Add embeddings to the index
            with message_container:
                st.success("Embeddings stored successfully")

            # Store embeddings, text chunks, and file path in session state
            st.session_state['index'] = index
            st.session_state['text_chunks'] = chunks
            st.session_state['file_path'] = file_path
        else:
            with message_container:
                st.success("PDF already processed. Using stored embeddings and chunks.")

        # User input for query
        query = st.text_input("Enter your query (for multiple queries, separate with ';'):")

        if query:
            query_list = query.split(";")
            # Filter out empty queries after splitting
            query_list = [q.strip() for q in query_list if q.strip()]

            if query_list:
                for temp_query in query_list:
                    agent = CustomQueryAgent(st.session_state["index"], st.session_state["text_chunks"])
                    response = agent.query(temp_query)
                    with message_container:
                        st.write(f"Response for '{temp_query}': {response}")
            else:
                with message_container:
                    st.error("All queries are empty after processing.")
        else:
            with message_container:
                st.error("Empty Query")
