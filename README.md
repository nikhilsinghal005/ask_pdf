
# PDF Query Application

This application allows users to upload PDF files, extract text, and query the content using embeddings and a custom query agent. The app is built using Streamlit and integrates with OpenAI's embedding and language models.

## Features

- **PDF Upload & Text Extraction**: Users can upload a PDF, which is then processed to extract the text.
- **Text Chunking & Embedding Creation**: The extracted text is split into smaller chunks, and embeddings are created using OpenAI's API.
- **Query Handling**: Users can input queries, and the application will search the text chunks to provide relevant answers.
- **Slack Integration**: Results can be posted to a specified Slack channel.

## Project Structure

- `main.py`: The main entry point for the Streamlit application. Handles file upload, text extraction, embedding creation, and query processing.
- `utils/`: Contains utility modules for handling different functionalities.
 - `embeddings_utils.py`: Utilities for creating and validating text embeddings using OpenAI's API.
 - `pdf_utils.py`: Utilities for extracting text from PDFs and splitting it into manageable chunks.
 - `streamlit_utils.py`: Utilities for handling Streamlit-specific tasks, such as saving uploaded files.
- `agent/create_agent_v1.py`: Contains the `CustomQueryAgent` class, responsible for handling queries and integrating with Slack.
- `requirements.txt`: Lists the Python dependencies required to run the application.
- `Dockerfile`: Contains the instructions to create a Docker image for the application.

## Setup

### Local Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repo-name/pdf-query-app.git
   cd pdf-query-app

2.  **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    
3.  **Set up environment variables**:
    
    -   Create a `.env` file in the root directory.
        
    -   Add your OpenAI API key and Slack API token:
        
        ```bash
        OPENAI_KEY_PROJECT=your_openai_api_key
        SLACK_BOT_TOKEN=your_slack_bot_token
        
4.  **Run the application**:
    ```bash 
    streamlit run main.py
    

### Docker Setup

1.  **Build the Docker image**:
    
    ```bash
    docker build -t ask-pdf-test .
    
2.  **Run the Docker container**:
    ```bash
    docker run -p 8501:8501 ask-pdf-test
    

### Dockerfile Explanation

-   **FROM python:3.11-slim**: Uses a slim version of Python 3.11 as the base image.
-   **WORKDIR /app**: Sets the working directory inside the container to `/app`.
-   **COPY requirements.txt .**: Copies the `requirements.txt` file into the container.
-   **RUN pip install --no-cache-dir -r requirements.txt**: Installs the required Python packages listed in `requirements.txt` without using the cache.
-   **COPY . .**: Copies all the files from the current directory into the container.
-   **EXPOSE 8501**: Exposes port 8501 to allow access to the Streamlit application.
-   **CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]**: Runs the Streamlit application when the container starts, listening on port 8501.

## Usage

1.  Upload a PDF file using the file uploader.
2.  After successful extraction and embedding creation, enter your query in the provided input box.
3.  The application will search through the extracted text and provide relevant answers.
4.  Optionally, results can be posted to a specified Slack channel.

## Dependencies

-   Python 3.11+
-   Streamlit 1.37.0
-   PDFPlumber 0.11.4
-   OpenAI Python Client 1.42.0
-   Python-Dotenv 0.19.1
-   Faiss-CPU 1.8.0
-   Pandas 2.2.2
-   Numpy 1.26.4
-   Langchain-Community 0.0.38
-   Langchain 0.1.19