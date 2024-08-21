
import os
from openai import OpenAI
from dotenv import load_dotenv

class EmbeddingsUtils:
    TOKEN_LIMIT = 8191  # Token limit for the text-embedding-3-large model
    EMBEDDING_MODEL = "text-embedding-3-large"  # Recommended model

    def __init__(self, api_key=None):
        """
        Initializes the EmbeddingsUtils class, loading the API key from the environment if not provided.

        Args:
            api_key (str): The OpenAI API key. If not provided, it is loaded from the environment variables.
        """
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_KEY_PROJECT")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in the environment variables.")
        self.client = OpenAI(api_key=self.api_key)

    def create_embeddings(self, text_chunks):
        """
        Create embeddings for a list of text chunks using OpenAI's API.

        Args:
            text_chunks (list of str): List of text chunks to create embeddings for.

        Returns:
            list of list of float: A list of embeddings corresponding to the input text chunks.
        """
        embeddings = []
        try:
            response = self.client.embeddings.create(
                input=text_chunks,
                model=self.EMBEDDING_MODEL
            )
            embeddings = [data.embedding for data in response.data]
        except Exception as e:
            print(f"An error occurred while creating embeddings: {str(e)}")
        return embeddings

    def validate_chunks(self, text_chunks):
        """
        Validate that the text chunks are within the token limit.

        Args:
            text_chunks (list of str): List of text chunks to validate.

        Returns:
            bool: True if all chunks are within the token limit, False otherwise.
        """
        return all(len(chunk) <= self.TOKEN_LIMIT for chunk in text_chunks)

