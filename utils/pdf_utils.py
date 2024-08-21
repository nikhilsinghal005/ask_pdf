import pdfplumber
import os

class PDFUtils:
    def __init__(self, pdf_path=None):
        """
        Initializes the PDFUtils class with an optional PDF path.

        Args:
            pdf_path (str): Path to the PDF file (optional).
        """
        self.pdf_path = pdf_path

    @classmethod
    def from_pdf(cls, pdf_path):
        """
        Class method to create an instance of PDFUtils and automatically extract text.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            PDFUtils: Instance of PDFUtils with text extracted from the PDF.
        """
        instance = cls(pdf_path)
        instance.text = instance.extract_text_from_pdf()
        return instance

    def extract_text_from_pdf(self, pdf_path=None):
        """
        Extracts text from a PDF file using pdfplumber.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        path = pdf_path or self.pdf_path
        if not path:
            raise ValueError("PDF path must be provided.")

        text = ""
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            text = f"An error occurred while reading the PDF: {str(e)}"
        
        return text

    @staticmethod
    def split_text(text, max_chunk_size=1000):
        """
        Splits the text into smaller chunks that can be processed by the embedding model.

        Args:
            text (str): The complete text to split.
            max_chunk_size (int): The maximum size of each chunk.

        Returns:
            list of str: List of text chunks.
        """
        if not text:
            raise ValueError("Text must be provided.")

        paragraphs = text.split('\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
