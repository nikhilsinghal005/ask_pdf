import os

UPLOAD_DIR = "uploads"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to the designated directory.

    Args:
        uploaded_file: The file uploaded by the user.
    
    Returns:
        file_path: The path where the file is saved.
    """

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


