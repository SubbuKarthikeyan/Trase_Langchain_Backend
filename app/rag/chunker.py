"""
chunker.py
───────────
Splits LangChain Documents into smaller overlapping chunks.

Uses RecursiveCharacterTextSplitter, which respects paragraph/sentence/word
boundaries — much better than the old fixed-character slicer.

Input:  list[Document]
Output: list[Document]  (each with inherited metadata + chunk index)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Split a list of LangChain Documents into overlapping text chunks.

    Args:
        documents:     list of LangChain Document objects from loader.py
        chunk_size:    max characters per chunk (default 500)
        chunk_overlap: overlapping characters between adjacent chunks (default 100)

    Returns:
        list of LangChain Document objects ready for embedding and storage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)