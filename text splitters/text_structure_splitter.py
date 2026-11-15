from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# cricket.pdf lives in the "document loader" folder, next to this "chains" folder.
file_path = Path(__file__).parent.parent / "document loader" / "cricket.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()  # one Document per PDF page

# Tries to split on paragraphs first, then lines, then words, then characters,
# so chunks end at natural boundaries instead of in the middle of a word.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,     # max characters per chunk
    chunk_overlap=10,   # characters shared between neighbouring chunks
)

# split_documents keeps each page's metadata (source, page number) on its chunks.
chunks = text_splitter.split_documents(docs)

print(len(docs), "pages ->", len(chunks), "chunks")

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print("-" * 50)
