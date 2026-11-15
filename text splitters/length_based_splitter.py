from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# cricket.pdf lives in the "document loader" folder, next to this "chains" folder.
file_path = Path(__file__).parent.parent / "document loader" / "cricket.pdf"

loader = PyPDFLoader(str(file_path))
docs = loader.load()  # one Document per PDF page

text_splitter = CharacterTextSplitter(
    chunk_size=100,    # max characters per chunk
    chunk_overlap=0,   # characters shared between neighbouring chunks
    separator="",      # "" = split purely by length, not at spaces or newlines
)

# split_documents keeps each page's metadata (source, page number) on its chunks.
chunks = text_splitter.split_documents(docs)

print(len(docs), "pages ->", len(chunks), "chunks")

for chunk in chunks:
    print(chunk.metadata)
    print(chunk.page_content)
    print("-" * 50)
