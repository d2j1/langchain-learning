from langchain_community.document_loaders import CSVLoader
from pathlib import Path

file_path = Path(__file__).parent / "cricket_players.csv"

loader = CSVLoader(file_path=str(file_path), encoding="utf-8")

docs = loader.load()

# One Document per row of the CSV.
print(len(docs))

for doc in docs:
    print(doc.metadata)
    print(doc.page_content)
    print("-" * 50)
