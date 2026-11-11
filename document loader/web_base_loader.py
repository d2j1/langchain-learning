from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://en.wikipedia.org/wiki/Cricket",
    "https://en.wikipedia.org/wiki/Test_cricket",
    "https://en.wikipedia.org/wiki/Cricket_World_Cup",
]

loader = WebBaseLoader(urls)

docs = loader.load()

print(len(docs))

# One Document per URL.
for doc in docs:
    print(doc.metadata["source"])
    print(doc.page_content[:500])
    print("-" * 50)
