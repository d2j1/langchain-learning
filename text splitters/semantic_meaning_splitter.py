from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Text that jumps between two unrelated topics: cricket and cooking.
text = """Cricket is a bat-and-ball game played between two teams of eleven players. \
The batting side scores runs by hitting the ball and running between the wickets. \
The bowling side tries to get the batters out. \
To make a simple pasta sauce, heat olive oil in a pan and add chopped garlic. \
Add crushed tomatoes and let the sauce simmer for twenty minutes. \
Season with salt, pepper and fresh basil before serving. \
The Cricket World Cup is held every four years and is watched by millions of fans."""

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# Embeds each sentence, compares neighbouring sentences, and starts a new chunk
# wherever the meaning changes a lot (a big jump in embedding distance).
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,  # lower = more splits, higher = fewer splits
)

chunks = text_splitter.split_text(text)

print(len(chunks), "chunks")

for chunk in chunks:
    print(chunk)
    print("-" * 50)
