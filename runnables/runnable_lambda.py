from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()


# A plain Python function: it knows nothing about LangChain.
def word_count(text: str) -> int:
    return len(text.split())


# RunnableLambda wraps the function so it becomes a Runnable,
# with invoke, batch and stream like every other component.

joke_prompt = PromptTemplate(
    template="Write a short joke about {topic}.",
    input_variables=["topic"],
)

# Step 1: generate the joke.
joke_chain = RunnableSequence(joke_prompt, chat_model, parser)

# Step 2: keep the joke and count its words at the same time.
# The word count is ordinary Python code, so no model call is made for it.
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count),
    }
)

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({"topic": "AI"})

print("Joke:\n", result["joke"])
print()
print("Word count:", result["word_count"])
