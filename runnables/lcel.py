from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()

joke_prompt = PromptTemplate(
    template="Write a short joke about {topic}.",
    input_variables=["topic"],
)

explain_prompt = PromptTemplate(
    template="Explain why the following joke is funny:\n\n{text}",
    input_variables=["text"],
)


def word_count(text: str) -> int:
    return len(text.split())


# LCEL (LangChain Expression Language) builds the same Runnables
# with the | operator instead of writing the class names out.

# 1. RunnableSequence -> a | b | c
#    Before: RunnableSequence(joke_prompt, chat_model, parser)
joke_chain = joke_prompt | chat_model | parser

# 2. RunnableParallel -> a plain dict
#    Before: RunnableParallel({"joke": ..., "explanation": ..., "word_count": ...})
#    A dict joined with | is turned into a RunnableParallel automatically.
#
# 3. RunnableLambda -> a plain function
#    Before: RunnableLambda(word_count)
#    A function inside a chain is turned into a RunnableLambda automatically.
chain = joke_chain | {
    "joke": RunnablePassthrough(),
    "explanation": explain_prompt | chat_model | parser,
    "word_count": word_count,
}

result = chain.invoke({"topic": "programmers"})

print("Joke:\n", result["joke"])
print()
print("Explanation:\n", result["explanation"])
print()
print("Word count:", result["word_count"])

# Print a diagram of the chain's steps (requires: pip install grandalf).
try:
    chain.get_graph().print_ascii()
except ImportError:
    print("\nInstall grandalf to see the chain diagram: pip install grandalf")
