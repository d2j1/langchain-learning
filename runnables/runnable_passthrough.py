from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()

# RunnablePassthrough returns its input unchanged.
print(RunnablePassthrough().invoke({"name": "DJ"}))
print("-" * 40)

joke_prompt = PromptTemplate(
    template="Write a short joke about {topic}.",
    input_variables=["topic"],
)

explain_prompt = PromptTemplate(
    template="Explain why the following joke is funny:\n\n{text}",
    input_variables=["text"],
)

# Step 1: generate the joke.
joke_chain = RunnableSequence(joke_prompt, chat_model, parser)

# Step 2: both branches receive the joke.
# - "joke" uses RunnablePassthrough, so it keeps the joke as it is.
# - "explanation" sends the joke to the model to explain it.
# Without the passthrough, only the explanation would be returned
# and the joke itself would be lost.
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(explain_prompt, chat_model, parser),
    }
)

chain = RunnableSequence(joke_chain, parallel_chain)

result = chain.invoke({"topic": "cricket"})

print("Joke:\n", result["joke"])
print()
print("Explanation:\n", result["explanation"])
