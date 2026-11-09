from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
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

# RunnableSequence runs its steps one after another.
# Each step's output becomes the next step's input.
chain = RunnableSequence(
    joke_prompt, chat_model, parser, explain_prompt, chat_model, parser
)

result = chain.invoke({"topic": "AI"})

print(result)

# The | operator builds exactly the same RunnableSequence.
pipe_chain = joke_prompt | chat_model | parser | explain_prompt | chat_model | parser

print()
print("Type of chain:", type(chain).__name__)
print("Type of pipe_chain:", type(pipe_chain).__name__)
