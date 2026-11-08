from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}.",
    input_variables=["topic"],
)

parser = StrOutputParser()

# A simple chain: prompt -> model -> parser.
# The | operator passes each step's output to the next step.
chain = prompt | chat_model | parser

result = chain.invoke({"topic": "cricket"})

print(result)

# Print a diagram of the chain's steps (requires: pip install grandalf).
try:
    chain.get_graph().print_ascii()
except ImportError:
    print("\nInstall grandalf to see the chain diagram: pip install grandalf")
