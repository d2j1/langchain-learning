from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()

# Step 1: write a detailed report on the topic.
report_prompt = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"],
)

# Step 2: pick the 5 most important points from that report.
points_prompt = PromptTemplate(
    template="Generate the 5 most important points from the following text:\n\n{text}",
    input_variables=["text"],
)

# A sequential chain: the output of the first model call
# becomes the input of the second prompt.
chain = report_prompt | chat_model | parser | points_prompt | chat_model | parser

result = chain.invoke({"topic": "unemployment in India"})

print(result)

# Print a diagram of the chain's steps (requires: pip install grandalf).
try:
    chain.get_graph().print_ascii()
except ImportError:
    print("\nInstall grandalf to see the chain diagram: pip install grandalf")
