from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# StrOutputParser turns the model's AIMessage into a plain string.
parser = StrOutputParser()

# Prompt 1: write a detailed report on a topic.
report_prompt = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"],
)

# Prompt 2: summarise the report produced by prompt 1.
summary_prompt = PromptTemplate(
    template="Write a 5-line summary of the following text:\n\n{text}",
    input_variables=["text"],
)

# Without the parser, the model's AIMessage would be passed to summary_prompt,
# and the {text} variable would get the whole message object, not just the text.
# The parser hands the next step a clean string instead.
chain = report_prompt | chat_model | parser | summary_prompt | chat_model | parser

result = chain.invoke({"topic": "black holes"})

print(type(result))
print(result)
