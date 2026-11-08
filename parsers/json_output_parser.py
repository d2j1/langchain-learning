from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# JsonOutputParser reads the model's reply as JSON and returns a Python dict.
parser = JsonOutputParser()

# get_format_instructions() returns text telling the model to reply in JSON.
# It is put into the prompt through partial_variables, so it's filled in
# automatically and doesn't need to be passed to invoke().
prompt = PromptTemplate(
    template=(
        "Give the name, age and city of a fictional person from {country}.\n"
        "{format_instruction}"
    ),
    input_variables=["country"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# Print the final prompt to see the format instructions the parser added.
print(prompt.format(country="India"))
print("-" * 40)

chain = prompt | chat_model | parser

result = chain.invoke({"country": "India"})

print(type(result))
print(result)
