from dotenv import load_dotenv
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# Each ResponseSchema is one field we want back: a name and a description.
# This fixes the keys of the result, which JsonOutputParser alone can't do.
response_schemas = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# The format instructions list the exact keys the model must use.
prompt = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# Print the final prompt to see the format instructions the parser added.
print(prompt.format(topic="black holes"))
print("-" * 40)

chain = prompt | chat_model | parser

result = chain.invoke({"topic": "black holes"})

print(type(result))
print(result)
print()
print("Fact 1:", result["fact_1"])
print("Fact 2:", result["fact_2"])
print("Fact 3:", result["fact_3"])
