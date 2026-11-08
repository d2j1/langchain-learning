from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)


# The schema: field names, types, rules and descriptions.
# Unlike StructuredOutputParser, the values are validated when parsed.
class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(gt=18, description="The person's age, must be over 18")
    city: str = Field(description="The city the person lives in")


parser = PydanticOutputParser(pydantic_object=Person)

# The format instructions include the full JSON schema of Person.
prompt = PromptTemplate(
    template=(
        "Generate the name, age and city of a fictional person "
        "from {country}.\n{format_instruction}"
    ),
    input_variables=["country"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# Print the final prompt to see the format instructions the parser added.
print(prompt.format(country="India"))
print("-" * 40)

chain = prompt | chat_model | parser

result = chain.invoke({"country": "sri lanka"})

print(type(result))
print(result)
print()

# The result is a Person object, so fields are read as attributes.
print("Name:", result.name)
print("Age:", result.age)
print("City:", result.city)
