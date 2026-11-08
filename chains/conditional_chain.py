from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

str_parser = StrOutputParser()


# Step 1: classify the feedback.
# A Pydantic parser guarantees the result is exactly "positive" or "negative",
# which makes it safe to use in the conditions below.
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

classify_prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback as positive or negative.\n\n"
        "Feedback: {feedback}\n\n{format_instruction}"
    ),
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)

classifier_chain = classify_prompt | chat_model | pydantic_parser

# Step 2: a different reply prompt for each sentiment.
positive_prompt = PromptTemplate(
    template="Write a short, warm thank-you reply to this positive feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

negative_prompt = PromptTemplate(
    template="Write a short, apologetic reply to this negative feedback:\n\n{feedback}",
    input_variables=["feedback"],
)

# RunnableBranch checks each (condition, chain) pair in order and runs the
# FIRST chain whose condition is True. The last argument is the default,
# used when no condition matches.
branch_chain = RunnableBranch(
    (
        lambda x: x["classification"].sentiment == "positive",
        positive_prompt | chat_model | str_parser,
    ),
    (
        lambda x: x["classification"].sentiment == "negative",
        negative_prompt | chat_model | str_parser,
    ),
    RunnableLambda(lambda x: "Could not determine the sentiment."),
)

# The reply prompts need the original feedback text, but the classifier only
# returns the sentiment. RunnablePassthrough.assign keeps the original input
# and adds the classifier's result under a new key:
# {"feedback": "...", "classification": Feedback(sentiment="negative")}
chain = RunnablePassthrough.assign(classification=classifier_chain) | branch_chain

feedback = "This phone is terrible. The battery dies within a few hours."

result = chain.invoke({"feedback": feedback})

print(result)
