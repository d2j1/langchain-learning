from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)


# The schema: the shape of the answer we want back.
# Annotated[type, "description"] tells the model what each field means.
class Review(TypedDict):
    summary: Annotated[str, "A one-sentence summary of the review"]
    sentiment: Annotated[
        Literal["positive", "negative", "neutral"], "The overall sentiment"
    ]
    key_themes: Annotated[list[str], "The main topics discussed in the review"]
    pros: Annotated[Optional[list[str]], "Good points mentioned, if any"]
    cons: Annotated[Optional[list[str]], "Bad points mentioned, if any"]
    reviewer_name: Annotated[Optional[str], "The reviewer's name, if given"]


# The model now returns a dict that follows the Review schema,
# instead of a message containing free text.
structured_model = chat_model.with_structured_output(Review)

review_text = """
I bought this phone two weeks ago. The battery easily lasts a full day
and the camera takes sharp photos even at night. However, it heats up
while gaming and the charger is not included in the box, which is
disappointing at this price. Overall it's a good phone, but not perfect.
Reviewed by Rahul Sharma
"""

result = structured_model.invoke(review_text)

print(type(result))
print(result)
print()
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
print("Themes:", result["key_themes"])
print("Pros:", result.get("pros"))
print("Cons:", result.get("cons"))
print("Reviewer:", result.get("reviewer_name"))
