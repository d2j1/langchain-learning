from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)


# The schema: the shape of the answer we want back.
# Unlike TypedDict, Pydantic checks the values at run time and raises
# a ValidationError if the model returns the wrong type.
class Review(BaseModel):
    summary: str = Field(description="A one-sentence summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment"
    )
    rating: int = Field(
        ge=1, le=5, description="The rating out of 5 implied by the review"
    )
    key_themes: list[str] = Field(
        description="The main topics discussed in the review"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="Good points mentioned, if any"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="Bad points mentioned, if any"
    )
    reviewer_name: Optional[str] = Field(
        default=None, description="The reviewer's name, if given"
    )


# The model now returns a Review object instead of free text.
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

# Fields are read as attributes, not dictionary keys.
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Rating:", result.rating)
print("Themes:", result.key_themes)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer:", result.reviewer_name)

# Convert to a dict or JSON when needed.
print()
print(result.model_dump())
print(result.model_dump_json(indent=2))
