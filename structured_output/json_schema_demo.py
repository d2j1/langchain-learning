from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# The schema is a plain Python dict in JSON Schema format.
# "title" and "description" at the top level are required by LangChain.
# Unlike TypedDict and Pydantic, this needs no Python classes, so the same
# schema can be shared with code written in other languages.
review_schema = {
    "title": "Review",
    "description": "Details extracted from a product review",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A one-sentence summary of the review",
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "The overall sentiment",
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "The rating out of 5 implied by the review",
        },
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The main topics discussed in the review",
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Good points mentioned, if any",
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Bad points mentioned, if any",
        },
        "reviewer_name": {
            "type": ["string", "null"],
            "description": "The reviewer's name, if given",
        },
    },
    # Fields listed here must be present; the rest are optional.
    "required": ["summary", "sentiment", "rating", "key_themes"],
}

# The model now returns a dict that follows the schema.
structured_model = chat_model.with_structured_output(review_schema)

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

# The result is a dict, so optional fields are read with .get().
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
print("Rating:", result["rating"])
print("Themes:", result["key_themes"])
print("Pros:", result.get("pros"))
print("Cons:", result.get("cons"))
print("Reviewer:", result.get("reviewer_name"))
