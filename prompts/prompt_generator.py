from pathlib import Path

from langchain_core.load import dumps
from langchain_core.prompts import PromptTemplate

TEMPLATE_PATH = Path(__file__).parent / "template.json"

# validate_template=True raises an error at creation if input_variables
# don't match the {placeholders} in the template.
prompt_template = PromptTemplate(
    template="Explain {topic} in {num_lines} simple lines for a {audience}.",
    input_variables=["topic", "num_lines", "audience"],
    validate_template=True,
)

TEMPLATE_PATH.write_text(dumps(prompt_template, pretty=True), encoding="utf-8")

print(f"Template saved to {TEMPLATE_PATH}")
