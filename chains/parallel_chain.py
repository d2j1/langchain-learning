from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Two models: parallel branches can use different models.
# The lighter model is fine for the simpler quiz task.
notes_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)
quiz_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", max_retries=3)

parser = StrOutputParser()

# Branch 1: short notes from the text.
notes_prompt = PromptTemplate(
    template="Generate short and simple notes from the following text:\n\n{text}",
    input_variables=["text"],
)

# Branch 2: a quiz from the same text.
quiz_prompt = PromptTemplate(
    template="Generate 5 short question-and-answer pairs from the following text:\n\n{text}",
    input_variables=["text"],
)

# Final step: combine the outputs of both branches.
merge_prompt = PromptTemplate(
    template=(
        "Merge the provided notes and quiz into a single document.\n\n"
        "Notes:\n{notes}\n\n"
        "Quiz:\n{quiz}"
    ),
    input_variables=["notes", "quiz"],
)

# RunnableParallel runs both branches at the same time on the SAME input.
# It returns a dict, using the names given here as the keys:
# {"notes": "...", "quiz": "..."}
parallel_chain = RunnableParallel(
    {
        "notes": notes_prompt | notes_model | parser,
        "quiz": quiz_prompt | quiz_model | parser,
    }
)

# The dict from parallel_chain matches merge_prompt's variables,
# so it can be piped straight into it.
merge_chain = merge_prompt | notes_model | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used
for classification, regression and outlier detection.

The advantages of support vector machines are:
- Effective in high-dimensional spaces.
- Still effective when the number of dimensions is greater than the number
  of samples.
- Memory efficient, because they use only a subset of the training points
  (called support vectors) in the decision function.
- Versatile: different kernel functions can be specified for the decision
  function.

The disadvantages of support vector machines include:
- If the number of features is much greater than the number of samples,
  choosing the kernel function and regularization term is crucial to avoid
  overfitting.
- SVMs do not directly provide probability estimates; these are calculated
  using an expensive five-fold cross-validation.
"""

result = chain.invoke({"text": text})

print(result)

# Print a diagram of the chain's steps (requires: pip install grandalf).
try:
    chain.get_graph().print_ascii()
except ImportError:
    print("\nInstall grandalf to see the chain diagram: pip install grandalf")
