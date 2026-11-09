from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()

WORD_LIMIT = 300

report_prompt = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"],
)

summary_prompt = PromptTemplate(
    template="Summarise the following text in under 100 words:\n\n{text}",
    input_variables=["text"],
)

# Step 1: generate the report.
report_chain = RunnableSequence(report_prompt, chat_model, parser)

# Step 2: decide what to do with the report.
# RunnableBranch takes (condition, runnable) pairs and runs the FIRST one
# whose condition is True. The last argument is the default.
# - Report longer than WORD_LIMIT words: summarise it (one more model call).
# - Otherwise: return it unchanged with RunnablePassthrough (no model call).
branch_chain = RunnableBranch(
    (
        lambda text: len(text.split()) > WORD_LIMIT,
        RunnableSequence(summary_prompt, chat_model, parser),
    ),
    RunnablePassthrough(),
)

chain = RunnableSequence(report_chain, branch_chain)

result = chain.invoke({"topic": "Russia vs Ukraine"})

print(result)
print()
print("Final word count:", len(result.split()))
