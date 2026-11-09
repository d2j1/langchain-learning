from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

parser = StrOutputParser()

tweet_prompt = PromptTemplate(
    template="Write a short tweet about {topic}.",
    input_variables=["topic"],
)

linkedin_prompt = PromptTemplate(
    template="Write a short LinkedIn post about {topic}.",
    input_variables=["topic"],
)

# RunnableParallel runs every branch at the same time on the SAME input.
# It returns a dict whose keys are the branch names given here.
parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(tweet_prompt, chat_model, parser),
        "linkedin": RunnableSequence(linkedin_prompt, chat_model, parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})

print(type(result))
print()
print("Tweet:\n", result["tweet"])
print()
print("LinkedIn post:\n", result["linkedin"])
