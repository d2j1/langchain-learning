from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# The template builds the full message list on every turn:
# system message -> past messages -> the new user message.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {role}. Keep answers short and clear."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ]
)

chain = chat_prompt | chat_model

# Only the conversation is stored here; the template adds the system message.
chat_history = []
role = "AI assistant"

print("Chatbot ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    response = chain.invoke(
        {"role": role, "chat_history": chat_history, "user_input": user_input}
    )

    # Save both sides of the turn so the next call includes them.
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.text))

    print("AI:", response.text)

print("\nChat history:")
for message in chat_history:
    print(f"{message.type}: {message.text}")
