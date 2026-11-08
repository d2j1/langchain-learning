from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# The full history is sent on every turn, which is how the model remembers.
chat_history = [SystemMessage(content="You are a helpful assistant.")]

print("Chatbot ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    chat_history.append(HumanMessage(content=user_input))

    response = chat_model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.text))
    print("AI:", response.text)

print("\nChat history:")
for message in chat_history:
    print(f"{message.type}: {message.text}")
