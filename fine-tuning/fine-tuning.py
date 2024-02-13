import os
from colorama import Fore
from openai import OpenAI

client = OpenAI()
api_key = os.environ.get("OPENAI_API_KEY")

# Function to interact with the chatbot model
def chat_with_model(user_input, conversation, chatbot):
    conversation.append({"role": "user", "content": user_input})
    messages_input = conversation.copy()
    messages_input.insert(0, {"role": "system", "content": chatbot})

    completion = client.chat.completions.create(
        model='ft:gpt-3.5-turbo-0613:personal::8Vn3Iyn5',
        messages=messages_input,
        temperature=0.7
        )

    chat_response = completion.choices[0].message.content

    conversation.append({"role": "assistant", "content": chat_response})

    # Print the chatbot's response in color
    print_colored("ChatBot1", chat_response)

    return chat_response

# Function to print text in color
def print_colored(agent, text):
    agent_colors = {
        "User": Fore.GREEN,
        "ChatBot1": Fore.CYAN,
    }
    colored_text = agent_colors.get(agent, Fore.RESET) + text + Fore.RESET
    print(colored_text)

conversation = []
chatbot_prompt = "You are a helpful assistant."

# Start the chat loop
while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ("exit", "quit", "stop"):
        print("ChatBot1: Goodbye!")
        break

    chat_response = chat_with_model(user_input, conversation, chatbot_prompt)
