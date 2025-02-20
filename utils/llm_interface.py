from openai import OpenAI
import google.generativeai as genai
import time


def get_client(model_type, model_name):
    if model_type == "openai":
        client = OpenAI()
    elif model_type == "gemini":
        genai.configure(api_key = os.environ["GEMINI_API_KEY"])
        client = genai.GenerateModel(model_name=model_name)

    return client


def query_llm(client, model_name, conversation):
    for attempt in range(5):
        try:
            if "gemini" in model_name:
                client = genai.GenerativeModel(model_name=model_name)
                session = client.start_chat(history = conversation[:-1])
                response = session.send_message(
                    conversation[-1]["parts"]
                )
                return response.text
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=conversation
                )
                return completion.choices[0].message.content

        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            if attempt == 4:
                raise Exception("Failed")
            else:
                print("Waiting for 240 seconds before retrying...")
                time.sleep(240)

