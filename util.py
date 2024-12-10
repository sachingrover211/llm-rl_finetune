import time

LLM_MODEL = "o1-preview"

def query_llm(client, conversation):

    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=conversation
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            if attempt == 4:
                raise Exception("Failed")
            else:
                print("Waiting for 120 seconds before retrying...")
                time.sleep(120)
