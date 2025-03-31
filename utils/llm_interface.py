from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI
import google.generativeai as genai
import time, torch


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


def get_local_client(model_path, base_model, model_type):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if model_type == "HF":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.float16,
            device_map = "auto",
        )
    elif model_type == "OFFLINE":
        _model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
        )
        _model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            model=_model,
            model_id = model_path,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
        )

    return model, tokenizer


def query_local_llm(_model, _tokenizer, conversations):
    prompt = "\n".join(entry['content'] for entry in conversations)
    print("############### Prompt sent")
    print(prompt)
    inputs = _tokenizer(prompt, return_tensors = "pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens = 128)

    generated_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #generated_text.replace(prompt, "")
    prompt_text_arr = prompt.split("\n")
    next_1 = False
    next_2 = False
    for _p in prompt_text_arr:
        _p = _p.strip()
        count = 1
        if _p != "":
        #    if next_1 or next_2:
        #        if not next_1:
        #            next_2 = False
        #        else:
        #            next_1 = False
        #        count = 5
        #    if _p == "Weights:":
        #        next_1 = True
        #        next_2 = True
        #    elif _p == "Bias:":
        #        next_1 = True
            generated_text = generated_text.replace(_p, "", count)

    num_input_tokens = inputs['input_ids'].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    generated_text = generated_text.strip()
    return generated_text

