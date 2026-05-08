import boto3
import json
import os 
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import difflib
import os
import csv
from thefuzz import fuzz
import numpy as np
import nltk
from nltk.corpus import words
import string

nltk.download('words')

system_prompt = """You are a helpful AI assistant. This is a conversation between you and a User. You is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision."""


native_request = {
    "max_gen_len": 500,
    "temperature": 0,
    "top_p": 0.9
}

def read_products(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results

def read_products_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def products_to_string(d):
    text = ""
    for row in d:
        text += str(row) + "\n"
    return text

def align(text, products):
    d = {}
    for i, row in enumerate(products):
        name = row["Name"]
        if name in text:
            d[name] = [text.index(name), i]

    aligned_output = []
    for i, (_, v) in enumerate(sorted(d.items(), key=lambda item: item[1][0])):
        aligned_output.append({
            "llm_output": i,
            "se_output": v[1]
        })

    return aligned_output

def fuzzy_align(text, products, window = 5, pattern = 'numbered', adjuster = 10, focus_on_beginning = True):
    d = {}
    if '|im_start|' not in text:
        print(text)
    text = text.split('|im_start|')[1] if '|im_start|' in text else text
    
    candidate_substrings = text.split('\n')
    product_names = [prod['Name'] for prod in products]

    str_d = {}

    english_words = set(words.words())

    def tokenize_and_normalize(s):
        return set(word.strip(string.punctuation).lower() for word in s.split() if word)
    
    # Identify key/important words (proper nouns or uncommon words)
    def is_unique(word):
        return word.lower() not in english_words

    for idx, row in enumerate(products):
        name = row["Name"]

        for substring in candidate_substrings:
          if pattern == 'numbered':
            expression = r'^\d+\.'
            if not bool(re.match(expression, substring)):
              continue

          #stay towards beggining of name because thats where prod name is
          substring_split = substring.split(' ')
          name_split = name.split(' ')

          if focus_on_beginning:
            substring_split = substring_split[:int(len(substring_split)/2)] if len(substring_split) > len(name_split) else substring_split
            name_split = name_split[:int(len(substring_split)/2)]
          
          potential_window = min(len(substring_split), len(name_split))
          final_window = potential_window if potential_window < window else window

          fuzz_ratios = list()
          i = 0

          
          while i < len(substring_split) - final_window + 1:
            j = 0
            windowed_substring = (' ').join(substring_split[i:(i + final_window)])

            while j < len(name_split) - final_window + 1: 
              windowed_name = (' ').join(name_split[j:(j + final_window)])
              fuzz_similarity = fuzz.ratio(windowed_substring, windowed_name)

              unique_amount_substring = {word for word in tokenize_and_normalize(windowed_substring) if is_unique(word)}
              unique_amount_name = {word for word in tokenize_and_normalize(windowed_name) if is_unique(word)}
              # print(unique_amount_substring)
              adjusted_similarity = fuzz_similarity + len(unique_amount_substring.intersection(unique_amount_name)) * adjuster

              fuzz_ratios.append((adjusted_similarity, (windowed_name, windowed_substring, final_window, len(substring_split), len(name_split))))
              j += 1
            i += 1

          str_d[substring] = [(name, max(fuzz_ratios, key=lambda x: x[0]))] if str_d.get(substring) is None else str_d[substring] + [(name, max(fuzz_ratios, key=lambda x: x[0]))]


    for substring, val_list in str_d.items():
      sorted_val_list = sorted(val_list, key=lambda x: x[1][0], reverse=True)
      prod = sorted_val_list[0][0]
      d[prod] = [text.index(substring), product_names.index(prod)]    

    aligned_output = []
    for i, (_, v) in enumerate(sorted(d.items(), key=lambda item: item[1][0])):
        aligned_output.append({
            "llm_output": i,
            "se_output": v[1]
        })

    return aligned_output
    
class BedrockLlamaChatTemplate:
    def apply_chat_template(self, chat, tokenize=False):
        prompt = "<|begin_of_text|>"
        for message in chat:
            role = message["role"]
            content = message["content"]
            prompt += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                f"{content}<|eot_id|>"
            )
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

def get_tokenizer_aws_model_id(model_name):
    tokenizer, aws_model_id = None, None
    if model_name == "llama3.1-8b":
        tokenizer = BedrockLlamaChatTemplate()
        aws_model_id= "meta.llama3-1-8b-instruct-v1:0"

    elif model_name == "llama3.1-70b":
        tokenizer = BedrockLlamaChatTemplate()
        aws_model_id= "meta.llama3-70b-instruct-v1:0"

    elif model_name == "llama3.1-405b":
        tokenizer = BedrockLlamaChatTemplate()
        aws_model_id= "meta.llama3-1-405b-instruct-v1:0"
        
    return tokenizer, aws_model_id

    
# model_name = "llama3.1-8b" #os.getenv("model_name")
# tokenizer, aws_model_id = get_tokenizer_aws_model_id(model_name)
    

def generate_response(user_instr, system_prompt, tokenizer, aws_model_id, aws_access_key_id, aws_secret_access_key, native_request = native_request):
    
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
        
    
    chat = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_instr},
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # Format the request payload using the model's native structure.
    native_request["prompt"] = formatted_prompt

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=aws_model_id, body=request)

        input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
        output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])

    
    except Exception as e:
        print(f"ERROR: Can't invoke '{aws_model_id}'. Reason: {e}")
        exit(1)


    # Decode the response body.
    model_response = json.loads(response["body"].read())


    # Extract and print the response text.
    response_text = model_response["generation"]
    return response_text
        
