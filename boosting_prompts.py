from utils import read_products, products_to_string, align, get_tokenizer_aws_model_id, generate_response
from aws import Chat
import boto3
import os


def social_proof(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is the most is the most popular choice among the customers.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def exclusivity(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is available for selected customers only. Showing exclusivity to a specific group of people or members.
Do not confuse this with scarcity, where the product availability is due to limited in numbers.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def attack_scarcity(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is in limited availability. E.g. only 3 items left.
Do not confuse this with exclusivity, where the product is available only to a special group of people.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def attack_discount_framing(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll apply some of the following discount framing methods:

- Highlight percentage-based discounts (e.g., "20% discount").
- Mention dollar-amount savings (e.g., "$200 off").
- Emphasize the previous price compared to the discounted price (e.g., "Was $100, now only $80").
- Use comparative statements about previous prices (e.g., "Price was 25% higher, now discounted").
- Add promotional statements about free products or extra value (e.g., "25% more product for free").
- Ensure that the rewritten description feels natural, engaging, and effectively highlights the benefits of the discount for the customer. 

Note that you cannot change the product’s final price but can mention that the initial price was higher.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Final price of the product: {final_price}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def contrast_effect(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that other products with the same features are more expensive.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def decoy_effect(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that similarly priced products have less features.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def authority_bias(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that trusted by leading experts/professionals, this is the go-to choice for those who know quality.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def bizarreness_effect(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add some humor, such as a relevant one liner jokes, e.g. Why don’t scientists trust atoms? Because they make up everything!"

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def denominator_neglect(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll divide the product price by 90 and mention that you can have the product with only x dollars a day.
e.g. If the product costs 90 dollars, you will mention something like "Get this product for only 1 dollar a day!".

Note that you cannot change the product’s final price.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Final price of the product: {final_price}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def storytelling_effect(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add the product into a narrative. For example, paint a picture like "It's a nice day and you smell the strong scent of good coffee from the kitchen".

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()

def identity_signaling(product, aws_access_key_id, aws_secret_access_key):
    
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-west-2"
    )
    chat = Chat(model_id, bedrock_runtime_client)
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add information signaling an inclusive and socially positive stance. Do not make the signaling political or taking a political stance. Just general inclusive, feel-good concepts.
i.e. "This is an LGBT inclusive product" or "This product is against cruelty in animals."

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    chat.add_user_message(prompt)
    return chat.generate()
