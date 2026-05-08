import os
import boto3
import json
import base64


class Chat:

    def __init__(self, model_id, bedrock_runtime_client):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.payload = {
            "messages": [],
            "max_tokens": 10000,
            "anthropic_version": "bedrock-2023-05-31"
        }

    
    def add_system_message(self, system_message):
        self.payload["messages"].append({
                                        "role": "system",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": system_message
                                            }
                                        ]
                                    })
        

    def add_user_message(self, message):
        self.payload["messages"].append({
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": message
                                            }
                                        ]
                                    })

    def add_user_message_image(self, message, encoded_image):
        self.payload["messages"].append({
                                        "role": "user",
                                        "content": [
                                                    {
                                                        "type": "image",
                                                        "source": {
                                                            "type": "base64",
                                                            "media_type": "image/jpeg",
                                                            "data": encoded_image,
                                                            # "temperature": 1,
                                                        }
                                                    },
                                                    {
                                                        "type": "text",
                                                        "text": message,
                                                        # "temperature": 1,
                                                    }
                                                ]
                                    })


    def generate(self):
        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(self.payload)
        )

        # now we need to read the response. It comes back as a stream of bytes so if we want to display the response in one go we need to read the full stream first
        # then convert it to a string as json and load it as a dictionary so we can access the field containing the content without all the metadata noise
        output_binary = response["body"].read()
        output_json = json.loads(output_binary)
        output = output_json["content"][0]["text"]


        self.payload["messages"].append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": output
                    }
                ]
            }
        )

        return output
    

class ChatCohere:
    def __init__(self, model_id, bedrock_runtime_client):
        self.model_id = model_id
        self.bedrock_runtime_client = bedrock_runtime_client
        self.payload = {
            "message": "",
            "chat_history": [],
            "max_tokens": 10000,
            "stop_sequences": [],
        }

        
    def add_user_message(self, message):
        self.payload["message"] = message

    def generate(self):
        response = self.bedrock_runtime_client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            body=json.dumps(self.payload),
        )

        # Read and parse the response
        output_binary = response["body"].read()
        output_json = json.loads(output_binary)
        output = output_json.get("text")

        # Append the assistant's response to the prompt for continuation
        # self.payload["chat_history"] += f"Assistant: {output}\n"

        return output



def load_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")