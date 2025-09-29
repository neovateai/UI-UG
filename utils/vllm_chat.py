from openai import OpenAI
import requests
import base64
import os


class VLLMChatInference:
    def __init__(self, api_key="EMPTY", api_base="http://localhost:8000/v1", max_tokens=7000):
        """
        Initialize OpenAI client.
        :param api_key: OpenAI API key
        :param api_base: OpenAI API base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.max_tokens = max_tokens

    @staticmethod
    def image_to_base64(input_value):
        """
        Convert image to Base64 encoding.
        :param input_value: Image URL or local file path
        :return: Base64 encoded string
        """
        if input_value.startswith('http://') or input_value.startswith('https://'):
            response = requests.get(input_value)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                raise Exception(f"Failed to fetch image, status code: {response.status_code}")
        else:
            if os.path.isfile(input_value):
                with open(input_value, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            else:
                raise Exception(f"File does not exist: {input_value}")

    def chat_with_image(self, model_name, prompt, image_path_or_url, temperature=0.1, extra_params={}):
        """
        Chat with model using image and text.
        :param model_name: Model name
        :param prompt: Text prompt
        :param image_path_or_url: Image URL or local file path
        :param temperature: Temperature parameter controlling text generation randomness
        :return: Model's response content
        """
        print("image_path_or_url", image_path_or_url)
        if image_path_or_url is not None:
            # Convert image to Base64 encoding
            image_base64 = self.image_to_base64(image_path_or_url)

            # Build message
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28
                    },
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": prompt
            }]

        # Call OpenAI API to get response
        chat_response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            extra_body=extra_params,
            max_tokens=self.max_tokens
        )
        print(chat_response)

        # Return model's response content
        response = chat_response.choices[0].message.content
        print("Chat completion output:", response)
        return response

    def chat_with_image_stream(self, model_name, prompt, image_path_or_url, temperature=0.1, extra_params={}):
        """
        Stream chat with model using image and text.
        :param model_name: Model name
        :param prompt: Text prompt
        :param image_path_or_url: Image URL or local file path
        :param temperature: Temperature parameter controlling text generation randomness
        :return: Complete response content
        """
        if image_path_or_url is not None:
            # Convert image to Base64 encoding
            image_base64 = self.image_to_base64(image_path_or_url)

            # Build message
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28
                    },
                ],
            }]
        else:
            messages = [{
                "role": "user",
                "content": prompt
            }]

        chat_stream = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=extra_params,
            max_tokens=self.max_tokens
        )

        print("client: Start streaming chat completions...")
        printed_reasoning_content = False
        printed_content = False

        response = ""
        for chunk in chat_stream:
            reasoning_content = None
            content = None
            if hasattr(chunk.choices[0].delta, "reasoning_content"):
                reasoning_content = chunk.choices[0].delta.reasoning_content
            elif hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content

            if reasoning_content is not None:
                if not printed_reasoning_content:
                    printed_reasoning_content = True
                    print("reasoning_content:", end="", flush=True)
                print(reasoning_content, end="", flush=True)
            elif content is not None:
                if not printed_content:
                    printed_content = True
                    print("\ncontent:", end="", flush=True)
                print(content, end="", flush=True)
                response += content
        return response
