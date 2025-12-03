SYS_PROMPT = '''
You are an advanced AI model equipped with OCR and image understanding capabilities, capable of analyzing visual elements in detail. 
'''

PROMPT = '''
Your task is to assess two UI images and output a score between 0 and 10 for each of the following questions.
If the answer to a question is a definite YES, output a score of 10, signifying perfect similarity. Conversely, a definite NO should yield a score of 0, indicating no similarity.
For answers that fall in between, assign a score accordingly, where a higher number indicates a greater degree of similarity. Only provide the numerical score for each question, without any additional text.
Example contexts are provided for clarity. Examples provides the idea, but you can output any number in 0-10 range accordingly. Only output a comma separated list containing 4 numbers. DO NOT give score of 10 for any category unless otherwise the two images are identical.

1. Visual structure and alignment(Score: 0-10):
Does the placement of elements like images, buttons, and text boxes aligned similarly on both images? 
Do the sizes and aspect ratios of images, buttons, and text boxes appear consistent across both images?  
Do both UI images exhibit a similar level of visual harmony and balance in their design? 
(e.g., A score of 10 for identical visual structure and alignment, 5 for similar but not exact consistent, and 0 for completely different structure.)

2. Color match and Aesthetic Resemblance(Score: 0-10):
How closely do the color schemes of the two UI images align in terms of background and text colors? Evaluate the similarity in hues, saturation, and overall color
aesthetics. Is the overall aesthetic appeal (modern, minimalistic, traditional, etc.) similar on both UI images? 
(e.g., A score of 10 for perfectly matching color schemes and identical aesthetics, including identical hues and saturation levels, 6 for similar color palettes and styles with minor variations, and 0 for starkly different color
schemes and aesthetics that create entirely different visual impacts.)

3. Textual and Content Consistency(Score: 0-10):
Do the font type, size, style, and weight of two UI images similar?
Do the words and sentences match between the two UI images?
(e.g., A score of 10 for complete uniformity in font type, size, style, weight across both UI images and identical text, 5 for consistency in font type and size but variations in style or weight, and 0 for wide disparities in font type, size, style, or weight, leading to a distinctly different
textual appearance and content.)

4. User Interface Consistency (Score: 0-10): 
Do the user interface elements (like menus, buttons, and forms) on both UI images share a similar design language and appearance? 
(e.g., A score of 10 for identical UI elements, 6 for slight design variations, and 0 for completely different UI designs.)
'''

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import base64
import requests
from io import BytesIO
import argparse
from utils import render_utils


def img_to_base64(img_path):
    """Convert image to Base64 (support network/local paths)"""
    print(img_path)
    try:
        if img_path.startswith(('http', 'https')):
            res = requests.get(img_path, timeout=5, verify=False)
            img_data = res.content
            print("---------------", res.status_code)
        else:
            with open(img_path.encode('utf-8').decode('latin1'), 'rb') as f:
                img_data = f.read()

        image_data = Image.open(BytesIO(img_data))
        mime = Image.open(BytesIO(img_data)).format.lower()
        return base64.b64encode(img_data).decode(), image_data
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        return None, None


def count_matching_images(folder_a, folder_b, folder_c):
    """
    Traverse all PNG images in folder_a and look for files with the same name in folder_b and folder_c.
    If files with the same name exist in both folder_b and folder_c, the counter increases by one.

    Args:
        folder_a (str): Folder path containing PNG images.
        folder_b (str): Folder path to look for files with the same name.
        folder_c (str): Another folder path to look for files with the same name.

    Returns:
        int: Number of qualifying images.
    """
    count = 0

    # Ensure folders exist
    if not os.path.isdir(folder_a):
        print(f"Error: Folder '{folder_a}' does not exist.")
        return count
    if not os.path.isdir(folder_b):
        print(f"Error: Folder '{folder_b}' does not exist.")
        return count
    if not os.path.isdir(folder_c):
        print(f"Error: Folder '{folder_c}' does not exist.")
        return count

    # Traverse all files in folder_a
    for filename in os.listdir(folder_a):
        # Check if file is a PNG image
        if filename.lower().endswith('.png'):
            # Build full paths in folder_b and folder_c
            path_b = os.path.join(folder_b, filename)
            path_c = os.path.join(folder_c, filename)

            # Check if files with the same name exist in folder_b and folder_c
            if os.path.isfile(path_b) and os.path.isfile(path_c):
                count += 1
                print(f"Found matching file: {filename}")
    return count


def save_to_jsonl(data_list, output_jsonl_path):
    """
    Saves a list of dictionaries to a JSONL file, with each dictionary on a new line.

    Args:
        data_list (list): A list of dictionaries to save.
        output_jsonl_path (str): The path to the output JSONL file.
    """
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in data_list:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(data_list)} entries to {output_jsonl_path}")
    except IOError as e:
        print(f"Error saving to JSONL file {output_jsonl_path}: {e}")


def call_openai_api(image1_path, image2_path, api_key=None, base_url=None, model=None):
    """
    Use OpenAI-compatible API for image evaluation

    Args:
        image1_path (str): First image path
        image2_path (str): Second image path
        api_key (str, optional): OpenAI API key, will get from environment variable if not provided
        base_url (str, optional): API base URL, will use official OpenAI URL if not provided
        model (str, optional): Model name to use

    Returns:
        str: API response result
    """
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or "https://api.openai.com/v1"
    )

    # Convert images to base64
    img1_base64, _ = img_to_base64(image1_path)
    img2_base64, _ = img_to_base64(image2_path)

    if not img1_base64 or not img2_base64:
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYS_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img1_base64}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img2_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.0
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None


def process_with_openai(data_path, api_key=None, base_url=None, model=None, ):
    folder_a = 'eval_files/ui_gen_images'
    ckpt = ""
    model_name = data_path.split("/")[-1].split(".")[0]
    folder_b = f'eval_files/render/{model_name}/'

    render_success_rate = render_images(data_path, folder_b)

    processed_data_list = []
    total = 0
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0

    for index in range(100 + 1):
        print(f"Processing sample {index}...")
        filename = f"{index}.png"

        if not os.path.exists(os.path.join(folder_a, filename)) or not os.path.exists(os.path.join(folder_b, filename)):
            continue

        file_name_b = f"{index}.png"

        image1_path = os.path.join(folder_a, filename)
        image2_path = os.path.join(folder_b, file_name_b)

        if not os.path.exists(image2_path):
            continue

        response = call_openai_api(image1_path, image2_path, api_key, base_url, model)

        if response:
            try:
                scores_str = response.split(',')
                scores_list = [int(s.strip()) for s in scores_str]
                if len(scores_list) == 4:
                    sum_score = sum(scores_list)
                    total += sum_score
                    score1 += scores_list[0]
                    score2 += scores_list[1]
                    score3 += scores_list[2]
                    score4 += scores_list[3]

                    json_score = {
                        "index": index,
                        "score": sum_score,
                        "scores": scores_list
                    }
                    processed_data_list.append(json_score)
                    print(f"Sample {index} processed, score: {sum_score}")
            except Exception as e:
                print(f"Failed to parse sample {index}: {str(e)}")

    num_processed_samples = len(processed_data_list)
    if num_processed_samples > 0:
        output_json_filename = f'eval_files/res/{model_name}.jsonl'
        save_to_jsonl(processed_data_list, output_json_filename)

        average_total = total / num_processed_samples
        average_score1 = score1 / num_processed_samples
        average_score2 = score2 / num_processed_samples
        average_score3 = score3 / num_processed_samples
        average_score4 = score4 / num_processed_samples

        metrics_data = {
            "num_processed_samples": num_processed_samples,
            "total_average": average_total,
            "score1_average": average_score1,
            "score2_average": average_score2,
            "score3_average": average_score3,
            "score4_average": average_score4,
            "render_success_rate": render_success_rate
        }

        output_json_filename_res = f'eval_files/res/{model_name}_{ckpt}_res.jsonl'
        with open(output_json_filename_res, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_data, ensure_ascii=False) + '\n')
        print(metrics_data)
        print(f"OpenAI API processing completed, statistics successfully saved to '{output_json_filename_res}'")


def render_images(data_path, folder):
    # Initialize renderer
    renderer = render_utils.DslRenderer()
    eval_cnt, success_cnt = 0, 0

    render_dsl_list = []
    if data_path.split(".")[-1] == "json":
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key, item in data.items():
                render_dsl_list.append(item["model_output"])
    elif data_path.split(".")[-1] == "jsonl":
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_data = json.loads(line)
                render_dsl_list.append(json_data["ui_dsl"])

    eval_cnt = len(render_dsl_list)
    print("eval_cnt", eval_cnt)

    for i in range(eval_cnt):
        try:
            if os.path.exists(folder + f"{i + 1}.png"):
                success_cnt += 1
            else:
                dsl_code = render_dsl_list[i]

                # Generate mock data or use custom data
                mock_data = renderer.generate_mock_data(dsl_code)

                # Prepare DSL data for processing
                dsl_data = {
                    'dsl_code': dsl_code,
                    'mock_data': json.dumps(mock_data, ensure_ascii=False)
                }

                # Process and capture screenshot
                res_message = render_utils.process_and_screenshot_task(renderer, dsl_data, folder + f"{i + 1}.png",
                                                                       f"render_ui.html")
                success_cnt += 1
                print(res_message)
        except Exception as e:
            print(e)

    return success_cnt / eval_cnt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UI image evaluation using OpenAI API')
    parser.add_argument('--data-path', type=str, help='Path to the data file (json or jsonl)')
    parser.add_argument('--mode', choices=['openai'], default='openai',
                        help='Processing mode: openai (use OpenAI API)')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--base-url', type=str, help='OpenAI API base URL')
    parser.add_argument('--model', type=str, help='OpenAI model name')

    args = parser.parse_args()

    process_with_openai(args.data_path, args.api_key, args.base_url, args.model)
