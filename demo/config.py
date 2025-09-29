"""Demo configuration constants."""

from typing import List, Dict, Any

# Model configuration
DEFAULT_MODEL_NAME = "ui_ug"
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"

# Default parameters
DEFAULT_EXTRA_PARAMS = {
    "repetition_penalty": 1.1
}

# Task types
SUPPORTED_TASKS = ["referring", "grounding", "generation", "captioning"]

# Demo prompts configuration
DEMO_PROMPTS = {
    "referring": [
        {
            "image_url": "https://mdn.alipayobjects.com/huamei_v87pyo/afts/img/A*0Hr3TpMzUN4AAAAAAAAAAAAAepd5AQ/original",
            "prompt": "Describe the region <|box_start|>(344, 358),(524, 390)<|box_end|>."
        }
    ],
    "grounding": [
        {
            "image_url": "https://mdn.alipayobjects.com/huamei_v87pyo/afts/img/A*0Hr3TpMzUN4AAAAAAAAAAAAAepd5AQ/original",
            "prompt": "List all ui items in the image."
        }
    ],
    "generation": [
        {
            "image_url": "https://mdn.alipayobjects.com/huamei_yebtei/afts/img/-gTwSql0JIwAAAAARzAAAAgADnpRAQFr/original",
            "prompt": "Generate a UI card to display user's borrowing limit and repayment information, helping users quickly understand available loan amounts and repayment status. The card top should show current borrowable amount (e.g., 250,002 yuan) with a \"Enable Credit Protection\" button to guide users to improve credit score. Below should clearly indicate total limit (e.g., 300,000 yuan) and daily interest rate (e.g., 0.035%), giving users clear understanding of borrowing costs. The middle section should include a \"Multiple discounts available\" label with red packet icon to attract user attention and encourage discount usage. Bottom should provide two main action buttons: \"Apply for loan\" and \"Repay loan\", with the \"Repay loan\" button displaying due amount for today (e.g., 5,000.00 yuan) to help users handle repayments promptly. Overall design should be clean and concise to ensure users can quickly access key information and take appropriate actions."
        },
        {
            "image_url": "https://mdn.alipayobjects.com/huamei_yebtei/afts/img/vfqvR76XF1wAAAAAR8AAAAgADnpRAQFr/original",
            "prompt": "Please design a UI card to display insurance claim evaluation results, ensuring users can immediately understand whether their claim meets requirements. The card top should include a prominent icon with text \"Meets claim requirements\" and a note \"See evaluation details for specific results\". Add a shield-shaped icon on the right to enhance visual trust. The lower area should clearly list consultation product name (e.g., \"Health Blessing·Cancer Prevention Plan 1 (Senior Version)\"), consultation disease (e.g., \"Cervical cancer\") and consultation time (e.g., \"2023.03.21\") for quick user verification. Additionally, include a \"Consultation History\" link next to the consultation time for users to view detailed historical records. Overall design should be clean and concise to ensure users can quickly access key information and proceed with next steps."
        }
    ],
    "captioning": [
        {
            "image_url": "https://mdn.alipayobjects.com/huamei_yebtei/afts/img/-gTwSql0JIwAAAAARzAAAAgADnpRAQFr/original",
            "prompt": "Describe this UI image."
        }
    ]
}