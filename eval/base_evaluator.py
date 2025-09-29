"""Base evaluator class for common evaluation functionality."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


from eval.config import (
    CUDA_DEVICES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_SAMPLING_CONFIG,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_PATH,
    OUTPUT_PATH,
    VLLM_CONFIG,
)


class BaseEvaluator(ABC):
    """Base class for all evaluation tasks."""
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        cuda_devices: str = CUDA_DEVICES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_path: str = OUTPUT_PATH,
    ):
        """Initialize the base evaluator."""
        self.model_path = model_path
        self.cuda_devices = cuda_devices
        self.batch_size = batch_size
        self.output_path = output_path
        self.cuda_sizes = len(cuda_devices.split(','))
        
        # Initialize model and processor
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=self.cuda_sizes,
            **VLLM_CONFIG
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.sampling_params = SamplingParams(**DEFAULT_SAMPLING_CONFIG)
    
    def get_experiment_info(self) -> tuple[str, str]:
        """Get experiment name and checkpoint from model path."""
        if 'checkpoint' not in self.model_path:
            exp_name = self.model_path.split('/')[-2] + '_' + self.model_path.split('/')[-1]
            checkpoint = '0000'
        else:
            exp_name = self.model_path.split('/')[-3]
            checkpoint = self.model_path.split('/')[-1].split('-')[1]
        return exp_name, checkpoint
    
    def get_save_dir(self, exp_name: str, checkpoint: str, eval_cnt: int) -> str:
        """Get the save directory path."""
        return os.path.join(self.output_path, exp_name, checkpoint, str(eval_cnt))
    
    def create_directories(self, save_dir: str) -> None:
        """Create necessary directories."""
        os.makedirs(save_dir, exist_ok=True)
    
    def get_next_eval_count(self, base_dir: str, file_name: str) -> int:
        """Get the next evaluation count by checking existing results."""
        if not os.path.exists(base_dir):
            return 0
        
        eval_cnt = 0
        for tmp_folder in os.listdir(base_dir):
            tmp_eval_file = os.path.join(base_dir, tmp_folder, file_name)
            if os.path.exists(tmp_eval_file):
                eval_cnt += 1
        return eval_cnt
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from TSV file."""
        return pd.read_csv(data_path, sep='\t')
    
    def create_image_messages(self, image_path: str, question: str) -> List[Dict[str, Any]]:
        """Create message format for image-based evaluation."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]
    
    def create_text_messages(self, question: str) -> List[Dict[str, Any]]:
        """Create message format for text-only evaluation."""
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            },
        ]
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Process a batch of inputs with the model."""
        if not batch:
            return []
            
        outputs = self.llm.generate(batch, sampling_params=self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def save_json(self, data: Any, file_path: str) -> None:
        """Save data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    @abstractmethod
    def run_evaluation(self, data_path: str, **kwargs) -> Dict[str, Any]:
        """Abstract method to run the specific evaluation."""
        pass
    
    def run_batch_inference(
        self,
        data: pd.DataFrame,
        save_dir: str,
        eval_file_name: str,
        is_image_based: bool = True,
    ) -> Dict[str, Any]:
        """Run batch inference for common patterns."""
        results = {}
        eval_data = []
        
        save_path = os.path.join(save_dir, eval_file_name)
        
        with open(save_path, 'w', encoding='utf-8') as writer:
            input_batch = []
            batch_metadata = []
            
            for idx, info in tqdm(data.iterrows(), total=len(data)):
                question = info.question
                
                if is_image_based:
                    messages = self.create_image_messages(info.image_path, question)
                else:
                    messages = self.create_text_messages(question)
                
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                

                if is_image_based:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
                    mm_data = {}
                    if image_inputs is not None:
                        mm_data["image"] = image_inputs
                    if video_inputs is not None:
                        mm_data["video"] = video_inputs
                    llm_inputs = {
                        "prompt": prompt,
                        "multi_modal_data": mm_data,
                        "mm_processor_kwargs": video_kwargs
                    }
                else:
                    llm_inputs = {
                        "prompt": prompt
                    }

                input_batch.append(llm_inputs)
                batch_metadata.append({
                    'question': question,
                    'image_path': info.image_path if hasattr(info, 'image_path') else None,
                    'index': idx,
                })
                
                if len(input_batch) == self.batch_size or idx == len(data) - 1:
                    outputs = self.process_batch(input_batch)
                    
                    for output, metadata in zip(outputs, batch_metadata):
                        result = {
                            'question': metadata['question'],
                            'model_output': output,
                        }
                        if metadata['image_path']:
                            result['image_path'] = metadata['image_path']
                        
                        writer.write(json.dumps(result, ensure_ascii=False) + '\n')
                        eval_data.append(result)
                        results[metadata['index']] = result
                    
                    input_batch = []
                    batch_metadata = []
        
        return results, eval_data