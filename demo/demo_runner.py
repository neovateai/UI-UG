"""Demo runner for UI understanding tasks."""

import json
import time
from typing import List, Dict, Any, Optional, Callable

from utils.vllm_chat import VLLMChatInference
from utils.grounding_utils import visualize_box_image
from demo.config import (
    DEFAULT_MODEL_NAME, 
    DEFAULT_API_BASE, 
    DEFAULT_EXTRA_PARAMS, 
    SUPPORTED_TASKS
)


class DemoRunner:
    """Unified demo runner for UI understanding tasks."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_base: str = DEFAULT_API_BASE,
        extra_params: Dict[str, Any] = None
    ):
        """Initialize the demo runner."""
        self.model_name = model_name
        self.api_base = api_base
        self.extra_params = extra_params or DEFAULT_EXTRA_PARAMS
        self.vllm = VLLMChatInference(api_base=api_base)
        
        # Task handlers
        self.task_handlers = {
            "grounding": self._handle_grounding_task,
            "generation": self._handle_generation_task,
            "referring": self._handle_referring_task,
            "captioning": self._handle_captioning_task,
            "other": self._handle_other_task
        }
    
    def extract_bind_fields(self, data: Dict[str, Any], bind_fields: List[str] = None) -> Dict[str, str]:
        """Extract bind fields from DSL data."""
        if bind_fields is None:
            bind_fields = []
        
        if isinstance(data, dict):
            if "bindField" in data:
                bind_fields.append(data["bindField"])
            for value in data.values():
                self.extract_bind_fields(value, bind_fields)
        elif isinstance(data, list):
            for item in data:
                self.extract_bind_fields(item, bind_fields)
        
        return {field: f"M{field}" for field in bind_fields}
    
    def _extract_name_from_url(self, image_url: str) -> str:
        """Extract a safe filename from image URL."""
        try:
            return image_url.split("img/")[-1].split("/")[0].replace("*", "_")
        except IndexError:
            return "demo_image"
    
    def _handle_grounding_task(self, image_url: str, response: str, **kwargs) -> Dict[str, Any]:
        """Handle grounding task-specific processing."""
        name = self._extract_name_from_url(image_url)
        print(f"Processing grounding for image: {name}")
        
        try:
            visualize_box_image(name, image_url, response, "sr")
            return {"visualization": True, "image_name": name}
        except Exception as e:
            return {"visualization": False, "error": str(e)}
    
    def _handle_generation_task(self, image_url: str, response: str, **kwargs) -> Dict[str, Any]:
        """Handle UI generation task-specific processing."""
        try:
            parsed_data = json.loads(response)
            bind_fields = self.extract_bind_fields(parsed_data)
            return {"bind_fields": bind_fields, "parsed_data": parsed_data}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}"}
    
    def _handle_referring_task(self, image_url: str, response: str, **kwargs) -> Dict[str, Any]:
        """Handle referring task-specific processing."""
        return {"description": response}
    
    def _handle_captioning_task(self, image_url: str, response: str, **kwargs) -> Dict[str, Any]:
        """Handle captioning task processing."""
        return {"response": response}
    
    def _handle_other_task(self, image_url: str, response: str, **kwargs) -> Dict[str, Any]:
        """Handle custom task processing."""
        return {"response": response}
    
    def process_single_demo(
        self,
        image_url: str,
        prompt: str,
        task: str = "referring",
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a single demo request."""
        if task not in self.task_handlers:
            raise ValueError(f"Unsupported task: {task}. Supported: {SUPPORTED_TASKS}")
        
        start_time = time.time()
        
        try:
            if stream:
                response = self.vllm.chat_with_image_stream(
                    self.model_name, prompt, image_url, 
                    temperature=0, extra_params=self.extra_params
                )
            else:
                response = self.vllm.chat_with_image(
                    self.model_name, prompt, image_url,
                    temperature=0, extra_params=self.extra_params
                )
            
            processing_time = time.time() - start_time
            
            # Handle task-specific processing
            task_result = self.task_handlers[task](image_url, response, **kwargs)
            
            return {
                "success": True,
                "task": task,
                "image_url": image_url,
                "prompt": prompt,
                "response": response,
                "processing_time": processing_time,
                **task_result
            }
        
        except Exception as e:
            return {
                "success": False,
                "task": task,
                "image_url": image_url,
                "prompt": prompt,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_task_batch(
        self,
        task_prompts: List[Dict[str, str]],
        task: str,
        stream: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a batch of prompts for a specific task."""
        results = []
        
        for i, demo_item in enumerate(task_prompts, 1):
            print(f"Processing {task} demo {i}/{len(task_prompts)}...")
            
            result = self.process_single_demo(
                demo_item["image_url"],
                demo_item["prompt"],
                task=task,
                stream=stream,
                **kwargs
            )
            
            results.append(result)
            
            # Print separator
            print("\n" + "-" * 80)
            print(f"Task: {task} | Demo {i}/{len(task_prompts)}")
            print(f"Success: {result['success']}")
            print(f"Time: {result['processing_time']:.2f}s")
            if result['success']:
                print(f"Response: {result['response'][:200]}...")
            else:
                print(f"Error: {result['error']}")
            print("-" * 80 + "\n")
        
        return results
    
    def run_all_tasks(
        self,
        tasks: List[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run all supported tasks."""
        from demo.config import DEMO_PROMPTS
        
        if tasks is None:
            tasks = SUPPORTED_TASKS
        
        all_results = {}
        
        for task in tasks:
            if task in DEMO_PROMPTS:
                print(f"\n{'='*80}")
                print(f"RUNNING {task.upper()} TASKS")
                print(f"{'='*80}\n")
                
                task_prompts = DEMO_PROMPTS[task]
                results = self.process_task_batch(
                    task_prompts, task, stream=stream, **kwargs
                )
                all_results[task] = results
            else:
                print(f"Warning: No prompts configured for task '{task}'")
        
        return all_results
    
    def run_single_task(
        self,
        image_url: str,
        prompt: str,
        task: str = "referring",
        **kwargs
    ) -> Dict[str, Any]:
        """Run a single demo task."""
        return self.process_single_demo(image_url, prompt, task=task, **kwargs)