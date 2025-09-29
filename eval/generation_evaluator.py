"""Generation evaluator for UI generation tasks."""

import json
import os
from typing import Any, Dict, List

import pandas as pd

from eval.base_evaluator import BaseEvaluator
from eval.config import OUTPUT_PATH


class GenerationEvaluator(BaseEvaluator):
    """Evaluator for generation tasks (with and without images)."""
    
    def __init__(self, **kwargs):
        """Initialize generation evaluator."""
        super().__init__(**kwargs)
        self.data_paths = [
            'eval_files/ui_gen_mm.tsv',
            'eval_files/ui_gen_txt.tsv'
        ]
    
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
    
    def evaluate_generation_results(self, predictions: List[str], file_idx: int) -> Dict[str, float]:
        """Evaluate generation results based on JSON format and mock data."""
        sum_score_json_format = 0
        sum_score_mock_data = 0
        total_samples = len(predictions)
        
        for pred in predictions:
            try:
                if file_idx == 0:  # Multi-modal
                    gen_json_data = json.loads(pred)
                    sum_score_json_format += 1
                    self.extract_bind_fields(gen_json_data)
                    sum_score_mock_data += 1
                else:  # Text-only
                    tmp_res = pred.split('卡片的mock数据：')[-1]
                    tmp_res = tmp_res.split('卡片的dsl：')
                    model_mock = tmp_res[0].strip()
                    model_dsl = tmp_res[-1].strip()
                    
                    gen_dsl_data = json.loads(model_dsl)
                    gen_mock_data = json.loads(model_mock)
                    sum_score_json_format += 1
                    self.extract_bind_fields(gen_dsl_data)
                    sum_score_mock_data += 1
            except Exception as e:
                print(f"Error processing prediction: {e}")
        
        return {
            'score_json_format': sum_score_json_format / total_samples,
            'score_mock_data': sum_score_mock_data / max(sum_score_json_format, 1),
        }
    
    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run generation evaluation."""
        exp_name, checkpoint = self.get_experiment_info()
        
        file_names = ['gen_mm_metric.json', 'gen_txt_metric.json']
        eval_file_names = ['gen_mm.json', 'gen_txt.json']
        
        results = {}
        
        for file_idx, (data_path, file_name, eval_file_name) in enumerate(
            zip(self.data_paths, file_names, eval_file_names)
        ):
            print(f"Processing file {file_idx + 1}/2: {data_path}")
            
            # Get save directory
            save_dir = self.get_save_dir(exp_name, checkpoint, 0)
            self.create_directories(save_dir)
            
            # Load data
            data = self.load_data(data_path)
            
            # Run inference
            results_dict, eval_data = self.run_batch_inference(
                data, save_dir, eval_file_name, is_image_based=(file_idx == 0)
            )
            
            # Get predictions
            predictions = [item['model_output'] for item in eval_data]
            
            # Evaluate results
            metrics = self.evaluate_generation_results(predictions, file_idx)
            
            # Save metrics
            results_file = os.path.join(save_dir, file_name)
            self.save_json(metrics, results_file)
            
            results[f"file_{file_idx}"] = metrics
        
        return results


if __name__ == '__main__':
    evaluator = GenerationEvaluator()
    results = evaluator.run_evaluation()
    print("Generation evaluation completed:", results)