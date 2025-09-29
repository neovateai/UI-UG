"""Referring evaluator for UI element description tasks."""

import json
import math
import os
import re
from typing import Any, Dict, List

import numpy as np

from eval.base_evaluator import BaseEvaluator
from eval.config import OUTPUT_PATH


class ReferringEvaluator(BaseEvaluator):
    """Evaluator for referring/recommendation tasks."""
    
    CATEGORIES = {
        'text': 78442, 'image': 9198, 'SelectabletextButton': 9403, 'close': 2783, 'popup': 2204,
        'icon': 29584, 'checkedText': 827, 'NonselectabletextButton': 844, 'uncheckedText': 1046,
        'uncheckedBox': 500, 'checkTextView': 745, 'switch': 191, 'progress': 165, 'textButton': 273,
        'dropDown': 413, 'editText': 862, 'return': 7321, 'more': 3203, 'checkedBox': 156
    }
    
    def __init__(self, **kwargs):
        """Initialize referring evaluator."""
        super().__init__(**kwargs)
        self.data_path = 'eval_files/referring_sr.tsv'
    
    def get_score_text_same(self, str1: str, str2: str) -> int:
        """Check if two strings are exactly the same."""
        return 1 if str1 == str2 else 0
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_score_text_similarity(self, str1: str, str2: str) -> float:
        """Calculate text similarity using Levenshtein distance."""
        if str1 == str2:
            return 1.0
        
        distance = self.levenshtein_distance(str1, str2)
        max_length = max(len(str1), len(str2))
        similarity_score = 1 - (distance / max_length) if max_length > 0 else 0.0
        return max(similarity_score, 0.0)
    
    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    
    def get_score_color_similarity(self, color_data1: List[str], color_data2: List[str]) -> float:
        """Calculate color similarity between two color sets."""
        if len(color_data1) != len(color_data2):
            return 0.0
        
        if not color_data1 and not color_data2:
            return 1.0
        
        try:
            color1 = color_data1[0]
            color2 = color_data2[0]
            
            rgb1 = self.hex_to_rgb(color1)
            rgb2 = self.hex_to_rgb(color2)
            
            distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)))
            max_distance = math.sqrt(3 * (255 ** 2))
            normalized_difference = distance / max_distance
            
            return 1 - normalized_difference
        except (IndexError, ValueError):
            return 0.0
    
    def get_score_clickable_boxes(self, box1: Any, box2: Any) -> int:
        """Check if clickable status matches."""
        return 1 if box1 == box2 else 0
    
    def normalize_box_data(self, box: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize box data format."""
        if 'clickable_boxes' in box:
            try:
                text_color = list(box['text_color'].keys()) if isinstance(box['text_color'], dict) else box['text_color']
            except (KeyError, TypeError):
                text_color = box.get('text_color', [])
            
            clickable = bool(box['clickable_boxes'])
            
            return {
                'category': box['category'],
                'text': box['text'],
                'text_color': text_color,
                'clickable': clickable
            }
        return box
    
    def evaluate_referring_results(
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate referring results across all categories."""
        
        # Initialize counters
        sum_score_json_format = 0
        sum_score_category_same = 0
        sum_score_text_same = 0
        sum_score_text_similarity = 0
        sum_score_color_similarity = 0
        sum_score_clickable_boxes = 0
        sum_box_cnt = 0
        
        # Initialize category-wise stats
        cat_data = {cat: {
            "sum_box_cnt": 0.000001,
            "sum_score_json_format": 0,
            "sum_score_category_same": 0,
            "sum_score_text_same": 0,
            "sum_score_text_similarity": 0,
            "sum_score_color_similarity": 0,
            "sum_score_clickable_boxes": 0
        } for cat in self.CATEGORIES}
        
        # Process predictions
        for prediction, answer in zip(predictions, ground_truths):
            try:
                box = json.loads(prediction)
                tmp_category = answer['category']
                
                # Update counters
                sum_box_cnt += 1
                cat_data[tmp_category]['sum_box_cnt'] += 1
                
                # Normalize box format
                box = self.normalize_box_data(box)
                
                # Validate required fields
                required_fields = ["category", "text", "text_color", "clickable"]
                if not all(field in box for field in required_fields):
                    continue
                
                sum_score_json_format += 1
                cat_data[tmp_category]['sum_score_json_format'] += 1
                
                # Calculate scores
                score_category_same = self.get_score_text_same(box["category"], answer["category"])
                score_text_same = self.get_score_text_same(box["text"], answer["text"])
                score_text_similarity = self.get_score_text_similarity(box["text"], answer["text"])
                score_color_similarity = self.get_score_color_similarity(
                    box["text_color"], answer["text_color"]
                )
                score_clickable_boxes = self.get_score_clickable_boxes(
                    box["clickable"], answer["clickable"]
                )
                
                # Update sums
                sum_score_category_same += score_category_same
                sum_score_text_same += score_text_same
                sum_score_text_similarity += score_text_similarity
                sum_score_color_similarity += score_color_similarity
                sum_score_clickable_boxes += score_clickable_boxes
                
                # Update category data
                cat_data[tmp_category]['sum_score_category_same'] += score_category_same
                cat_data[tmp_category]['sum_score_text_same'] += score_text_same
                cat_data[tmp_category]['sum_score_text_similarity'] += score_text_similarity
                cat_data[tmp_category]['sum_score_color_similarity'] += score_color_similarity
                cat_data[tmp_category]['sum_score_clickable_boxes'] += score_clickable_boxes
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing prediction: {e}")
                continue
        
        # Calculate final results
        ret = {
            "sum_score_json_format": sum_score_json_format / max(sum_box_cnt, 1),
            "sum_score_category_same": sum_score_category_same / max(sum_box_cnt, 1),
            "sum_score_text_same": sum_score_text_same / max(sum_box_cnt, 1),
            "sum_score_text_similarity": sum_score_text_similarity / max(sum_box_cnt, 1),
            "sum_score_color_similarity": sum_score_color_similarity / max(sum_box_cnt, 1),
            "sum_score_clickable_boxes": sum_score_clickable_boxes / max(sum_box_cnt, 1)
        }
        
        # Add category-wise results
        for cat in cat_data:
            ret[cat] = {}
            for key in cat_data[cat]:
                if key != 'sum_box_cnt':
                    ret[cat][key] = cat_data[cat][key] / max(cat_data[cat]['sum_box_cnt'], 1)
        
        return ret
    
    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run referring evaluation."""
        exp_name, checkpoint = self.get_experiment_info()
        save_dir = self.get_save_dir(exp_name, checkpoint, 0)
        self.create_directories(save_dir)
        
        # Load data
        data = self.load_data(self.data_path)
        
        
        # Run inference
        results_dict, eval_data = self.run_batch_inference(
            data, save_dir, 'referring.jsonl', is_image_based=True
        )
        
        # Collect predictions and ground truths
        predictions = [item['model_output'] for item in eval_data]
        ground_truths = []
        
        for idx, info in data.iterrows():
            answer_boxes = json.loads(info.answer)
            ground_truths.append(answer_boxes)
        
        # Evaluate results
        metrics = self.evaluate_referring_results(predictions, ground_truths)
        
        # Save metrics
        self.save_json(metrics, os.path.join(save_dir, 'referring.json'))
        
        print("Referring evaluation completed")
        return metrics


if __name__ == '__main__':
    evaluator = ReferringEvaluator()
    results = evaluator.run_evaluation()
    print("Referring evaluation completed:", results)