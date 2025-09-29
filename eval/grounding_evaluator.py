"""Grounding evaluator for bounding box detection tasks."""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.base_evaluator import BaseEvaluator
from eval.config import OUTPUT_PATH


class GroundingEvaluator(BaseEvaluator):
    
    """Evaluator for grounding/detection tasks."""
    
    def __init__(self, **kwargs):
        """Initialize grounding evaluator."""
        super().__init__(batch_size=48, **kwargs)
        self.data_path = 'eval_files/grounding_sr.tsv'
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int], threshold: float = 0.5) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            x1, y1, x2, y2 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            xi1, yi1 = max(x1, x1_2), max(y1, y1_2)
            xi2, yi2 = min(x2, x2_2), min(y2, y2_2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            
            intersection_area = (xi2 - xi1) * (yi2 - yi1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area
            
            if union_area <= 0:
                return 0.0
            
            iou = intersection_area / union_area
            return iou if iou >= threshold else 0.0
        
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    def match_boxes_hungarian(
        self,
        pred_boxes: List[List[int]],
        gt_boxes: List[List[int]],
        iou_thresholds: List[float] = [0.5, 0.75]
    ) -> Dict[str, float]:
        """Match predicted boxes with ground truth using Hungarian algorithm."""
        num_pred, num_gt = len(pred_boxes), len(gt_boxes)
        
        if num_pred == 0 or num_gt == 0:
            return {'mIoU50': 0.0, 'mIoU75': 0.0}
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((num_pred, num_gt))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Apply Hungarian algorithm
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        results = {}
        for threshold in iou_thresholds:
            total_iou = 0.0
            for i, j in zip(row_ind, col_ind):
                if iou_matrix[i, j] >= threshold:
                    total_iou += iou_matrix[i, j]
            results[f'mIoU{int(threshold*100)}'] = total_iou / num_gt
        
        return results
    
    def compute_iou(self, pred_box: List[int], gt_box: List[int]) -> float:
        """Compute IoU between two boxes."""
        x_min_inter = max(pred_box[0], gt_box[0])
        y_min_inter = max(pred_box[1], gt_box[1])
        x_max_inter = min(pred_box[2], gt_box[2])
        y_max_inter = min(pred_box[3], gt_box[3])
        
        if x_max_inter <= x_min_inter or y_max_inter <= y_min_inter:
            return 0.0
        
        intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        union_area = pred_area + gt_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def calculate_ap(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float
    ) -> float:
        """Calculate Average Precision for a specific category and IoU threshold."""
        if not predictions or not ground_truths:
            return 0.0
        
        # Sort predictions by score (descending)
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # Initialize variables
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_matched = set()
        
        # Find best matches
        for i, pred in enumerate(predictions):
            best_iou = 0.0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truths):
                if j in gt_matched:
                    continue
                
                iou = self.compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / len(ground_truths) if ground_truths else np.zeros_like(tp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        return ap
    
    def grounding_evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_thresholds: List[float] = [0.5, 0.75]
    ) -> Dict[str, Any]:
        """Evaluate grounding results."""
        pred_by_image_cat = {}
        gt_by_image_cat = {}
        
        # Group predictions and ground truths
        for pred in predictions:
            key = (pred['image_id'], pred['category_id'])
            pred_by_image_cat.setdefault(key, []).append(pred)
        
        for gt in ground_truths:
            key = (gt['image_id'], gt['category_id'])
            gt_by_image_cat.setdefault(key, []).append(gt)
        
        # Get all unique categories
        all_categories = set()
        for pred in predictions:
            all_categories.add(pred['category_id'])
        for gt in ground_truths:
            all_categories.add(gt['category_id'])
        
        # Calculate AP for each category and IoU threshold
        ap_results = {}
        for threshold in iou_thresholds:
            ap_results[threshold] = {}
            for category in all_categories:
                all_predictions = []
                all_ground_truths = []
                
                for (img_id, cat_id), preds in pred_by_image_cat.items():
                    if cat_id == category:
                        all_predictions.extend(preds)
                
                for (img_id, cat_id), gts in gt_by_image_cat.items():
                    if cat_id == category:
                        all_ground_truths.extend(gts)
                
                ap = self.calculate_ap(all_predictions, all_ground_truths, threshold)
                ap_results[threshold][category] = ap
        
        # Calculate mAP for each IoU threshold
        map_results = {}
        for threshold in iou_thresholds:
            aps = list(ap_results[threshold].values())
            map_results[threshold] = np.mean(aps) if aps else 0.0
        
        overall_map = np.mean(list(map_results.values()))
        
        return {
            'mAP': overall_map,
            'per_class': ap_results,
            'AP50': map_results.get(0.5, 0.0),
            'AP75': map_results.get(0.75, 0.0)
        }
    
    def parse_bbox_from_string(self, box_str: str) -> List[int]:
        """Parse bounding box coordinates from string format."""
        return [int(num.strip()) for pair in box_str.strip("()").split("),(") for num in pair.split(",")]
    
    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run grounding evaluation."""
        exp_name, checkpoint = self.get_experiment_info()
        save_dir = self.get_save_dir(exp_name, checkpoint, self.get_next_eval_count(
            os.path.join(self.output_path, exp_name, checkpoint), 'grounding.json'
        ))
        self.create_directories(save_dir)
        
        # Load data
        data = self.load_data(self.data_path)
        
        # Initialize collections
        predictions = []
        ground_truths = []
        sum_score_json_format = 0
        sum_score_box_format = 0
        total_boxes = 0
        total_samples = 0
        
        iou_results = {'mIoU50': 0, 'mIoU75': 0, 'cnt': 0}
        
        # Process data in batches
        results_dict, eval_data = self.run_batch_inference(
            data, save_dir, 'grounding.jsonl', is_image_based=True
        )
        
        for idx, info in data.iterrows():
            total_samples += 1
            answer_boxes = json.loads(info.answer)
            iou_results['cnt'] += 1
            
            tmp_gt_box = []
            for box in answer_boxes:
                try:
                    box_coord = self.parse_bbox_from_string(box["box"])
                    ground_truths.append({
                        'image_id': info.image_path,
                        'category_id': box['type'],
                        'bbox': box_coord,
                        'score': 1
                    })
                    tmp_gt_box.append(box_coord)
                except Exception as e:
                    print(f"Parse box failed: {e}")
            
            # Process predictions
            prediction = results_dict[idx]['model_output']
            try:
                prediction_boxes = json.loads(prediction)
                sum_score_json_format += 1
            except:
                matches = re.findall(r'\{[^{}]*\}', prediction)
                unique_matches = set(matches)
                prediction_boxes = []
                for match in unique_matches:
                    try:
                        item = json.loads(match)
                        prediction_boxes.append(item)
                    except:
                        continue
                if prediction_boxes:
                    sum_score_json_format += 1
            
            tmp_pred_boxes = []
            for box in prediction_boxes:
                try:
                    total_boxes += 1
                    if 'box' in box:
                        box_coord = list(map(int, re.findall(r'\d+', box['box'])))
                    else:
                        box_coord = list(map(int, re.findall(r'\d+', box['bbox_2d'])))
                    
                    assert len(box_coord) == 4
                    tmp_pred_boxes.append(box_coord)
                    predictions.append({
                        'image_id': info.image_path,
                        'category_id': box['type'],
                        'bbox': box_coord,
                        'score': 1
                    })
                    sum_score_box_format += 1
                except Exception as e:
                    print(f"Parse box failed: {e}")
            
            # Calculate IoU results
            tmp_results = self.match_boxes_hungarian(tmp_pred_boxes, tmp_gt_box)
            for key, value in tmp_results.items():
                iou_results[key] += value

        # Evaluate overall results
        results = self.grounding_evaluate(predictions, ground_truths)
        
        # Normalize IoU results
        iou_results['mIoU50'] /= max(iou_results['cnt'], 1)
        iou_results['mIoU75'] /= max(iou_results['cnt'], 1)
        
        # Create final metrics
        ret = {
            "grounding_json_format": sum_score_json_format / max(total_samples, 1),
            "grounding_box_format": sum_score_box_format / max(total_boxes, 1),
            "mAP": results['mAP'],
            "AP50": results['AP50'],
            "AP75": results['AP75'],
            "mIoU50": iou_results['mIoU50'],
            "mIoU75": iou_results['mIoU75'],
            "Per Class AP": results['per_class']
        }
        
        # Save results
        self.save_json(ret, os.path.join(save_dir, 'grounding.json'))
        
        print("\n评测结果：")
        print(f"total_samples: {total_samples}")
        print(f"mAP: {results['mAP']}")
        print(f"AP50: {results['AP50']}")
        print(f"AP75: {results['AP75']}")
        print(f"mIoU50: {iou_results['mIoU50']}")
        print(f"mIoU75: {iou_results['mIoU75']}")
        
        return ret


if __name__ == '__main__':
    evaluator = GroundingEvaluator()
    results = evaluator.run_evaluation()
    print("Grounding evaluation completed:", results)