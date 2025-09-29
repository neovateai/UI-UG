#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UI-UG Demo Runner
"""

import argparse
import sys
from typing import List

from demo.demo_runner import DemoRunner
from demo.config import SUPPORTED_TASKS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UI Understanding Demo")
    
    parser.add_argument(
        "--task", 
        type=str, 
        choices=SUPPORTED_TASKS,
        help=f"Task type: {', '.join(SUPPORTED_TASKS)}"
    )
    
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all supported tasks"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single demo mode"
    )
    
    parser.add_argument(
        "--image-url",
        type=str,
        help="Image URL for single demo"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for single demo"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode (default: True)"
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="API base URL"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="ui_ug",
        help="Model name"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if args.single and not (args.image_url and args.prompt):
        print("Error: --single mode requires both --image-url and --prompt")
        sys.exit(1)
    
    if not args.single and not args.task and not args.all_tasks:
        print("Error: Must specify --task, --all-tasks, or --single")
        sys.exit(1)
    
    # Initialize demo runner
    runner = DemoRunner(
        api_base=args.api_base,
        model_name=args.model_name
    )
    
    try:
        # Run in appropriate mode
        if args.single:
            print(f"Running single demo: {args.task}")
            result = runner.run_single_task(
                args.image_url,
                args.prompt,
                task="other",
                stream=args.stream
            )
            
            print("\n" + "="*80)
            if result["success"]:
                print("✅ Demo completed successfully")
                print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
                print(f"📄 Response: {result['response']}")
            else:
                print("❌ Demo failed")
                print(f"❗ Error: {result['error']}")
            print("="*80)
            
        elif args.all_tasks:
            print("Running all supported tasks...")
            results = runner.run_all_tasks(stream=args.stream)
            
            print("\n" + "="*80)
            print("📊 SUMMARY")
            print("="*80)
            for task, task_results in results.items():
                success_count = sum(1 for r in task_results if r["success"])
                total_count = len(task_results)
                print(f"{task}: {success_count}/{total_count} successful")
        
        else:
            print(f"Running {args.task} task...")
            from demo.config import DEMO_PROMPTS
            
            if args.task not in DEMO_PROMPTS:
                print(f"Error: No prompts configured for task '{args.task}'")
                sys.exit(1)
            
            results = runner.process_task_batch(
                DEMO_PROMPTS[args.task],
                args.task,
                stream=args.stream
            )
            
            print("\n" + "="*80)
            print("📊 SUMMARY")
            print("="*80)
            success_count = sum(1 for r in results if r["success"])
            print(f"{args.task}: {success_count}/{len(results)} successful")
            
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()