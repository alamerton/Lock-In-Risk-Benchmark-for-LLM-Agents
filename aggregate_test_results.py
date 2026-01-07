"""
Aggregate Test Results by Agent Model

This script aggregates test results from test/results directory
Structure: category/model/dataset/variant_type/
Aggregation: by model → category → dataset/variant
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


class TestResultsAggregator:
    """Aggregates test results organized by model"""
    
    def __init__(self, base_dir: str = "test/results"):
        self.base_dir = Path(base_dir)
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        
    def find_all_summary_files(self) -> List[Path]:
        """Find all summary JSON files"""
        summary_files = []
        for json_file in self.base_dir.rglob("*_summary.json"):
            summary_files.append(json_file)
        return summary_files
    
    def load_summary(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load a summary file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add file path info
                data['_source_file'] = str(filepath)
                data['_relative_path'] = str(filepath.relative_to(self.base_dir))
                return data
        except Exception as e:
            print(f"  Warning: Could not load {filepath}: {e}")
            return None
    
    def extract_metadata_from_path(self, filepath: Path) -> Dict[str, str]:
        """Extract category, model, dataset, variant from path"""
        relative = filepath.relative_to(self.base_dir)
        parts = relative.parts
        
        info = {
            "category": "unknown",
            "model": "unknown",
            "dataset": "unknown",
            "variant_type": "unknown"
        }
        
        # Structure: category/model/dataset/variant_type/file.json
        if len(parts) >= 5:
            info["category"] = parts[0]
            info["model"] = parts[1]
            info["dataset"] = parts[2]
            info["variant_type"] = parts[3]
        
        return info
    
    def load_all_summaries(self) -> int:
        """Load all summary files"""
        summary_files = self.find_all_summary_files()
        print(f"Found {len(summary_files)} summary files")
        
        loaded = 0
        for filepath in summary_files:
            summary = self.load_summary(filepath)
            if summary:
                # Add metadata from path
                metadata = self.extract_metadata_from_path(filepath)
                summary.update({
                    "_category": metadata["category"],
                    "_model": metadata["model"],
                    "_dataset": metadata["dataset"],
                    "_variant_type": metadata["variant_type"]
                })
                self.results.append(summary)
                loaded += 1
        
        print(f"Successfully loaded {loaded} summaries")
        return loaded
    
    def aggregate_by_model(self) -> Dict[str, Any]:
        """Aggregate results by model → category → dataset/variant"""
        if not self.results:
            print("No results to aggregate")
            return {}
        
        # Group by model
        by_model = defaultdict(lambda: {
            "total_scenarios": 0,
            "by_category": defaultdict(lambda: {
                "total_scenarios": 0,
                "by_variant": {},
                "by_dataset": {},
                "risk_statistics": {"scores": []},
                "risk_levels": defaultdict(int)
            })
        })
        
        for result in self.results:
            model = result.get("_model", "unknown")
            category = result.get("_category", "unknown")
            dataset = result.get("_dataset", "unknown")
            variant = result.get("_variant_type", "unknown")
            
            total = result.get("total_scenarios", 0)
            
            # Extract risk statistics
            risk_stats = result.get("risk_statistics", {})
            avg_risk = risk_stats.get("avg_risk_score", 0)
            max_risk = risk_stats.get("max_risk_score", 0)
            min_risk = risk_stats.get("min_risk_score", 0)
            
            # Risk levels
            risk_levels = result.get("by_risk_level", {})
            
            # Model level
            by_model[model]["total_scenarios"] += total
            
            # Category level
            cat_data = by_model[model]["by_category"][category]
            cat_data["total_scenarios"] += total
            
            if avg_risk:
                cat_data["risk_statistics"]["scores"].append({
                    "avg": avg_risk,
                    "max": max_risk,
                    "min": min_risk,
                    "count": total
                })
            
            for level, count in risk_levels.items():
                cat_data["risk_levels"][level] += count
            
            # By variant
            variant_key = variant
            if variant_key not in cat_data["by_variant"]:
                cat_data["by_variant"][variant_key] = {
                    "count": 0,
                    "scores": []
                }
            cat_data["by_variant"][variant_key]["count"] += total
            if avg_risk:
                cat_data["by_variant"][variant_key]["scores"].append(avg_risk)
            
            # By dataset
            dataset_key = dataset
            if dataset_key not in cat_data["by_dataset"]:
                cat_data["by_dataset"][dataset_key] = {
                    "count": 0,
                    "by_variant": {},
                    "scores": []
                }
            cat_data["by_dataset"][dataset_key]["count"] += total
            if avg_risk:
                cat_data["by_dataset"][dataset_key]["scores"].append(avg_risk)
            
            # Dataset x Variant - Fixed to properly aggregate multiple occurrences
            if variant_key not in cat_data["by_dataset"][dataset_key]["by_variant"]:
                cat_data["by_dataset"][dataset_key]["by_variant"][variant_key] = {
                    "count": total,  # Initialize with total from first occurrence
                    "scores": [],
                    "max_scores": [],
                    "min_scores": []
                }
            else:
                cat_data["by_dataset"][dataset_key]["by_variant"][variant_key]["count"] += total
            
            # Track scores from ALL occurrences (first and subsequent)
            if avg_risk:
                cat_data["by_dataset"][dataset_key]["by_variant"][variant_key]["scores"].append(avg_risk)
            if max_risk:
                cat_data["by_dataset"][dataset_key]["by_variant"][variant_key]["max_scores"].append(max_risk)
            if min_risk:
                cat_data["by_dataset"][dataset_key]["by_variant"][variant_key]["min_scores"].append(min_risk)
        
        # Calculate statistics
        for model, model_data in by_model.items():
            for category, cat_data in model_data["by_category"].items():
                # Calculate category-level statistics
                scores_data = cat_data["risk_statistics"]["scores"]
                if scores_data:
                    all_avgs = [s["avg"] for s in scores_data]
                    all_maxs = [s["max"] for s in scores_data]
                    all_mins = [s["min"] for s in scores_data]
                    
                    cat_data["risk_statistics"] = {
                        "avg_risk_score": round(sum(all_avgs) / len(all_avgs), 4),
                        "max_risk_score": round(max(all_maxs), 4),
                        "min_risk_score": round(min(all_mins), 4),
                        "scenarios_count": cat_data["total_scenarios"]
                    }
                
                # Calculate variant statistics
                for variant, variant_data in cat_data["by_variant"].items():
                    if variant_data["scores"]:
                        variant_data["avg_risk_score"] = round(
                            sum(variant_data["scores"]) / len(variant_data["scores"]), 4
                        )
                        variant_data["max_risk_score"] = round(max(variant_data["scores"]), 4)
                        variant_data["min_risk_score"] = round(min(variant_data["scores"]), 4)
                        del variant_data["scores"]
                
                # Calculate dataset statistics
                for dataset, dataset_data in cat_data["by_dataset"].items():
                    if dataset_data["scores"]:
                        dataset_data["avg_risk_score"] = round(
                            sum(dataset_data["scores"]) / len(dataset_data["scores"]), 4
                        )
                        dataset_data["max_risk_score"] = round(max(dataset_data["scores"]), 4)
                        dataset_data["min_risk_score"] = round(min(dataset_data["scores"]), 4)
                        del dataset_data["scores"]
                    
                    # Calculate dataset x variant statistics from accumulated scores
                    for variant, variant_data in dataset_data.get("by_variant", {}).items():
                        if variant_data.get("scores"):
                            variant_data["avg_risk"] = round(
                                sum(variant_data["scores"]) / len(variant_data["scores"]), 4
                            )
                            del variant_data["scores"]
                        if variant_data.get("max_scores"):
                            variant_data["max_risk"] = round(max(variant_data["max_scores"]), 4)
                            del variant_data["max_scores"]
                        if variant_data.get("min_scores"):
                            variant_data["min_risk"] = round(min(variant_data["min_scores"]), 4)
                            del variant_data["min_scores"]
                
                # Convert defaultdicts to regular dicts
                cat_data["risk_levels"] = dict(cat_data["risk_levels"])
        
        # Convert to regular dicts
        result = {}
        for model, model_data in by_model.items():
            result[model] = {
                "total_scenarios": model_data["total_scenarios"],
                "by_category": {}
            }
            for category, cat_data in model_data["by_category"].items():
                result[model]["by_category"][category] = dict(cat_data)
        
        self.summary = {
            "generated_at": datetime.now().isoformat(),
            "by_model": result
        }
        
        return self.summary
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown report"""
        if not self.summary:
            return "No summary available"
        
        md = []
        md.append("# Test Results Aggregation by Agent Model\n")
        md.append(f"**Generated:** {self.summary['generated_at']}\n")
        md.append("---\n")
        
        for model, model_data in self.summary.get("by_model", {}).items():
            md.append(f"## Model: `{model}`\n")
            md.append(f"**Total Scenarios Tested:** {model_data['total_scenarios']}\n")
            
            for category, cat_data in model_data.get("by_category", {}).items():
                md.append(f"\n### Category: {category}\n")
                
                # Overall statistics
                risk_stats = cat_data.get("risk_statistics", {})
                if risk_stats:
                    md.append(f"**Scenarios:** {risk_stats.get('scenarios_count', 0)}  ")
                    md.append(f"**Avg Risk:** {risk_stats.get('avg_risk_score', 0):.4f}  ")
                    md.append(f"**Max Risk:** {risk_stats.get('max_risk_score', 0):.4f}  ")
                    md.append(f"**Min Risk:** {risk_stats.get('min_risk_score', 0):.4f}\n")
                
                # Risk level distribution
                risk_levels = cat_data.get("risk_levels", {})
                if risk_levels:
                    md.append("\n**Risk Level Distribution:**\n")
                    md.append("| Risk Level | Count |\n")
                    md.append("|------------|-------|\n")
                    for level in sorted(risk_levels.keys()):
                        md.append(f"| {level} | {risk_levels[level]} |\n")
                
                # By Variant Type
                by_variant = cat_data.get("by_variant", {})
                if by_variant:
                    md.append("\n**By Variant Type:**\n")
                    md.append("| Variant | Count | Avg Risk | Max Risk | Min Risk |\n")
                    md.append("|---------|-------|----------|----------|----------|\n")
                    for variant in sorted(by_variant.keys()):
                        v_data = by_variant[variant]
                        md.append(f"| {variant} | {v_data['count']} | "
                                f"{v_data.get('avg_risk_score', 0):.4f} | "
                                f"{v_data.get('max_risk_score', 0):.4f} | "
                                f"{v_data.get('min_risk_score', 0):.4f} |\n")
                
                # By Dataset
                by_dataset = cat_data.get("by_dataset", {})
                if by_dataset:
                    md.append("\n**By Dataset:**\n")
                    md.append("| Dataset | Count | Avg Risk | Max Risk | Min Risk |\n")
                    md.append("|---------|-------|----------|----------|----------|\n")
                    for dataset in sorted(by_dataset.keys()):
                        d_data = by_dataset[dataset]
                        md.append(f"| {dataset} | {d_data['count']} | "
                                f"{d_data.get('avg_risk_score', 0):.4f} | "
                                f"{d_data.get('max_risk_score', 0):.4f} | "
                                f"{d_data.get('min_risk_score', 0):.4f} |\n")
                    
                    # Detailed breakdown: Dataset x Variant
                    md.append("\n**Detailed Breakdown (Dataset × Variant):**\n")
                    for dataset in sorted(by_dataset.keys()):
                        d_data = by_dataset[dataset]
                        if "by_variant" in d_data and d_data["by_variant"]:
                            md.append(f"\n*{dataset}:*\n")
                            md.append("| Variant | Count | Avg Risk | Max Risk | Min Risk |\n")
                            md.append("|---------|-------|----------|----------|----------|\n")
                            for variant in sorted(d_data["by_variant"].keys()):
                                v_data = d_data["by_variant"][variant]
                                md.append(f"| {variant} | {v_data.get('count', 0)} | "
                                        f"{v_data.get('avg_risk', 0):.4f} | "
                                        f"{v_data.get('max_risk', 0):.4f} | "
                                        f"{v_data.get('min_risk', 0):.4f} |\n")
                
                md.append("\n---\n")
        
        return "".join(md)
    
    def save_json(self, output_path: str = "test_results_aggregated.json"):
        """Save aggregated results as JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2)
        print(f"JSON saved to: {output_path}")
    
    def save_markdown(self, output_path: str = "TEST_RESULTS_SUMMARY.md"):
        """Save markdown report"""
        md_content = self.generate_markdown_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown saved to: {output_path}")
    
    def print_summary(self):
        """Print summary to console"""
        print("\n" + "="*80)
        print("TEST RESULTS AGGREGATION BY MODEL")
        print("="*80)
        
        for model, model_data in self.summary.get("by_model", {}).items():
            print(f"\n{'='*60}")
            print(f"MODEL: {model}")
            print(f"Total Scenarios: {model_data['total_scenarios']}")
            print(f"{'='*60}")
            
            for category, cat_data in model_data.get("by_category", {}).items():
                print(f"\n  Category: {category}")
                print(f"  {'-'*56}")
                
                risk_stats = cat_data.get("risk_statistics", {})
                if risk_stats:
                    print(f"  Scenarios: {risk_stats.get('scenarios_count', 0)}")
                    print(f"  Avg Risk:  {risk_stats.get('avg_risk_score', 0):.4f}")
                    print(f"  Max Risk:  {risk_stats.get('max_risk_score', 0):.4f}")
                    print(f"  Min Risk:  {risk_stats.get('min_risk_score', 0):.4f}")
                
                print(f"\n  By Variant:")
                for variant, v_data in cat_data.get("by_variant", {}).items():
                    print(f"    {variant:8} - count: {v_data['count']:3}, "
                          f"avg: {v_data.get('avg_risk_score', 0):.4f}")
                
                print(f"\n  By Dataset:")
                for dataset, d_data in cat_data.get("by_dataset", {}).items():
                    print(f"    {dataset:12} - count: {d_data['count']:3}, "
                          f"avg: {d_data.get('avg_risk_score', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate test results by model")
    parser.add_argument(
        "--base-dir", "-d",
        default="test/results",
        help="Base directory (default: test/results)"
    )
    parser.add_argument(
        "--output-json", "-j",
        default="test_results_aggregated.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--output-md", "-m",
        default="TEST_RESULTS_SUMMARY.md",
        help="Output Markdown file"
    )
    
    args = parser.parse_args()
    
    print(f"Aggregating results from: {args.base_dir}")
    print("="*60)
    
    aggregator = TestResultsAggregator(args.base_dir)
    
    # Load summaries
    loaded = aggregator.load_all_summaries()
    if loaded == 0:
        print("No summaries found")
        return
    
    # Aggregate
    print("\nAggregating...")
    aggregator.aggregate_by_model()
    
    # Print summary
    aggregator.print_summary()
    
    # Save outputs
    aggregator.save_json(args.output_json)
    aggregator.save_markdown(args.output_md)
    
    print("\n[OK] Aggregation complete!")


if __name__ == "__main__":
    main()
