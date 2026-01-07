"""
Aggregate Phase 1 Benchmark Results

This script aggregates all phase1 evaluation results from the phase1_results folder.
Organized by: Model → Category → Statistics
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any
import statistics


class Phase1ResultsAggregator:
    """Aggregates phase1 benchmark results across models and categories"""
    
    def __init__(self, results_dir: str = "phase1_results"):
        self.results_dir = Path(results_dir)
        self.results = []
        self.summary = {
            "aggregation_timestamp": datetime.now().isoformat(),
            "results_directory": str(self.results_dir),
            "by_model": {},
            "overall_statistics": {},
            "comparison_table": []
        }
    
    def load_all_summaries(self) -> int:
        """Load all summary.json files from the results directory"""
        loaded = 0
        
        print(f"Scanning {self.results_dir} for results...")
        
        # Iterate through model directories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            print(f"\n  Model: {model_name}")
            
            # Iterate through category directories
            for category_dir in model_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                
                category_name = category_dir.name
                
                # Find the most recent summary file
                summary_files = list(category_dir.glob("*_summary.json"))
                if not summary_files:
                    print(f"    [WARNING] {category_name}: No summary file found")
                    continue
                
                # Use the most recent summary file
                summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
                
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add metadata
                    data["_model"] = model_name
                    data["_category"] = category_name
                    data["_file"] = str(summary_file)
                    
                    self.results.append(data)
                    loaded += 1
                    
                    total = data.get("total_scenarios", 0)
                    avg_risk = data.get("risk_statistics", {}).get("avg_risk_score", "N/A")
                    print(f"    [OK] {category_name}: {total} scenarios, avg risk: {avg_risk}")
                    
                except Exception as e:
                    print(f"    [ERROR] {category_name}: Error loading {summary_file.name}: {e}")
        
        print(f"\n[OK] Loaded {loaded} result summaries")
        return loaded
    
    def aggregate(self) -> Dict[str, Any]:
        """Generate comprehensive aggregated statistics"""
        
        if not self.results:
            print("[WARNING] No results to aggregate")
            return self.summary
        
        # Track all scores for overall statistics
        all_scores = []
        model_scores = defaultdict(list)
        category_scores = defaultdict(list)
        model_category_scores = defaultdict(lambda: defaultdict(list))
        
        # Track scenario counts
        model_scenario_counts = defaultdict(int)
        category_scenario_counts = defaultdict(int)
        
        # Track risk level distributions
        model_risk_levels = defaultdict(lambda: defaultdict(int))
        category_risk_levels = defaultdict(lambda: defaultdict(int))
        
        # Process each result
        for result in self.results:
            model = result["_model"]
            category = result["_category"]
            
            # Get scenario count
            total_scenarios = result.get("total_scenarios", 0)
            model_scenario_counts[model] += total_scenarios
            category_scenario_counts[category] += total_scenarios
            
            # Get risk statistics
            risk_stats = result.get("risk_statistics", {})
            avg_risk = risk_stats.get("avg_risk_score")
            max_risk = risk_stats.get("max_risk_score")
            min_risk = risk_stats.get("min_risk_score")
            
            # Get individual scenario scores if available
            scenario_results = result.get("scenario_results", [])
            for scenario in scenario_results:
                score = self._extract_score(scenario)
                if score is not None:
                    all_scores.append(score)
                    model_scores[model].append(score)
                    category_scores[category].append(score)
                    model_category_scores[model][category].append(score)
            
            # If no individual scores, use aggregate statistics
            if not scenario_results and avg_risk is not None:
                # Use avg_risk as representative for overall aggregation
                all_scores.append(avg_risk)
                model_scores[model].append(avg_risk)
                category_scores[category].append(avg_risk)
                model_category_scores[model][category].append(avg_risk)
            
            # Risk level distribution
            by_risk_level = result.get("by_risk_level", {})
            for level, count in by_risk_level.items():
                model_risk_levels[model][level] += count
                category_risk_levels[category][level] += count
        
        # Calculate overall statistics
        if all_scores:
            self.summary["overall_statistics"] = {
                "total_scenarios": sum(model_scenario_counts.values()),
                "total_models_tested": len(model_scores),
                "total_categories": len(category_scores),
                "avg_risk_score": round(statistics.mean(all_scores), 4),
                "max_risk_score": round(max(all_scores), 4),
                "min_risk_score": round(min(all_scores), 4),
                "median_risk_score": round(statistics.median(all_scores), 4),
                "std_dev_risk_score": round(statistics.stdev(all_scores), 4) if len(all_scores) > 1 else 0,
                "score_range": round(max(all_scores) - min(all_scores), 4)
            }
        
        # Aggregate by model
        for model in sorted(model_scores.keys()):
            scores = model_scores[model]
            
            model_data = {
                "total_scenarios": model_scenario_counts[model],
                "overall_statistics": self._calculate_stats(scores),
                "by_category": {},
                "risk_level_distribution": dict(model_risk_levels[model])
            }
            
            # Add per-category breakdown for this model
            for category in sorted(model_category_scores[model].keys()):
                cat_scores = model_category_scores[model][category]
                model_data["by_category"][category] = self._calculate_stats(cat_scores)
            
            self.summary["by_model"][model] = model_data
        
        # Create comparison table
        self._create_comparison_table(model_category_scores)
        
        return self.summary
    
    def _extract_score(self, scenario_result: Dict[str, Any]) -> float:
        """Extract risk score from various result formats"""
        # Try different score locations
        if "risk_score" in scenario_result:
            return scenario_result["risk_score"]
        
        if "detailed_assessment" in scenario_result:
            detail = scenario_result["detailed_assessment"]
            if "risk_score" in detail:
                return detail["risk_score"]
        
        if "result" in scenario_result:
            result = scenario_result["result"]
            if isinstance(result, list) and len(result) > 0:
                first_result = result[0]
                if "metadata" in first_result:
                    metadata = first_result["metadata"]
                    if "lock_in_evaluation" in metadata:
                        eval_data = metadata["lock_in_evaluation"]
                        if "metrics" in eval_data:
                            metrics = eval_data["metrics"]
                            if "primary_score" in metrics:
                                return metrics["primary_score"]
        
        return None
    
    def _calculate_stats(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of scores"""
        if not scores:
            return {
                "count": 0,
                "avg": 0,
                "max": 0,
                "min": 0,
                "median": 0,
                "std_dev": 0
            }
        
        return {
            "count": len(scores),
            "avg": round(statistics.mean(scores), 4),
            "max": round(max(scores), 4),
            "min": round(min(scores), 4),
            "median": round(statistics.median(scores), 4),
            "std_dev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0
        }
    
    def _create_comparison_table(self, model_category_scores: Dict):
        """Create a comparison table for easy model-category comparison"""
        
        # Get all categories
        all_categories = set()
        for model in model_category_scores:
            all_categories.update(model_category_scores[model].keys())
        
        # Build comparison rows
        for model in sorted(model_category_scores.keys()):
            row = {"model": model}
            
            for category in sorted(all_categories):
                scores = model_category_scores[model].get(category, [])
                if scores:
                    row[f"{category}_avg"] = round(statistics.mean(scores), 4)
                    row[f"{category}_count"] = len(scores)
                else:
                    row[f"{category}_avg"] = None
                    row[f"{category}_count"] = 0
            
            self.summary["comparison_table"].append(row)
    
    def save_json(self, output_file: str = "phase1_aggregated_results.json"):
        """Save aggregated results as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved JSON results to: {output_file}")
    
    def generate_markdown_report(self, output_file: str = "PHASE1_RESULTS_SUMMARY.md"):
        """Generate a human-readable markdown report"""
        
        lines = []
        lines.append("# Phase 1 Benchmark Results Summary")
        lines.append(f"\n**Generated:** {self.summary['aggregation_timestamp']}")
        lines.append(f"**Source:** `{self.summary['results_directory']}`")
        lines.append("\n---\n")
        
        # Overall Statistics
        overall = self.summary.get("overall_statistics", {})
        if overall:
            lines.append("## Overall Statistics")
            lines.append(f"- **Total Scenarios Evaluated:** {overall.get('total_scenarios', 0)}")
            lines.append(f"- **Models Tested:** {overall.get('total_models_tested', 0)}")
            lines.append(f"- **Categories:** {overall.get('total_categories', 0)}")
            lines.append(f"- **Average Risk Score:** {overall.get('avg_risk_score', 0)}")
            lines.append(f"- **Risk Score Range:** {overall.get('min_risk_score', 0)} - {overall.get('max_risk_score', 0)}")
            lines.append(f"- **Median Risk Score:** {overall.get('median_risk_score', 0)}")
            lines.append(f"- **Standard Deviation:** {overall.get('std_dev_risk_score', 0)}")
            lines.append("\n---\n")
        
        # Model Comparison Table
        lines.append("## Model Performance Comparison\n")
        comparison = self.summary.get("comparison_table", [])
        
        if comparison:
            # Create markdown table
            # Header
            categories = []
            for key in comparison[0].keys():
                if key != "model" and key.endswith("_avg"):
                    categories.append(key.replace("_avg", ""))
            
            lines.append("| Model | " + " | ".join([f"{cat.replace('_', ' ').title()} (Avg)" for cat in categories]) + " |")
            lines.append("|-------|" + "|".join(["-------" for _ in categories]) + "|")
            
            # Data rows
            for row in comparison:
                model = row["model"]
                values = []
                for cat in categories:
                    avg = row.get(f"{cat}_avg")
                    count = row.get(f"{cat}_count", 0)
                    if avg is not None:
                        values.append(f"{avg:.4f} (n={count})")
                    else:
                        values.append("N/A")
                
                lines.append(f"| {model} | " + " | ".join(values) + " |")
            
            lines.append("")
        
        # Detailed Model Results
        lines.append("\n---\n## Detailed Results by Model\n")
        
        for model, model_data in sorted(self.summary.get("by_model", {}).items()):
            lines.append(f"### {model}")
            lines.append(f"\n**Total Scenarios:** {model_data.get('total_scenarios', 0)}")
            
            overall_stats = model_data.get("overall_statistics", {})
            if overall_stats:
                lines.append(f"- **Average Risk Score:** {overall_stats.get('avg', 0)}")
                lines.append(f"- **Risk Range:** {overall_stats.get('min', 0)} - {overall_stats.get('max', 0)}")
                lines.append(f"- **Median:** {overall_stats.get('median', 0)}")
                lines.append(f"- **Std Dev:** {overall_stats.get('std_dev', 0)}")
            
            # Risk level distribution
            risk_dist = model_data.get("risk_level_distribution", {})
            if risk_dist:
                lines.append("\n**Risk Level Distribution:**")
                lines.append("| Risk Level | Count |")
                lines.append("|------------|-------|")
                for level in sorted(risk_dist.keys()):
                    lines.append(f"| {level} | {risk_dist[level]} |")
            
            # By category
            by_category = model_data.get("by_category", {})
            if by_category:
                lines.append("\n**By Category:**")
                lines.append("| Category | Count | Avg Risk | Max Risk | Min Risk | Median |")
                lines.append("|----------|-------|----------|----------|----------|--------|")
                
                for category, stats in sorted(by_category.items()):
                    lines.append(
                        f"| {category.replace('_', ' ').title()} | "
                        f"{stats.get('count', 0)} | "
                        f"{stats.get('avg', 0):.4f} | "
                        f"{stats.get('max', 0):.4f} | "
                        f"{stats.get('min', 0):.4f} | "
                        f"{stats.get('median', 0):.4f} |"
                    )
            
            lines.append("\n---\n")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"[OK] Saved Markdown report to: {output_file}")
    
    def print_summary(self):
        """Print a quick summary to console"""
        print("\n" + "="*70)
        print("PHASE 1 RESULTS SUMMARY")
        print("="*70)
        
        overall = self.summary.get("overall_statistics", {})
        print(f"\nTotal Scenarios: {overall.get('total_scenarios', 0)}")
        print(f"Models Tested: {overall.get('total_models_tested', 0)}")
        print(f"Average Risk Score: {overall.get('avg_risk_score', 0):.4f}")
        print(f"Risk Score Range: {overall.get('min_risk_score', 0):.4f} - {overall.get('max_risk_score', 0):.4f}")
        
        print("\n" + "-"*70)
        print("MODEL COMPARISON")
        print("-"*70)
        
        for row in self.summary.get("comparison_table", []):
            model = row["model"]
            print(f"\n{model}:")
            
            for key, value in row.items():
                if key != "model" and key.endswith("_avg") and value is not None:
                    category = key.replace("_avg", "").replace("_", " ").title()
                    count_key = key.replace("_avg", "_count")
                    count = row.get(count_key, 0)
                    print(f"  {category:25s}: {value:.4f} (n={count})")
        
        print("\n" + "="*70)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 benchmark results")
    parser.add_argument(
        "--results-dir",
        default="phase1_results",
        help="Directory containing phase1 results (default: phase1_results)"
    )
    parser.add_argument(
        "--output-json",
        default="phase1_aggregated_results.json",
        help="Output JSON file (default: phase1_aggregated_results.json)"
    )
    parser.add_argument(
        "--output-markdown",
        default="PHASE1_RESULTS_SUMMARY.md",
        help="Output Markdown file (default: PHASE1_RESULTS_SUMMARY.md)"
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Don't print summary to console"
    )
    
    args = parser.parse_args()
    
    # Create aggregator
    aggregator = Phase1ResultsAggregator(args.results_dir)
    
    # Load results
    loaded = aggregator.load_all_summaries()
    
    if loaded == 0:
        print("\n[WARNING] No results found to aggregate.")
        return
    
    # Aggregate
    print("\nAggregating results...")
    aggregator.aggregate()
    
    # Save outputs
    aggregator.save_json(args.output_json)
    aggregator.generate_markdown_report(args.output_markdown)
    
    # Print summary
    if not args.no_console:
        aggregator.print_summary()
    
    print("\n[OK] Aggregation complete!")


if __name__ == "__main__":
    main()

