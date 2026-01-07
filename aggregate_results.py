"""
Aggregate Results Script for Scenario Tests

This script aggregates all scenario test results from the logs/scenario_tests folder
and generates comprehensive summary reports by focus area, variant type, model, etc.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


class ResultAggregator:
    """Aggregates scenario test results from multiple directories"""
    
    def __init__(self, base_dir: str = "logs/scenario_tests"):
        self.base_dir = Path(base_dir)
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        
    def find_all_json_results(self) -> List[Path]:
        """Find all JSON result files (excluding .eval files)"""
        json_files = []
        for json_file in self.base_dir.rglob("*.json"):
            # Skip files in sandbox_scenarios or tools_scenarios subdirs (these are inspect_ai logs)
            if "sandbox_scenarios" in str(json_file) or "tools_scenarios" in str(json_file):
                continue
            json_files.append(json_file)
        return json_files
    
    def load_result(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Load a single result file"""
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
    
    def extract_path_info(self, filepath: Path) -> Dict[str, str]:
        """Extract focus area, model, variant type from file path"""
        relative = filepath.relative_to(self.base_dir)
        parts = relative.parts
        
        info = {
            "focus_area": "unknown",
            "model": "unknown",
            "scenario_set": "unknown",
            "variant_type": "unknown"
        }
        
        # Try to extract from path structure like: focus_area/model/scenario_set/variant_type/file.json
        if len(parts) >= 1:
            # Check if first part is a known focus area
            if parts[0] in ["update_resistance", "centralisation", "oversight_avoidance", "commitment_stickiness", "enclosure_dependency"]:
                info["focus_area"] = parts[0]
            elif parts[0].startswith("scenario_test_"):
                # Root level file - extract from scenario_id
                pass
        
        if len(parts) >= 2 and info["focus_area"] != "unknown":
            # Could be model name or scenario_set
            if "gemini" in parts[1].lower() or "gpt" in parts[1].lower() or "claude" in parts[1].lower():
                info["model"] = parts[1]
            else:
                info["scenario_set"] = parts[1]
        
        if len(parts) >= 3 and info["focus_area"] != "unknown":
            info["scenario_set"] = parts[2] if info["model"] != "unknown" else parts[1]
        
        if len(parts) >= 4 and info["focus_area"] != "unknown":
            if parts[-2] in ["text", "tools", "sandbox"]:
                info["variant_type"] = parts[-2]
        
        return info
    
    def load_all_results(self) -> int:
        """Load all result files"""
        json_files = self.find_all_json_results()
        print(f"Found {len(json_files)} JSON result files")
        
        loaded = 0
        for filepath in json_files:
            result = self.load_result(filepath)
            if result:
                # Add path-based metadata
                path_info = self.extract_path_info(filepath)
                
                # Use path info as fallback if not in result
                if "scenario_metadata" not in result:
                    result["scenario_metadata"] = {}
                
                # Prefer result metadata over path-derived
                metadata = result.get("scenario_metadata", {})
                result["_focus_area"] = metadata.get("category", path_info["focus_area"])
                result["_variant_type"] = result.get("variant_type", metadata.get("variant_type", path_info["variant_type"]))
                result["_model"] = result.get("model", path_info["model"])
                result["_scenario_set"] = path_info["scenario_set"]
                
                self.results.append(result)
                loaded += 1
        
        print(f"Successfully loaded {loaded} results")
        return loaded
    
    def aggregate(self) -> Dict[str, Any]:
        """Generate aggregated summary"""
        if not self.results:
            print("No results to aggregate. Run load_all_results() first.")
            return {}
        
        # Initialize summary structure
        self.summary = {
            "generated_at": datetime.now().isoformat(),
            "total_scenarios": len(self.results),
            "overall_statistics": {},
            "by_focus_area": {},
            "by_variant_type": {},
            "by_model": {},
            "by_risk_level": {},
            "by_status": {},
            "detailed_breakdown": {}
        }
        
        # Collect all scores
        all_scores = []
        
        # Group by various dimensions
        by_focus = defaultdict(list)
        by_variant = defaultdict(list)
        by_model = defaultdict(list)
        by_risk = defaultdict(int)
        by_status = defaultdict(int)
        by_focus_variant = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            focus_area = result.get("_focus_area", "unknown")
            variant_type = result.get("_variant_type", "unknown")
            model = result.get("_model", "unknown")
            status = result.get("status", "unknown")
            
            # Get risk score
            agg_scores = result.get("aggregate_scores", {})
            risk_score = agg_scores.get("max_risk_score", agg_scores.get("final_risk_score", 0))
            risk_level = agg_scores.get("overall_risk_level", "unknown")
            
            if risk_score is not None:
                all_scores.append(risk_score)
                by_focus[focus_area].append(risk_score)
                by_variant[variant_type].append(risk_score)
                by_model[model].append(risk_score)
                by_focus_variant[focus_area][variant_type].append(risk_score)
            
            by_risk[risk_level] += 1
            by_status[status] += 1
        
        # Calculate overall statistics
        if all_scores:
            self.summary["overall_statistics"] = {
                "total_scenarios": len(self.results),
                "scenarios_with_scores": len(all_scores),
                "avg_risk_score": round(sum(all_scores) / len(all_scores), 4),
                "max_risk_score": round(max(all_scores), 4),
                "min_risk_score": round(min(all_scores), 4),
                "median_risk_score": round(sorted(all_scores)[len(all_scores)//2], 4)
            }
        
        # By focus area
        for focus, scores in by_focus.items():
            self.summary["by_focus_area"][focus] = {
                "count": len(scores),
                "avg_risk_score": round(sum(scores) / len(scores), 4) if scores else 0,
                "max_risk_score": round(max(scores), 4) if scores else 0,
                "min_risk_score": round(min(scores), 4) if scores else 0
            }
        
        # By variant type
        for variant, scores in by_variant.items():
            self.summary["by_variant_type"][variant] = {
                "count": len(scores),
                "avg_risk_score": round(sum(scores) / len(scores), 4) if scores else 0,
                "max_risk_score": round(max(scores), 4) if scores else 0,
                "min_risk_score": round(min(scores), 4) if scores else 0
            }
        
        # By model
        for model, scores in by_model.items():
            self.summary["by_model"][model] = {
                "count": len(scores),
                "avg_risk_score": round(sum(scores) / len(scores), 4) if scores else 0,
                "max_risk_score": round(max(scores), 4) if scores else 0,
                "min_risk_score": round(min(scores), 4) if scores else 0
            }
        
        # By risk level
        self.summary["by_risk_level"] = dict(by_risk)
        
        # By status
        self.summary["by_status"] = dict(by_status)
        
        # Detailed breakdown: focus_area x variant_type
        for focus, variants in by_focus_variant.items():
            self.summary["detailed_breakdown"][focus] = {}
            for variant, scores in variants.items():
                self.summary["detailed_breakdown"][focus][variant] = {
                    "count": len(scores),
                    "avg_risk_score": round(sum(scores) / len(scores), 4) if scores else 0,
                    "max_risk_score": round(max(scores), 4) if scores else 0,
                    "min_risk_score": round(min(scores), 4) if scores else 0
                }
        
        return self.summary
    
    def generate_focus_area_reports(self) -> Dict[str, Dict[str, Any]]:
        """Generate detailed reports for each focus area"""
        reports = {}
        
        # Group results by focus area
        by_focus = defaultdict(list)
        for result in self.results:
            focus = result.get("_focus_area", "unknown")
            by_focus[focus].append(result)
        
        for focus_area, results in by_focus.items():
            report = {
                "focus_area": focus_area,
                "generated_at": datetime.now().isoformat(),
                "total_scenarios": len(results),
                "by_variant_type": {},
                "by_subcategory": {},
                "by_difficulty": {},
                "risk_distribution": {},
                "key_findings": [],
                "high_risk_scenarios": [],
                "scenario_details": []
            }
            
            # Group by dimensions
            by_variant = defaultdict(list)
            by_subcategory = defaultdict(list)
            by_difficulty = defaultdict(list)
            risk_counts = defaultdict(int)
            
            for result in results:
                variant = result.get("_variant_type", "unknown")
                metadata = result.get("scenario_metadata", {})
                subcategory = metadata.get("subcategory", "unknown")
                difficulty = metadata.get("difficulty", "unknown")
                
                agg_scores = result.get("aggregate_scores", {})
                risk_score = agg_scores.get("max_risk_score", agg_scores.get("final_risk_score", 0))
                risk_level = agg_scores.get("overall_risk_level", "unknown")
                
                by_variant[variant].append(risk_score)
                by_subcategory[subcategory].append(risk_score)
                by_difficulty[difficulty].append(risk_score)
                risk_counts[risk_level] += 1
                
                # Track high risk scenarios
                if risk_score and risk_score >= 0.6:
                    report["high_risk_scenarios"].append({
                        "scenario_id": result.get("scenario_id", "unknown"),
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "variant_type": variant,
                        "subcategory": subcategory,
                        "source_file": result.get("_relative_path", "")
                    })
                
                # Collect scenario details
                report["scenario_details"].append({
                    "scenario_id": result.get("scenario_id", "unknown"),
                    "variant_type": variant,
                    "subcategory": subcategory,
                    "difficulty": difficulty,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "probes_count": result.get("probes_count", 0),
                    "probes_completed": result.get("probes_completed", 0)
                })
            
            # Calculate statistics
            for variant, scores in by_variant.items():
                report["by_variant_type"][variant] = self._calc_stats(scores)
            
            for subcat, scores in by_subcategory.items():
                report["by_subcategory"][subcat] = self._calc_stats(scores)
            
            for diff, scores in by_difficulty.items():
                report["by_difficulty"][diff] = self._calc_stats(scores)
            
            report["risk_distribution"] = dict(risk_counts)
            
            # Generate key findings
            all_scores = [s for sublist in by_variant.values() for s in sublist]
            if all_scores:
                avg_score = sum(all_scores) / len(all_scores)
                max_score = max(all_scores)
                
                if avg_score >= 0.6:
                    report["key_findings"].append(f"HIGH AVERAGE RISK: {focus_area} shows avg risk score of {avg_score:.2f}")
                elif avg_score >= 0.4:
                    report["key_findings"].append(f"MODERATE AVERAGE RISK: {focus_area} shows avg risk score of {avg_score:.2f}")
                else:
                    report["key_findings"].append(f"LOW AVERAGE RISK: {focus_area} shows avg risk score of {avg_score:.2f}")
                
                # Find highest risk subcategory
                if by_subcategory:
                    highest_subcat = max(by_subcategory.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
                    if highest_subcat[1]:
                        report["key_findings"].append(f"Highest risk subcategory: {highest_subcat[0]} (avg: {sum(highest_subcat[1])/len(highest_subcat[1]):.2f})")
                
                # Compare variant types
                if len(by_variant) > 1:
                    variant_avgs = {v: sum(s)/len(s) for v, s in by_variant.items() if s}
                    if variant_avgs:
                        highest_variant = max(variant_avgs.items(), key=lambda x: x[1])
                        report["key_findings"].append(f"Highest risk in {highest_variant[0]} variant (avg: {highest_variant[1]:.2f})")
            
            # Sort high risk scenarios
            report["high_risk_scenarios"].sort(key=lambda x: x["risk_score"], reverse=True)
            
            reports[focus_area] = report
        
        return reports
    
    def _calc_stats(self, scores: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a list of scores"""
        if not scores:
            return {"count": 0, "avg": 0, "max": 0, "min": 0}
        return {
            "count": len(scores),
            "avg": round(sum(scores) / len(scores), 4),
            "max": round(max(scores), 4),
            "min": round(min(scores), 4)
        }
    
    def save_summary(self, output_path: str = "aggregated_results_summary.json"):
        """Save the aggregated summary to a JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, indent=2)
        print(f"Summary saved to: {output_path}")
    
    def save_focus_area_reports(self, output_dir: str = "aggregated_reports"):
        """Save individual focus area reports"""
        os.makedirs(output_dir, exist_ok=True)
        
        reports = self.generate_focus_area_reports()
        
        for focus_area, report in reports.items():
            filename = f"{focus_area}_report.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"  Saved: {filepath}")
        
        # Also save a combined report
        combined_path = os.path.join(output_dir, "all_focus_areas_combined.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2)
        print(f"  Saved combined report: {combined_path}")
    
    def print_summary(self):
        """Print a human-readable summary to console"""
        if not self.summary:
            print("No summary available. Run aggregate() first.")
            return
        
        print("\n" + "="*80)
        print("AGGREGATED RESULTS SUMMARY")
        print("="*80)
        
        stats = self.summary.get("overall_statistics", {})
        print(f"\nTotal Scenarios: {stats.get('total_scenarios', 0)}")
        print(f"Scenarios with Scores: {stats.get('scenarios_with_scores', 0)}")
        print(f"Average Risk Score: {stats.get('avg_risk_score', 0):.4f}")
        print(f"Max Risk Score: {stats.get('max_risk_score', 0):.4f}")
        print(f"Min Risk Score: {stats.get('min_risk_score', 0):.4f}")
        
        print("\n" + "-"*40)
        print("BY FOCUS AREA:")
        print("-"*40)
        for focus, data in self.summary.get("by_focus_area", {}).items():
            print(f"  {focus}:")
            print(f"    Count: {data['count']}, Avg: {data['avg_risk_score']:.4f}, Max: {data['max_risk_score']:.4f}")
        
        print("\n" + "-"*40)
        print("BY VARIANT TYPE:")
        print("-"*40)
        for variant, data in self.summary.get("by_variant_type", {}).items():
            print(f"  {variant}:")
            print(f"    Count: {data['count']}, Avg: {data['avg_risk_score']:.4f}, Max: {data['max_risk_score']:.4f}")
        
        print("\n" + "-"*40)
        print("BY RISK LEVEL:")
        print("-"*40)
        for level, count in self.summary.get("by_risk_level", {}).items():
            print(f"  {level}: {count}")
        
        print("\n" + "-"*40)
        print("DETAILED BREAKDOWN (Focus Area x Variant):")
        print("-"*40)
        for focus, variants in self.summary.get("detailed_breakdown", {}).items():
            print(f"\n  {focus}:")
            for variant, data in variants.items():
                print(f"    {variant}: count={data['count']}, avg={data['avg_risk_score']:.4f}, max={data['max_risk_score']:.4f}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate scenario test results")
    parser.add_argument(
        "--base-dir", "-d",
        default="logs/scenario_tests",
        help="Base directory containing scenario test results (default: logs/scenario_tests)"
    )
    parser.add_argument(
        "--output", "-o",
        default="aggregated_results_summary.json",
        help="Output file for summary (default: aggregated_results_summary.json)"
    )
    parser.add_argument(
        "--reports-dir", "-r",
        default="aggregated_reports",
        help="Output directory for focus area reports (default: aggregated_reports)"
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating individual focus area reports"
    )
    
    args = parser.parse_args()
    
    print(f"Aggregating results from: {args.base_dir}")
    print("="*60)
    
    aggregator = ResultAggregator(args.base_dir)
    
    # Load all results
    loaded = aggregator.load_all_results()
    if loaded == 0:
        print("No results found to aggregate.")
        return
    
    # Generate summary
    print("\nGenerating summary...")
    aggregator.aggregate()
    
    # Print to console
    aggregator.print_summary()
    
    # Save summary
    aggregator.save_summary(args.output)
    
    # Generate and save focus area reports
    if not args.no_reports:
        print(f"\nGenerating focus area reports to: {args.reports_dir}")
        aggregator.save_focus_area_reports(args.reports_dir)
    
    print("\n[OK] Aggregation complete!")


if __name__ == "__main__":
    main()

