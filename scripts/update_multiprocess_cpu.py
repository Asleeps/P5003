#!/usr/bin/env python3
"""
Update parallelism_multiprocess summary files with CPU percent data from raw system.json files
"""

import json
import csv
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "parallelism"
    processed_dir = project_root / "data" / "processed"
    
    # Load existing JSON summary
    json_path = processed_dir / "parallelism_multiprocess.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Add CPU percent fields to each algorithm
    for algo_name, algo_data in data["algorithms"].items():
        if "error" in algo_data:
            continue
        
        # Initialize CPU percent dicts if not present
        if "sign_cpu_percent" not in algo_data:
            algo_data["sign_cpu_percent"] = {}
        if "verify_cpu_percent" not in algo_data:
            algo_data["verify_cpu_percent"] = {}
        
        # Extract CPU data from system.json (not worker CSV - CSV has wrong field)
        for operation in ["sign", "verify"]:
            for worker_count in data["metadata"]["thread_counts"]:
                system_file = raw_dir / f"{algo_name}_{operation}_t{worker_count}_system.json"
                
                if system_file.exists():
                    with open(system_file, "r") as f:
                        system_data = json.load(f)
                    
                    # Extract process_cpu_raw (total CPU before normalization)
                    cpu_samples = system_data.get("cpu_samples", [])
                    if cpu_samples:
                        # Get raw process tree CPU (main + all children)
                        raw_cpus = [s.get("process_cpu_raw", 0) for s in cpu_samples if "process_cpu_raw" in s]
                        if raw_cpus:
                            avg_raw_cpu = sum(raw_cpus) / len(raw_cpus)
                            # Normalize by core count (10 cores on M4)
                            avg_cpu_normalized = avg_raw_cpu / 10.0
                            cpu_key = f"{operation}_cpu_percent"
                            # Use string key to match existing JSON format
                            algo_data[cpu_key][str(worker_count)] = round(avg_cpu_normalized, 2)
    
    # Save updated JSON
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Updated {json_path}")
    
    # Regenerate CSV with CPU column
    csv_path = processed_dir / "parallelism_multiprocess_summary.csv"
    header = ["algorithm", "operation", "worker_count", "ops_per_sec", "scaling_efficiency", "p99_latency_ms", "cpu_percent"]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for algo_name, algo_data in data["algorithms"].items():
            if "error" in algo_data:
                continue
            
            for operation in ["sign", "verify"]:
                ops_key = f"{operation}_ops_per_sec"
                eff_key = f"{operation}_scaling_efficiency"
                lat_key = f"{operation}_latency_ms"
                cpu_key = f"{operation}_cpu_percent"
                
                for worker_count in data["metadata"]["thread_counts"]:
                    # Convert worker_count to string for dict lookup (JSON keys are strings)
                    wc_str = str(worker_count)
                    writer.writerow([
                        algo_name,
                        operation,
                        worker_count,
                        algo_data[ops_key].get(wc_str),
                        algo_data[eff_key].get(wc_str),
                        algo_data[lat_key].get(wc_str),
                        algo_data[cpu_key].get(wc_str),  # CPU also uses string key
                    ])
    
    print(f"✓ Updated {csv_path}")
    print("\nDone! Summary files now include cpu_percent data.")

if __name__ == "__main__":
    main()
