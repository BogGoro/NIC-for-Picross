import json

def analyze(file: str):
    with open(file) as f:
        data = json.load(f)

    for size, records in data.items():
        total = len(records)
        solved = sum(1 for r in records if r["solved"])
        avg_time = sum(r["time"] for r in records) / total
        print(f"{file} - {size}:")
        print(f"  Accuracy: {solved / total * 100:.1f}%")
        print(f"  Avg time: {avg_time:.4f}s\n")

analyze("GA-results.json")
analyze("SA-results.json")
