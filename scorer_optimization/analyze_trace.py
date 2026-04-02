import json
from collections import defaultdict

with open(r'C:\Users\sigle\OneDrive\Documents\eacas\repos\ASPLOS\ASPLOS2\Analysis\scorer_optimization\outputs\scorer_trace.json') as f:
    data = json.load(f)

times = defaultdict(float)
counts = defaultdict(int)

for e in data.get('traceEvents', []):
    cat = str(e.get('cat', '')).lower()
    if True:
        name = e.get('name', 'Unknown')
        times[name] += e.get('dur', 0)
        counts[name] += 1

print(f"\n{'Total Time (ms)':<18} | {'Calls':<6} | {'Kernel Name'}")
print("-" * 100)
for name, dur in sorted(times.items(), key=lambda x: -x[1])[:15]:
    print(f"{dur/1000:<18.2f} | {counts[name]:<6} | {name[:70]}")



# HOW TO RUN
# python -m embodied_ai.common.scripts.scorer_self_test --device cuda --save_path scorer_optimization/outputs/scorer_self_test.json
# python scorer_optimization/analyze_trace.py

# For CUDA runs: Open windows x64 Native Tools Command Prompt for VS 2022,
# CD into analysis, run: \.venv\Scripts\activate, then run the first line above, then go back in VSCODE run the analyze trace
