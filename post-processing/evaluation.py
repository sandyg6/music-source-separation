# Install the BSS Eval toolkit (if you don't already have it)
# pip install bss_eval

# Use BSS Eval to calculate SDR, SIR, and SAR
from bss_eval import bss_eval_sources
sd, sir, sar, _ = bss_eval_sources(ground_truth, predicted_sources)
print(f"SDR: {sd}, SIR: {sir}, SAR: {sar}")
