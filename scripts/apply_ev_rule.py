"""Re-apply build_deal_universe.enterprise_value() to an existing deal_universe.json. Equivalent to
re-running stage 3 (no other parsed field depends on EV) without repeating clause selection."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_deal_universe import OUT_PATH, enterprise_value

deals = json.load(open(OUT_PATH))
before = sum(d["ev_estimate"] is not None for d in deals)
for d in deals:
    d["ev_estimate"] = enterprise_value(d, d)
json.dump(deals, open(OUT_PATH, "w"), default=str)
print(f"EV available: {before} -> {sum(d['ev_estimate'] is not None for d in deals)} deals")
