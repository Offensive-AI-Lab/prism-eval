#!/usr/bin/env python
"""Reconstruct the indirect-prompt-injection corpus from public sources.

The corpus is NOT redistributed. This script rebuilds a comparable corpus at
the same per-source volume from the three public benchmarks it draws on:

    BIPIA        git clone https://github.com/microsoft/BIPIA           (MIT; Yi et al. 2023)
    InjecAgent   git clone https://github.com/uiuc-kang-lab/InjecAgent  (MIT; Zhan et al. 2024)
    LLMail       HF datasets: microsoft/llmail-inject-challenge         (MIT; Microsoft)

The assembly logic follows Max Fomin, "When Benchmarks Lie" (arXiv:2602.14161)
and the loaders in https://github.com/maxf-zn/prompt-mining (MIT, (c) Zenity /
Z Labs). Row selection is a seeded sample to the released per-source counts
(BIPIA 13,950 = 13,750 attack + 200 benign / LLMail 9,998 / InjecAgent 1,054;
total 25,002); it reproduces the same
benchmarks at the same volume, not the identical rows behind the reported
numbers (that selection + the LLM taxonomy pass lived in a private pipeline).

Requirements (this is a standalone tool, not a training/eval dependency):
    pip install datasets jsonlines nltk transformers pandas pyarrow
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
    git clone https://github.com/microsoft/BIPIA
    git clone https://github.com/uiuc-kang-lab/InjecAgent

Usage:
    python scripts/build_xpia_corpus.py \
        --bipia-root ./BIPIA --injecagent-root ./InjecAgent \
        --out data/xpia_corpus.parquet
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

SEED = 2024
COUNTS = {"bipia": 13950, "llmail": 9998, "injecagent": 1054}  # bipia total = 13,750 attack + 200 benign
BENIGN_BIPIA = 200  # released corpus carries 200 benign bipia rows

SCHEMA = [
    "content", "label", "dataset", "category", "attacker_goal",
    "delivery_technique", "evasion_technique", "injection_position",
    "tool_output_type", "scope", "taxonomy_rationale", "taxonomy_source",
    "tier1_nature", "tier2_category",
]

# Coarse, deterministic tags per source. The original fine taxonomy came from a
# private LLM pass and is neither reproducible nor needed to run the eval; these
# keep the schema populated for scripts/fetch_xpia_evals.py and analyze_xpia.py.
BIPIA_TOOL = {"email": "email_messaging", "table": "structured_data", "code": "code"}


def _attack_row(content, dataset, tool, position, goal="action_hijacking",
                delivery="direct_instruction", tier2="direct_harm"):
    return {
        "content": content, "label": "attack", "dataset": dataset,
        "category": "injection", "attacker_goal": goal,
        "delivery_technique": delivery, "evasion_technique": "none",
        "injection_position": position, "tool_output_type": tool,
        "scope": "security", "taxonomy_rationale": "",
        "taxonomy_source": "reconstructed", "tier1_nature": "security",
        "tier2_category": tier2,
    }


def build_injecagent(root: Path) -> list[dict]:
    """InjecAgent base test cases (dh_base + ds_base = 1,054), flattened."""
    rows = []
    for attack_type, fname, goal, tier2 in [
        ("dh", "test_cases_dh_base.json", "action_hijacking", "direct_harm"),
        ("ds", "test_cases_ds_base.json", "data_exfiltration_pii", "data_exfiltration"),
    ]:
        cases = json.loads((root / "data" / fname).read_text())
        for c in cases:
            content = (
                f"{c['User Instruction']}\n"
                f"[tool_call] {c['User Tool']}({c['Tool Parameters']})\n"
                f"[tool_response] {c['Tool Response']}"
            )
            rows.append(_attack_row(
                content, "injecagent", "structured_data", "middle",
                goal=goal, delivery="tool_call_injection", tier2=tier2,
            ))
    return rows


def build_llmail(count: int) -> list[dict]:
    """LLMail-Inject Phase1, seed-sampled to `count`, formatted subject+body."""
    from datasets import load_dataset
    ds = load_dataset("microsoft/llmail-inject-challenge", split="Phase1")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= count:
            break
        r = ds[i]
        subject = (r.get("subject") or "").strip()
        body = (r.get("body") or "").strip()
        if not body:
            continue
        content = f"Subject: {subject}\nBody:\n{body}"
        rows.append(_attack_row(content, "llmail", "email_messaging", "middle"))
    return rows


def build_bipia(root: Path, count: int, n_benign: int) -> list[dict]:
    """BIPIA email/table/code attacks (all positions), seed-sampled to `count`,
    plus `n_benign` clean-context benign rows. Uses BIPIA's own MIT builder.

    construct_prompt(..., require_system_prompt=False) returns the single
    user string that concatenates the (poisoned) context and the question —
    the same shape the corpus stores in `content`.
    """
    import pandas as pd
    sys.path.insert(0, str(root))
    from bipia.data import AutoPIABuilder

    bench = root / "benchmark"
    tasks = [("email", "text_attack_test.json"), ("table", "text_attack_test.json"),
             ("code", "code_attack_test.json")]

    attack_pool: list[dict] = []
    benign_pool: list[dict] = []
    for task, af in tasks:
        tool = BIPIA_TOOL[task]
        pia = AutoPIABuilder.from_name(task)(seed=SEED)
        for split in ("train", "test"):
            df = pia(str(bench / task / f"{split}.jsonl"), str(bench / af),
                     enable_stealth=False)
            for _, ex in df.iterrows():
                user = pia.construct_prompt(ex, require_system_prompt=False)
                attack_pool.append(_attack_row(
                    user, "bipia", tool, str(ex["position"]),
                    delivery="contextual_manipulation",
                ))
            # benign: clean (un-poisoned) contexts, one prompt per context
            for ctx in pia.context_samples:
                user = pia.construct_prompt(pd.Series(ctx), require_system_prompt=False)
                b = _attack_row(user, "bipia", tool, "", goal="", delivery="", tier2="")
                b.update(label="benign", category="benign", scope="benign",
                         tier1_nature="", attacker_goal=None, delivery_technique=None,
                         evasion_technique=None, injection_position=None)
                benign_pool.append(b)

    rng = random.Random(SEED)
    rng.shuffle(attack_pool)
    rng.shuffle(benign_pool)
    return attack_pool[:count - n_benign] + benign_pool[:n_benign]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bipia-root", type=Path, required=True)
    ap.add_argument("--injecagent-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/xpia_corpus.parquet"))
    ap.add_argument("--skip-llmail", action="store_true",
                    help="skip the LLMail HF download (build bipia+injecagent only)")
    args = ap.parse_args()

    import pandas as pd
    rows = []
    print("building injecagent ...")
    rows += build_injecagent(args.injecagent_root)
    print("building bipia ...")
    rows += build_bipia(args.bipia_root, COUNTS["bipia"], BENIGN_BIPIA)
    if not args.skip_llmail:
        print("building llmail (HF download) ...")
        rows += build_llmail(COUNTS["llmail"])

    df = pd.DataFrame(rows)[SCHEMA]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(df.groupby(["dataset", "label"]).size())
    return 0


if __name__ == "__main__":
    sys.exit(main())
