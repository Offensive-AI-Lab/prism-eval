#!/usr/bin/env python3
"""Render a xpia eval suite to a readable HTML review page.

Shows, per record: dataset/label/taxonomy badges, the reconstructed
conversation (the untrusted `tool` turn highlighted), and the ground-truth
instruction bullets — with the adversarial one(s) flagged by the live
adversarial identifier (unless --no-identify).

Usage:
    export PRISM_EVAL_MODEL=... PRISM_EVAL_BASE_URL=... PRISM_EVAL_API_KEY=...
    python scripts/view_xpia_suite.py data/eval_suite_xpia_smoke.json \
        -o xpia_smoke_review.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROLE_COLORS = {
    "system": "#6b7280", "user": "#2563eb",
    "assistant": "#059669", "tool": "#b91c1c",
}

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
margin:0 auto;padding:24px;background:#f8fafc;color:#0f172a;line-height:1.5}
h1{font-size:22px}.sub{color:#64748b;margin-bottom:20px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;
margin:18px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.eid{font-family:ui-monospace,monospace;font-weight:600;font-size:14px}
.badges{margin:6px 0 12px}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;
border-radius:999px;margin-right:6px;color:#fff}
.attack{background:#dc2626}.benign{background:#16a34a}
.meta{background:#e2e8f0;color:#334155}
.turn{border-left:4px solid #ccc;padding:6px 10px;margin:6px 0;border-radius:4px;
background:#f8fafc}
.role{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.tool-turn{background:#fef2f2;border-left-color:#b91c1c}
.content{white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12.5px;
margin-top:4px;max-height:280px;overflow:auto}
.bullets{margin-top:12px;border-top:1px dashed #e2e8f0;padding-top:10px}
.bullet{padding:5px 10px;margin:4px 0;border-radius:6px;background:#f1f5f9;font-size:13px}
.adv{background:#fef2f2;border:1px solid #fecaca}
.advtag{background:#dc2626;color:#fff;font-size:10px;font-weight:700;padding:1px 6px;
border-radius:4px;margin-left:8px}
.lbl{font-size:11px;color:#64748b;font-weight:600;text-transform:uppercase}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def render(suite_path: Path, out_path: Path, do_identify: bool) -> None:
    data = json.loads(suite_path.read_text())
    evals = data["evals"] if isinstance(data, dict) else data

    identify = None
    client = model = None
    if do_identify:
        from prism_eval.scoring.adversarial_identifier import identify as _id
        from openai import OpenAI
        identify = _id
        model = os.environ.get("PRISM_EVAL_MODEL", "")
        kw: dict = {"api_key": os.environ.get("PRISM_EVAL_API_KEY") or "not-needed"}
        if os.environ.get("PRISM_EVAL_BASE_URL"):
            kw["base_url"] = os.environ["PRISM_EVAL_BASE_URL"]
        client = OpenAI(**kw)

    n_attack = sum((r.get("metadata") or {}).get("label") == "attack" for r in evals)
    parts = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]
    parts.append("<h1>XPIA XPIA suite — ground-truth review</h1>")
    parts.append(f"<div class='sub'>{esc(suite_path.name)} · {len(evals)} records "
                 f"({n_attack} attack / {len(evals)-n_attack} benign) · "
                 f"adversarial bullets flagged by the identifier"
                 f"{'' if do_identify else ' (skipped)'}</div>")

    for r in evals:
        md = r.get("metadata") or {}
        bullets = (r.get("instruction_sources") or {}).get("original") or []
        label = md.get("label", "?")
        adv_idx: list[int] = []
        if identify is not None and label == "attack":
            try:
                adv_idx = identify(bullets, r.get("prompt"), model=model, client=client)
            except Exception as e:  # noqa: BLE001
                adv_idx = []
                print(f"  identify failed {r['eval_id']}: {str(e)[:60]}", file=sys.stderr)

        parts.append("<div class='card'>")
        parts.append(f"<div class='eid'>{esc(r['eval_id'])}</div>")
        lbl_cls = "attack" if label == "attack" else "benign"
        parts.append("<div class='badges'>")
        parts.append(f"<span class='badge {lbl_cls}'>{esc(label)}</span>")
        for k in ("attacker_goal", "injection_position", "tool_output_type"):
            if md.get(k) and str(md.get(k)) != "nan":
                parts.append(f"<span class='badge meta'>{esc(k)}: {esc(md[k])}</span>")
        parts.append("</div>")

        parts.append("<div class='lbl'>Reconstructed conversation</div>")
        for m in (r.get("prompt_messages") or []):
            role = m.get("role", "?")
            cls = "turn tool-turn" if role == "tool" else "turn"
            color = ROLE_COLORS.get(role, "#ccc")
            note = " &larr; untrusted content (injection lives here)" if role == "tool" else ""
            parts.append(
                f"<div class='{cls}' style='border-left-color:{color}'>"
                f"<span class='role' style='color:{color}'>{esc(role)}{note}</span>"
                f"<div class='content'>{esc(m.get('content'))}</div></div>"
            )

        parts.append("<div class='bullets'><div class='lbl'>Ground-truth instructions"
                     " (extracted by gemma4)</div>")
        for i, b in enumerate(bullets, 1):
            is_adv = i in adv_idx
            cls = "bullet adv" if is_adv else "bullet"
            tag = "<span class='advtag'>ADVERSARIAL</span>" if is_adv else ""
            parts.append(f"<div class='{cls}'>{i}. {esc(b)}{tag}</div>")
        parts.append("</div></div>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}  ({len(evals)} records)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("suite")
    p.add_argument("-o", "--out", default="xpia_smoke_review.html")
    p.add_argument("--no-identify", action="store_true",
                   help="skip the live adversarial-identifier calls")
    args = p.parse_args()
    render(Path(args.suite), Path(args.out), not args.no_identify)


if __name__ == "__main__":
    main()
