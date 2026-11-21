"""LLM prompt attacker (OPRO-style optimisation in context, Lab 9). Backend: OpenAI API (default) or a local model.

For each species, an instruction-tuned LLM proposes context modifiers. Each modifier is tried on a few images and
scored with the attack reward (target + REWARD oracle). The best modifiers and their scores are put back into the
LLM's context, and it proposes new ones. The LLM never sees an image: it learns from the scores only.
Output: a JSON {latin: [best modifiers]} for `02_generate.py --method llm`, plus the full search log.

Example:  python scripts/07_llm_attacker.py --out outputs/llm_modifiers.json --split A
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biofool.cli import parse, select_species  # noqa: E402
from biofool.generator import Generator  # noqa: E402
from biofool.prompts import MODIFIERS, baseline_prompt  # noqa: E402
from biofool.scoring import Scorer  # noqa: E402

SYSTEM = (
    "You help to test the robustness of an image classifier for biodiversity. You write short modifiers that are "
    "appended to an image-generation prompt. A modifier changes the context of the photo (background, light, "
    "viewpoint, camera, life stage, surroundings) but the organism itself must stay the same species and be clearly "
    "visible. Never name another species, never ask for text in the image.")

USER = """Species: {common} ({latin}), {group}.
Base prompt: "{base}, <modifier>"
Examples of modifiers: {examples}

Modifiers tried so far, with their score (higher = the classifier is more often wrong while the species stays
recognisable; maximum 1.0):
{history}

Propose {k} NEW modifiers likely to score higher. Answer with a JSON list of strings only."""


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="A", choices=["all", "A", "B"])
    p.add_argument("--groups", default=None)
    p.add_argument("--limit", type=int, default=None)


class OpenAIChat:
    """The attacker LLM through the OpenAI API (gpt-4o-mini by default): no GPU memory needed."""

    def __init__(self, model):
        from openai import OpenAI
        self.client, self.model = OpenAI(), model

    def __call__(self, messages) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.9,
                                                max_completion_tokens=300)
        return r.choices[0].message.content or ""


class LocalChat:
    """The attacker LLM as a local Hugging Face model."""

    def __init__(self, model_id, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device).eval()

    @torch.no_grad()
    def __call__(self, messages) -> str:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.9, top_p=0.95)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def propose(chat, s, history, k, rng):
    examples = ", ".join(f'"{m}"' for m in rng.sample([m for v in MODIFIERS.values() for m in v], 4))
    hist = "\n".join(f'- "{m}": {r:.3f}' for m, r in sorted(history.items(), key=lambda x: -x[1])[:15]) or "(none yet)"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(common=s.common_name, latin=s.latin, group=s.group,
                                                        base=baseline_prompt(s), examples=examples, history=hist, k=k)}]
    text = chat(messages)
    match = re.search(r"\[.*\]", text, re.S)
    try:
        mods = [m.strip() for m in json.loads(match.group(0)) if isinstance(m, str)] if match else []
    except json.JSONDecodeError:
        mods = re.findall(r'"([^"]{3,120})"', text)
    return [m for m in mods if m not in history][:k]


def main():
    args, cfg, species = parse(__doc__, add_args)
    if Path(args.out).exists():
        print(f"{args.out} exists: skipping (delete it to run again)")
        return
    c = cfg.llm_attacker
    rng = random.Random(cfg.seed)
    chat = OpenAIChat(c.openai_model) if c.get("backend", "local") == "openai" else LocalChat(c.model_id, cfg.device)
    gen, scorer = Generator(cfg), Scorer(cfg, species)
    best, log = {}, []
    for s in select_species(species, args.split, args.groups, args.limit):
        history = {}
        for rnd in range(c.rounds):
            mods = propose(chat, s, history, c.proposals_per_round, rng)
            for m in mods:
                seeds = [rng.randrange(2**31) for _ in range(c.images_per_modifier)]
                out = gen.sample([f"{baseline_prompt(s)}, {m}"] * len(seeds), seeds)
                history[m] = scorer(out.images, [s.idx] * len(seeds))["reward"].mean().item()
                log.append(dict(latin=s.latin, round=rnd, modifier=m, reward=history[m]))
            print(f"{s.latin:28s} round {rnd}: best {max(history.values(), default=0):.3f}")
        best[s.latin] = [m for m, _ in sorted(history.items(), key=lambda x: -x[1])[:c.keep]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=2))
    Path(args.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
