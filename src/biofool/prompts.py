"""Prompt construction: baseline prompt, attack modifiers (context engineering) and targeted prompts."""

from __future__ import annotations

import random

from .taxonomy import Species

BASELINE = "a high quality nature photograph of a {common} ({latin})"

# Context modifiers: they change everything around the organism, never its identity.
# Each family is a hypothesis about a failure mode of BioCLIP (shortcut on background, viewpoint, life stage...).
MODIFIERS: dict[str, list[str]] = {
    "background": ["on a plain white studio background", "on a city sidewalk", "inside a supermarket",
                   "on a kitchen table next to a coffee cup", "in a snowy landscape", "in a tropical rainforest",
                   "underwater"],
    "lighting": ["at night, lit by a camera flash", "backlit at sunset, silhouette", "under neon light",
                 "in heavy fog", "under harsh midday sun with strong shadows"],
    "viewpoint": ["extreme close-up macro shot", "seen from far away, small in the frame", "seen from directly above",
                  "seen from below", "partially hidden behind leaves"],
    "camera": ["blurry smartphone photo", "trail camera infrared photo", "old scanned film photo",
               "photo taken through a dirty window", "low resolution security camera still"],
    "style": ["museum specimen photograph", "photo of a printed field-guide page", "macro photo with bokeh"],
}

LIFE_STAGE: dict[str, list[str]] = {
    "insects": ["larva stage", "caterpillar stage", "pupa stage", "dead specimen pinned in a collection"],
    "birds": ["juvenile", "chick in a nest", "in flight", "molting"],
    "plants": ["seedling", "flower bud", "withered in autumn", "seed head"],
    "fungi": ["young button stage", "old and decaying", "sliced in half", "growing in a large cluster"],
}

LIFE_STAGE_OTHER = ["juvenile", "very young individual", "old individual", "dead specimen in a museum collection"]

TARGETED = [
    "in the typical habitat of the {target_common}",
    "photographed in the same style as typical photos of the {target_common}",
    "with colors reminiscent of a {target_common}",
]


def baseline_prompt(s: Species) -> str:
    return BASELINE.format(common=s.common_name, latin=s.latin)


def all_modifiers(s: Species) -> list[tuple[str, str]]:
    """(family, modifier) pairs available for species s."""
    out = [(fam, m) for fam, mods in MODIFIERS.items() for m in mods]
    out += [("life_stage", m) for m in LIFE_STAGE.get(s.group, LIFE_STAGE_OTHER)]
    return out


def attack_prompts(s: Species, k: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """k random (family, modifier, prompt) triples for species s."""
    mods = rng.sample(all_modifiers(s), k)
    return [(fam, m, f"{baseline_prompt(s)}, {m}") for fam, m in mods]


def targeted_prompts(s: Species, target: Species) -> list[tuple[str, str, str]]:
    out = []
    for t in TARGETED:
        m = t.format(target_common=target.common_name)
        out.append(("targeted", m, f"{baseline_prompt(s)}, {m}"))
    return out