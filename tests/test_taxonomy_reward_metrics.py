import pandas as pd
import pytest
import torch

from biofool.metrics import add_outcomes, bootstrap_ci, summary
from biofool.prompts import all_modifiers, attack_prompts, targeted_prompts
from biofool.reward import attack_reward, group_advantages
from biofool.taxonomy import load_pairs, taxonomic_distance, wrong_at_rank


def by_latin(species, name):
    return next(s for s in species if s.latin == name)


def test_species_file(species):
    assert len(species) == 80
    assert {g: sum(s.group == g for s in species) for g in {s.group for s in species}} == \
        {"insects": 20, "birds": 20, "plants": 20, "fungi": 20}
    assert len({s.latin for s in species}) == 80
    assert all(s.split in "AB" for s in species)


def test_pairs_resolve(species):
    pairs = load_pairs("data/confusable_pairs.csv", species)
    assert sum(len(v) for v in pairs.values()) == 21


def test_taxonomic_distance(species):
    monarch, viceroy = by_latin(species, "Danaus plexippus"), by_latin(species, "Limenitis archippus")
    raven, crow = by_latin(species, "Corvus corax"), by_latin(species, "Corvus brachyrhynchos")
    fly_agaric = by_latin(species, "Amanita muscaria")
    assert taxonomic_distance(monarch, monarch) == 0
    assert taxonomic_distance(raven, crow) == 1          # same genus
    assert taxonomic_distance(monarch, viceroy) == 2     # same family
    assert taxonomic_distance(monarch, fly_agaric) == 7  # different kingdom
    assert wrong_at_rank(monarch, viceroy, "genus") and not wrong_at_rank(monarch, viceroy, "family")


def test_labels(species):
    s = by_latin(species, "Danaus plexippus")
    assert s.label() == ("Animalia Arthropoda Insecta Lepidoptera Nymphalidae Danaus plexippus "
                         "with common name monarch butterfly")
    assert s.label("common") == "monarch butterfly"


def test_prompts(species):
    import random
    s, t = species[0], species[1]
    ps = attack_prompts(s, 5, random.Random(0))
    assert len(ps) == 5 and all(p.startswith("a high quality nature photograph of a monarch butterfly") for _, _, p in ps)
    assert len(all_modifiers(s)) > 25
    assert "viceroy butterfly" in targeted_prompts(s, t)[0][2]


def test_group_advantages_lecture_example():
    adv = group_advantages(torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(adv, torch.tensor([[0.7071, 0.7071, -1.4142]]), atol=1e-3)
    assert torch.allclose(group_advantages(torch.ones(2, 4)), torch.zeros(2, 4))  # no signal in a uniform group


def test_reward_modes():
    p_true = torch.tensor([0.9, 0.2])                     # target's probability of the true species, 2 images
    p_tgt = torch.tensor([0.1, 0.8])                      # ... of the target species
    oracle = torch.tensor([1.0, 0.1])
    assert torch.allclose(attack_reward(p_true, oracle, mode="gated"), torch.tensor([0.1, 0.08]))
    assert torch.allclose(attack_reward(p_true, oracle, mode="hard"), torch.tensor([0.1, 0.0]))
    assert torch.allclose(attack_reward(p_true, oracle, mode="fool_only"), torch.tensor([0.1, 0.8]))
    assert torch.allclose(attack_reward(p_true, oracle, p_tgt, mode="gated"), torch.tensor([0.1, 0.08]))
    nan = attack_reward(torch.tensor([float("nan")]), torch.tensor([1.0]))
    assert nan.item() == 1.0                              # species unknown to the target counts as fooled
    with pytest.raises(ValueError):
        attack_reward(p_true, oracle, mode="nope")


def test_metrics():
    df = pd.DataFrame(dict(
        species_idx=[0, 0, 1, 1], latin=["a a", "a a", "b b", "b b"],
        bioclip_top1=[0, 5, 1, 9], bioclip_top1_prob=[0.9, 0.7, 0.5, 0.3],
        bioclip_shared_ranks=[7, 6, 7, 0], bioclip_entropy=[0.1, 0.2, 0.3, 0.4],
        vlm_judge_p_true=[0.9, 0.9, 0.2, 0.9],
    ))
    s = summary(add_outcomes(df, "bioclip", "vlm_judge", 0.5))
    assert s["faithful_rate"] == 0.75
    assert s["ASR"] == pytest.approx(2 / 3)               # among the 3 faithful images, 2 fooled
    assert s["effective_ASR"] == 0.5
    assert s["taxo_dist"] == pytest.approx((1 + 7) / 2)
    assert s["ASR@family"] == pytest.approx(1 / 3)
    assert s["confident_err"] == pytest.approx(1 / 3)    # only the error with top-1 prob 0.7 is confident
    lo, hi = bootstrap_ci(add_outcomes(df, "bioclip", "vlm_judge", 0.5), "ASR", n_boot=200)
    assert 0 <= lo <= hi <= 1
