# BioFool — fooling BioCLIP with generated images

BioFool adapts a text-to-image diffusion model so that it produces **realistic images of a given species that the
[BioCLIP](https://imageomics.github.io/bioclip/) tree-of-life classifier gets wrong**. The target is BioCLIP exactly
as in its public demo: open-set over the ~450k species of TreeOfLife-10M. The project is a robustness audit of a
scientific foundation model. We use *semantic* attacks (background, light, viewpoint, life stage, camera, editing
of real photos), not invisible pixel noise.

| Stage | Technique | Course link |
|---|---|---|
| Context engineering | attack modifiers, retrieval of confusable species, LLM prompt attacker (OPRO-style) | Lab 9 |
| Editing real photos | img2img / SDEdit with a strength sweep (latent diffusion) | Labs 1, 2 |
| Inference-time compute | best-of-N, and **best-of-N with early rejection** (prune candidates from a TAESD preview of x̂₀) | Lab 11 |
| Post-training | **RAFT** (reward-ranked LoRA SFT) and **DDPO with group-relative advantages** (GRPO for diffusion) | Labs 3, 4, 5 |
| Inference optimisation | DDIM 50→10 steps, DPM-Solver++, LCM-LoRA (4 steps), tiny VAE, fp16, `torch.compile`, cached text embeddings | Labs 1, 2, 11 |
| Interpretability | t-SNE of embeddings, layer-wise drift of the CLS token, SmoothGrad saliency, prediction entropy | Lab 3 |

## What counts as "fooling"

An image of species *y* is a **successful attack** if

1. an **independent evaluation oracle** says it shows *y* (GPT-4o as a judge, validated by a blind human study),
2. **and** BioCLIP's top-1 species is not *y*.

Two rules keep the evaluation honest:

* the **reward oracle** (BioCLIP-2, used in training and search) is never the **evaluation oracle** (GPT-4o +
  humans, a different model family);
* synthetic results are reported on the **attackable set**: species whose clean generations are faithful and
  correctly classified. This removes the species that Stable Diffusion simply cannot draw.

## Datasets

| Dataset | Content | Role |
|---|---|---|
| `synth80` | text-to-image images of 80 species in 4 groups (insects, birds, plants, fungi), with 21 look-alike pairs | main attack set; split A (56 species) for training, B (24) for every reported number |
| `inat80` | real research-grade iNaturalist photos of the same 80 species | BioCLIP's accuracy on real photos + img2img attacks |
| `invasive` | real iNaturalist photos of 40 invasive alien species (EU Union list) | the high-stakes use case + img2img attacks |
| `rare` | the **Rare Species** benchmark of the BioCLIP paper (IUCN species held out of its training) | hardest real-world set + img2img attacks |

## Pipeline

```
 species y ──► prompt (baseline | modifiers | targeted | LLM) ──► Stable Diffusion 1.5 (+ LoRA)  ◄── real photo
                                                                         │ image                   (img2img)
                         ┌───────────────────────────────────────────────┼──────────────────────┐
                         ▼                                               ▼                      ▼
         BioCLIP, open set (target)                       BioCLIP-2 (reward oracle)  GPT-4o judge + humans
              p_T(y), top-1                                       p_O(y)                (evaluation only)
                         └──────────► reward = (1 − p_T(y)) · p_O(y) ◄──┘
                                           │
                     best-of-N / early rejection / RAFT / DDPO (GRPO)
```

## Repository layout

```
configs/default.yaml            every hyper-parameter; experiments/*.yaml override a few keys
data/                           species.csv (80 species, taxonomy, split), confusable_pairs.csv, invasive_species.txt
src/biofool/
  classifiers.py                BioCLIP open set (TreeOfLife), closed-set CLIP models (80 CLIP templates), VLM judge
  generator.py, ddim.py         SD wrapper with our own DDIM loop: log-probs, mid-chain pruning, img2img
  prompts.py, reward.py         attack prompts; attack rewards and group advantages
  scoring.py, search.py         reward computation, best-of-N and early rejection
  raft.py, ddpo.py, lora.py     post-training
  metrics.py                    ASR, effective ASR, ASR@rank, confident errors, taxonomic distance, bootstrap CIs
  taxonomy.py, io.py, cli.py    species, run directories, command line
scripts/                        00 check · 01 datasets · 02 generate · 03 score · 04 RAFT · 05 DDPO · 06 benchmark
                                07 LLM attacker · 08 analysis · 09 human study · 10 embeddings · 11 saliency
tests/                          offline CPU tests with tiny random models (run in CI)
notebooks/run_pipeline.ipynb    the same commands for Colab / a cluster notebook
docs/PLAN.md                    plan, open questions, timeline, team roles, rubric mapping (French)
report/                         ICML report draft (sections already written) and the summary figure (TikZ)
```

## Setup and running

**Step-by-step guide (in French): [`docs/RUN_GUIDE.md`](docs/RUN_GUIDE.md)**: Colab T4, a 6 GB GPU, the API keys,
the human study.

* Colab: open `notebooks/run_pipeline.ipynb` (T4 runtime, secrets `OPENAI_API_KEY` and `HF_TOKEN`).
* Your own GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match your CUDA version
pip install -r requirements.txt && pip install -e .
hf auth login                      # then accept the conditions of the imageomics datasets (see the guide)
export OPENAI_API_KEY=sk-...       # GPT-4o judge and LLM attacker
python run.py test                 # offline CPU tests
python run.py check --profile t4   # downloads every model, generates and scores one image
python run.py all --profile t4     # everything that needs no human, resumable
```

`python run.py list` shows every step and group (`data`, `attacks`, `training`, `speed`, `eval`, `gpu`, `light`).
Profiles: `t4` (Colab, 16 GB) and `rtx2060` (6 GB). A finished step is skipped, an interrupted one restarts, DDPO
resumes from its last checkpoint, and scoring only computes what is missing. Logs are written to `logs/<step>.log`;
tables and figures to `results/`.

| Step | Command | Output |
|---|---|---|
| Real datasets | `python run.py data` | `outputs/{inat80,invasive,rare}` |
| Text-to-image attacks | `python run.py baseline templates targeted llm bon bon_early` | `outputs/*_B` |
| Attacks on real photos | `python run.py i2i i2i_bon` | `outputs/i2i_*` |
| Post-training | `python run.py training` | `checkpoints/`, `outputs/{raft,ddpo,ddpo_nofid}_B` |
| Acceleration | `python run.py speed` | speed/memory per mode, early-vs-final reward correlation, VAE vs TAESD |
| Scoring and analysis | `python run.py score analyze interpret` | `results/tables/*.{md,tex}`, `results/figures/*.png` |
| Human study | `python run.py human_sample`, then `python scripts/09_human_eval.py annotate --annotator <name>` and `... report` | human fidelity, Cohen's κ with the GPT-4o judge |

## Metrics

Faithful rate · ASR = P(fooled | faithful) with 95% cluster-bootstrap CI · effective ASR · ASR@genus/family/order ·
taxonomic distance of the errors · confident errors (wrong with top-1 > 0.5) · entropy · transfer ASR to 4 other
classifiers · LPIPS diversity · seconds/image · peak VRAM · faithful attacks per GPU-hour.

## Responsible use

This project measures the weaknesses of a biodiversity classifier so that they can be documented and fixed. Generated
images are kept separate from real data, and they must never be uploaded to citizen-science platforms. Real photos
keep their licence and their source URL in `meta.jsonl`.

## References

Stevens et al., *BioCLIP*, CVPR 2024 · Black et al., *DDPO*, ICLR 2024 · Dong et al., *RAFT*, TMLR 2023 · Shao et al.,
*DeepSeekMath* (GRPO), 2024 · Yang et al., *OPRO*, ICLR 2024 · Meng et al., *SDEdit*, ICLR 2022 · Luo et al.,
*LCM-LoRA*, 2023 · Song et al., *Unrestricted Adversarial Examples*, NeurIPS 2018 · Hendrycks et al.,
*Natural Adversarial Examples*, CVPR 2021 · Smilkov et al., *SmoothGrad*, 2017.
