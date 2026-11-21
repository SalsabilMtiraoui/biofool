"""The classifiers that look at the generated images.

Every classifier has the same interface:

    clf.set_species(species)                      the label set of the run being scored
    clf.predict(images, true_idx, target_idx)     -> dict of per-image values:
        p_true        probability of the true species                       (VLM judge: p("yes"))
        p_target      probability of the target species (targeted attacks)  (nan otherwise)
        top1          index of the top-1 species in `species`, -1 if it is not in the list (open set)
        top1_name     binomial of the top-1 species
        top1_prob     its probability
        shared_ranks  ranks shared by the top-1 and the truth: 7 = correct species, 6 = right genus, ... 0
        entropy       entropy of the predicted distribution (uncertainty, as in Lab 3)

Three kinds:
  TreeOfLifeClassifier  BioCLIP exactly as in its public demo: open-set over the ~450k species of TreeOfLife-10M.
                        This is THE tool we try to fool.
  ZeroShotCLIP          closed set over the species of the run (BioCLIP, BioCLIP-2, OpenAI CLIP, SigLIP), with the
                        80 CLIP templates averaged, as pybioclip's CustomLabelsClassifier does. Text embeddings are
                        computed once and cached on disk (the same idea as prefix caching, Lab 11).
  OpenAIJudge           GPT-4o (OpenAI API) answers "is this <species>?": the independent EVALUATION oracle (default).
  VLMJudge              the same question to a local Qwen2.5-VL (needs ~18 GB of GPU memory for the 7B model).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .taxonomy import Species, shared_prefix
from .templates import OPENAI_TEMPLATES

NAN = float("nan")

# BioCLIP's own preprocessing for the TreeOfLife models (pybioclip: preprocess_img): resize to 224x224, no crop.
TOL_PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])


def _hash(*parts: str) -> str:
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def load_open_clip(model_id: str, device: str, dtype: torch.dtype):
    """model_id: "hf-hub:<repo>" or "<architecture>/<pretrained tag>" ("<arch>/none" = random weights, for tests)."""
    import open_clip

    if model_id.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
    else:
        arch, tag = model_id.split("/")
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=None if tag == "none" else tag)
        tokenizer = open_clip.get_tokenizer(arch)
    return model.to(device, dtype).eval().requires_grad_(False), preprocess, tokenizer


class _CLIPBase:
    """Image side shared by the open-set and closed-set CLIP classifiers."""

    def __init__(self, name, model_id, device, dtype, batch_size, preprocess):
        self.name, self.model_id, self.device, self.batch_size = name, model_id, device, batch_size
        self.dtype = dtype if device.startswith("cuda") else torch.float32
        self.model, default_pre, self.tokenizer = load_open_clip(model_id, device, self.dtype)
        self.preprocess = TOL_PREPROCESS if preprocess == "tol" else default_pre
        self.logit_scale = self.model.logit_scale.exp().float().item()

    def image_tensor(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device, self.dtype)

    @torch.no_grad()
    def image_features(self, images: list[Image.Image]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), self.batch_size):
            f = self.model.encode_image(self.image_tensor(images[i:i + self.batch_size])).float()
            out.append(F.normalize(f, dim=-1))
        return torch.cat(out)


class ZeroShotCLIP(_CLIPBase):
    kind = "closed_set"

    def __init__(self, name: str, model_id: str, species: list[Species], label_format: str = "taxonomic_common",
                 templates: str = "openai", preprocess: str = "default", device: str = "cuda",
                 cache_dir: str | Path = ".cache/biofool", batch_size: int = 64, dtype: torch.dtype = torch.float16):
        super().__init__(name, model_id, device, dtype, batch_size, preprocess)
        self.label_format, self.templates, self.cache_dir = label_format, templates, Path(cache_dir)
        self.set_species(species)

    @torch.no_grad()
    def encode_classnames(self, classnames: list[str]) -> torch.Tensor:
        tpls = OPENAI_TEMPLATES if self.templates == "openai" else ["a photo of {c}."]
        feats = []
        for c in classnames:
            tok = self.tokenizer([t.format(c=c) for t in tpls]).to(self.device)
            f = F.normalize(self.model.encode_text(tok).float(), dim=-1).mean(0)
            feats.append(f / f.norm())
        return torch.stack(feats)

    def set_species(self, species: list[Species]):
        self.species = species
        labels = [s.label(self.label_format) for s in species]
        cache = self.cache_dir / f"text_{self.name}_{_hash(self.model_id, self.templates, *labels)}.pt"
        if cache.exists():
            self.text_features = torch.load(cache, map_location=self.device)
        else:
            self.text_features = self.encode_classnames(labels)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.text_features.cpu(), cache)
        self.text_features = self.text_features.to(self.device)

    def probs_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return (self.logit_scale * feats @ self.text_features.T).softmax(-1)

    def probs(self, images: list[Image.Image]) -> torch.Tensor:
        return self.probs_from_features(self.image_features(images)).cpu()

    def predict(self, images, true_idx, target_idx=None) -> dict:
        p = self.probs(images)
        rows = torch.arange(len(images))
        true_idx = torch.as_tensor(true_idx)
        top_p, top = p.max(-1)
        return dict(
            p_true=p[rows, true_idx],
            p_target=(p[rows, torch.as_tensor(target_idx)] if target_idx is not None
                      else torch.full((len(images),), NAN)),
            top1=top, top1_name=[self.species[int(i)].latin for i in top], top1_prob=top_p,
            shared_ranks=torch.tensor([shared_prefix(self.species[int(t)].lineage(), self.species[int(k)].lineage())
                                       for t, k in zip(true_idx, top)]),
            entropy=_entropy(p))

    def nearest_species(self, idx: int, k: int, same_group: bool = True) -> list[int]:
        """The k species whose text embeddings are closest to species idx: candidate targets for a targeted attack."""
        sims = self.text_features @ self.text_features[idx]
        sims[idx] = -float("inf")
        if same_group:
            for s in self.species:
                if s.group != self.species[idx].group:
                    sims[s.idx] = -float("inf")
        return sims.topk(k).indices.tolist()


class TreeOfLifeClassifier(_CLIPBase):
    """BioCLIP as in the public demo: zero-shot over every species of TreeOfLife-10M (embeddings precomputed by the
    BioCLIP authors in the dataset repo, exactly what pybioclip's TreeOfLifeClassifier uses)."""

    kind = "open_set"

    def __init__(self, name: str, model_id: str, species: list[Species], tol_repo: str = "imageomics/TreeOfLife-10M",
                 device: str = "cuda", batch_size: int = 64, dtype: torch.dtype = torch.float16, **_):
        from huggingface_hub import hf_hub_download

        super().__init__(name, model_id, device, dtype, batch_size, preprocess="tol")
        emb = torch.from_numpy(np.load(hf_hub_download(tol_repo, "embeddings/txt_emb_species.npy",
                                                       repo_type="dataset")))
        with open(hf_hub_download(tol_repo, "embeddings/txt_emb_species.json", repo_type="dataset")) as f:
            names = json.load(f)  # [[[kingdom, ..., genus, epithet], common name], ...]
        dim = self.model.text_projection.shape[1] if hasattr(self.model, "text_projection") else emb.shape[0]
        self.text_features = (emb if emb.shape[0] == dim else emb.T).to(device, self.dtype)  # [dim, n_taxa]
        self.lineages = [tuple(n[0][:6]) + (f"{n[0][5]} {n[0][6]}",) for n in names]
        self.common = [n[1] for n in names]
        self.index: dict[str, list[int]] = {}
        for i, lin in enumerate(self.lineages):
            self.index.setdefault(lin[-1].lower(), []).append(i)
        self.set_species(species)

    def set_species(self, species: list[Species]):
        self.species = species
        self.species_taxa = [self.index.get(s.latin.lower(), []) for s in species]
        self.by_binomial = {s.latin.lower(): s.idx for s in species}
        missing = [s.latin for s, t in zip(species, self.species_taxa) if not t]
        if missing:
            print(f"[{self.name}] {len(missing)} species not in TreeOfLife names (p_true = nan): {missing[:5]}...")

    def tol_lineage(self, s: Species) -> tuple[str, ...]:
        """The lineage of species s as written in TreeOfLife (falls back on our csv if s is not found)."""
        t = self.species_taxa[s.idx]
        return self.lineages[t[0]] if t else s.lineage()

    @torch.no_grad()
    def predict(self, images, true_idx, target_idx=None) -> dict:
        out = {k: [] for k in ("p_true", "p_target", "top1", "top1_name", "top1_prob", "shared_ranks", "entropy")}
        for i in range(0, len(images), self.batch_size):
            feats = self.image_features(images[i:i + self.batch_size]).to(self.dtype)
            p = (self.logit_scale * feats @ self.text_features).float().softmax(-1)
            top_p, top = p.max(-1)
            for j in range(len(feats)):
                t = int(true_idx[i + j])
                taxa = self.species_taxa[t]
                out["p_true"].append(p[j, taxa].sum().item() if taxa else NAN)
                tgt = None if target_idx is None else target_idx[i + j]
                tt = self.species_taxa[int(tgt)] if tgt is not None and not _isnan(tgt) else []
                out["p_target"].append(p[j, tt].sum().item() if tt else NAN)
                lin = self.lineages[int(top[j])]
                out["top1"].append(self.by_binomial.get(lin[-1].lower(), -1))
                out["top1_name"].append(lin[-1])
                out["top1_prob"].append(top_p[j].item())
                out["shared_ranks"].append(shared_prefix(self.tol_lineage(self.species[t]), lin))
                out["entropy"].append(_entropy(p[j]).item())
        return {k: (v if k == "top1_name" else torch.tensor(v)) for k, v in out.items()}


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


class VLMJudge:
    """Fidelity judge: asks a vision-language model whether the image shows the given species.

    Returns p(yes) / (p(yes) + p(no)) for the first answer token. Independent of the CLIP family, so it is used as the
    *evaluation* oracle (never inside the reward)."""

    kind = "judge"
    QUESTION = ("Look at this photo. Does it show a {common} ({latin})? It must be this exact species, "
                "realistically depicted. Answer with Yes or No.")

    def __init__(self, name: str, model_id: str, species: list[Species], device: str = "cuda", batch_size: int = 8,
                 **_):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name, self.device, self.batch_size = name, device, batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto").to(device).eval()
        tok = self.processor.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")})
        self.set_species(species)

    def set_species(self, species: list[Species]):
        self.species = species

    @torch.no_grad()
    def p_yes(self, images: list[Image.Image], species_idx: list[int]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), self.batch_size):
            ims, idxs = images[i:i + self.batch_size], species_idx[i:i + self.batch_size]
            texts = []
            for idx in idxs:
                s = self.species[int(idx)]
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.QUESTION.format(common=s.common_name, latin=s.latin)}]}]
                texts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))
            batch = self.processor(text=texts, images=[im.convert("RGB") for im in ims], return_tensors="pt",
                                   padding=True).to(self.device)
            p = self.model(**batch).logits[:, -1].float().softmax(-1)
            yes, no = p[:, self.yes].sum(-1), p[:, self.no].sum(-1)
            out.append((yes / (yes + no + 1e-8)).cpu())
        return torch.cat(out)

    def predict(self, images, true_idx, target_idx=None) -> dict:
        return dict(p_true=self.p_yes(images, list(true_idx)))


class OpenAIJudge:
    """Fidelity judge through the OpenAI API (GPT-4o by default): p(yes) from the log-probabilities of the first answer
    token. Needs OPENAI_API_KEY in the environment.

    Every answer is cached on disk (key = model + question + image bytes), so re-running the scoring never pays twice.
    Images are sent as 512 px JPEGs with detail "low" (a fixed, small number of tokens per image)."""

    kind = "judge"
    QUESTION = VLMJudge.QUESTION

    def __init__(self, name: str, model_id: str, species: list[Species], detail: str = "low", workers: int = 8,
                 cache_dir: str | Path = ".cache/biofool", **_):
        import threading

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (see docs/RUN_GUIDE.md)")
        from openai import OpenAI

        self.client, self.name, self.model_id = OpenAI(), name, model_id
        self.detail, self.workers = detail, workers
        self.cache_path = Path(cache_dir) / f"openai_{name}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path):
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["key"]] = d["p"]
        self.lock = threading.Lock()
        self.n_calls = 0
        self.set_species(species)

    def set_species(self, species: list[Species]):
        self.species = species

    def _ask(self, image: Image.Image, s: Species) -> float:
        import base64
        import io
        import time

        im = image.convert("RGB")
        im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        question = self.QUESTION.format(common=s.common_name, latin=s.latin)
        key = hashlib.sha1(self.model_id.encode() + question.encode() + data).hexdigest()
        if key in self.cache:
            return self.cache[key]
        url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": question},
                                                 {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}]}]
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(model=self.model_id, messages=messages, max_completion_tokens=1,
                                                        temperature=0, logprobs=True, top_logprobs=10)
                break
            except Exception as e:  # rate limits and transient server errors: back off and retry
                if attempt == 5:
                    print(f"[{self.name}] giving up on one image: {e}")
                    return NAN
                time.sleep(2 ** attempt)
        choice = r.choices[0]
        yes = no = 0.0
        if choice.logprobs and choice.logprobs.content:
            for t in choice.logprobs.content[0].top_logprobs:
                word = t.token.strip().lower()
                yes += math.exp(t.logprob) if word == "yes" else 0.0
                no += math.exp(t.logprob) if word == "no" else 0.0
        if yes + no == 0:  # no log-probabilities: fall back on the text
            text = (choice.message.content or "").strip().lower()
            yes, no = float(text.startswith("yes")), float(text.startswith("no"))
        p = yes / (yes + no) if yes + no > 0 else NAN
        with self.lock:
            self.cache[key] = p
            self.n_calls += 1
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "p": p}) + "\n")
        return p

    def p_yes(self, images: list[Image.Image], species_idx: list[int]) -> torch.Tensor:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            ps = list(pool.map(lambda a: self._ask(a[0], self.species[int(a[1])]), zip(images, species_idx)))
        return torch.tensor(ps, dtype=torch.float32)

    def predict(self, images, true_idx, target_idx=None) -> dict:
        return dict(p_true=self.p_yes(images, list(true_idx)))


def build_classifier(name: str, cfg, species: list[Species]):
    spec = dict(cfg.classifiers[name])
    kind = spec.pop("kind", "closed_set")
    if "classifiers_batch_size" in cfg and kind != "openai_judge":
        spec.setdefault("batch_size", cfg.classifiers_batch_size)
    if kind == "openai_judge":
        return OpenAIJudge(name, species=species, cache_dir=cfg.cache_dir, **spec)
    if kind == "judge":
        return VLMJudge(name, species=species, device=cfg.device, **spec)
    if kind == "open_set":
        return TreeOfLifeClassifier(name, species=species, device=cfg.device, **spec)
    return ZeroShotCLIP(name, species=species, device=cfg.device, cache_dir=cfg.cache_dir, **spec)
