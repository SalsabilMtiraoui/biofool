"""GPT-4o judge (with a fake OpenAI client), hardware profiles, resumable steps, the launcher."""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from biofool.config import load_config
from biofool.io import begin_run, finish_run


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, model, messages, **kw):
        self.calls += 1
        assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        top = [SimpleNamespace(token="Yes", logprob=math.log(0.6)), SimpleNamespace(token=" yes", logprob=math.log(0.1)),
               SimpleNamespace(token="No", logprob=math.log(0.3))]
        lp = SimpleNamespace(content=[SimpleNamespace(top_logprobs=top)])
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=lp, message=SimpleNamespace(content="Yes"))])


def test_openai_judge(species, tmp_path, monkeypatch):
    import openai

    from biofool.classifiers import OpenAIJudge

    fake = FakeCompletions()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai, "OpenAI", lambda: SimpleNamespace(chat=SimpleNamespace(completions=fake)))
    judge = OpenAIJudge("j", "gpt-4o", species, cache_dir=tmp_path, workers=2)
    ims = [Image.new("RGB", (64, 64), c) for c in ("red", "blue")]
    p = judge.predict(ims, [0, 1])["p_true"]
    assert p.tolist() == pytest.approx([0.7, 0.7])          # (0.6 + 0.1) / (0.6 + 0.1 + 0.3)
    assert fake.calls == 2
    again = OpenAIJudge("j", "gpt-4o", species, cache_dir=tmp_path)  # a new process: answers come from the disk cache
    again.predict(ims, [0, 1])
    assert fake.calls == 2 and again.n_calls == 0


def test_openai_judge_needs_key(species, tmp_path, monkeypatch):
    from biofool.classifiers import OpenAIJudge

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        OpenAIJudge("j", "gpt-4o", species, cache_dir=tmp_path)


def test_profiles():
    t4 = load_config("configs/experiments/bon.yaml", profile="t4")
    small = load_config("configs/experiments/bon.yaml", profile="rtx2060")
    assert t4.best_of_n.n == 6 and t4.best_of_n.prune_at is None   # the experiment keeps its own keys
    assert small.generator.batch == 2 and small.ddpo.gradient_checkpointing   # inherits t4
    assert load_config("configs/default.yaml", ["ddpo.epochs=3"], profile="t4").ddpo.epochs == 3  # CLI wins


def test_resumable_steps(tmp_path):
    run = tmp_path / "run"
    assert begin_run(run)
    run.mkdir()
    (run / "partial.png").write_text("x")
    assert begin_run(run) and not run.exists()               # interrupted run: cleaned and restarted
    finish_run(run)
    assert not begin_run(run)                                # complete run: skipped


def test_launcher_dry_run():
    out = subprocess.run([sys.executable, "run.py", "gpu", "--profile", "t4", "--dry-run"], capture_output=True,
                         text=True, check=True).stdout
    assert "05_train_ddpo.py" in out and "profile: t4" in out
