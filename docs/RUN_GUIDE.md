# Guide d'exécution : obtenir les vrais résultats

Matériel prévu : **Google Colab (T4, 16 Go)** pour tout ce qui demande du GPU, **un PC avec une RTX 2060 (6 Go)**
pour l'annotation humaine, l'analyse et, en option, les étapes légères. **GPT-4o** (clé OpenAI) sert de juge de
fidélité et d'attaquant LLM.

## 0. Une seule fois, pour toute l'équipe

1. **Hugging Face** : créer un compte, puis un token *Read* sur https://huggingface.co/settings/tokens.
   Ouvrir ces 3 pages connecté et cliquer sur « Agree / Access repository » si un bouton apparaît :
   - https://huggingface.co/datasets/imageomics/TreeOfLife-10M (obligatoire : c'est la cible)
   - https://huggingface.co/datasets/imageomics/TreeOfLife-200M
   - https://huggingface.co/datasets/imageomics/rare-species

   C'est la **seule** chose à faire à la main côté Hugging Face. Tout le reste (modèles, embeddings, images)
   se télécharge automatiquement.
2. **OpenAI** : vérifier que la clé a du crédit et accès à `gpt-4o` et `gpt-4o-mini`. Le coût estimé est
   d'environ 2 à 6 $ pour tout le projet (≈ 10 000 images en détail « low », plus les appels de l'attaquant LLM).
   Vérifiez les tarifs actuels sur la page d'OpenAI. Les réponses du juge sont mises en cache
   (`.cache/biofool/openai_*.jsonl`) : relancer le scoring ne coûte rien.
3. **Ne jamais mettre une clé dans le code ou dans git.** On passe toujours par des variables d'environnement ou par
   les secrets de Colab.

## 1. Colab (T4) : tout le calcul GPU

Ouvrir `notebooks/run_pipeline.ipynb` dans Colab (*File > Open notebook > GitHub*, puis l'URL du dépôt) et suivre
les cellules. Le notebook :
- met le code et les sorties sur votre **Google Drive** (`MyDrive/biofool_work/biofool`), pour ne rien perdre ;
- lit `OPENAI_API_KEY` et `HF_TOKEN` dans les **secrets Colab** (icône clé à gauche) ;
- utilise le profil `t4` (moins d'images par étape, gradient checkpointing, fp16) ;
- est découpé en 5 sessions de 2 à 3 h.

**Si Colab coupe** : relancer les cellules 1 à 3, puis la cellule en cours. Les étapes finies sont sautées, une
étape interrompue recommence, DDPO repart de son dernier point de contrôle, et le scoring ne calcule que ce qui
manque.

Durée totale estimée : **10 à 14 h de T4**. C'est une estimation, à vérifier avec `logs/*.log`, qui affichent la
durée de chaque étape. Colab gratuit limite les sessions ; Colab Pro aide si vous êtes pressés.

## 2. PC avec RTX 2060 (6 Go) : annotation, analyse, et étapes légères

### Installation sous Windows (PowerShell)

Il faut Python 3.11 (python.org, cocher « Add to PATH ») et Git for Windows.

```powershell
cd "C:\chemin\vers\biofool"
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Pour `cu124` : prenez la version CUDA affichée par `nvidia-smi` (`cu124` si ≥ 12.4, `cu121` si 12.1–12.3,
`cu118` si 11.8). La commande `Set-ExecutionPolicy` ne sert qu'une fois, pour autoriser l'activation de l'environnement.

### Installation sous Linux / WSL2

```bash
cd biofool
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
```

### Les clés, sur le PC

```powershell
hf auth login
setx OPENAI_API_KEY "sk-..."
```

`setx` enregistre la clé pour toujours : fermez et rouvrez le terminal ensuite. Sous Linux, c'est
`export OPENAI_API_KEY="sk-..."`. Sous Windows, `torch.compile` n'existe pas : le benchmark correspondant est sauté
automatiquement.

### Ce que la RTX 2060 peut faire (profil `rtx2060`)

```bash
python run.py test
python run.py check light --profile rtx2060
```

Le groupe `light` contient les téléchargements, la baseline, les modificateurs, le ciblé, l'attaquant LLM,
l'img2img et le benchmark VAE. Les étapes du groupe `gpu` (best-of-N, RAFT, DDPO, benchmark) gardent le générateur
et deux classifieurs en mémoire en même temps : sur 6 Go, faites-les sur Colab.

**Partager les sorties entre le PC et Colab** : installez *Google Drive pour ordinateur* et travaillez dans le même
dossier `MyDrive/biofool_work/biofool`. Sinon, faites tout le GPU sur Colab, c'est le plus simple.

## 3. Annotation humaine (chacun, ~20 min, sans GPU)

Après `human_sample` (session 5 sur Colab), le dossier `human_eval/` est sur le Drive. Sur votre ordinateur, dans
le dépôt, avec ce dossier `human_eval/` copié à la racine du dépôt :

```bash
pip install gradio pandas scikit-learn
python scripts/09_human_eval.py annotate --annotator prenom
```

Le navigateur s'ouvre. Répondez « yes », « no » ou « unsure » pour chaque image, puis remettez le fichier
`human_eval/answers_prenom.csv` dans le dossier `human_eval/` du Drive. Quand tout le monde a fini :

```bash
python scripts/09_human_eval.py report
```

## 4. Si quelque chose casse

Chaque étape écrit `logs/<étape>.log`. Envoyez la fin du fichier (les 50 dernières lignes) : on corrige et on
relance la même commande, et seules les étapes manquantes tournent.

| Symptôme | Cause probable | Solution |
|---|---|---|
| `CUDA out of memory` | GPU trop petit pour cette étape | faire l'étape sur Colab, ou `--set generator.batch=1` |
| `401` / `GatedRepoError` sur `imageomics/...` | conditions non acceptées ou pas connecté | étape 0.1, puis `hf auth login` |
| `OPENAI_API_KEY is not set` | clé absente de l'environnement | secrets Colab, ou `export OPENAI_API_KEY=...` |
| `RateLimitError` répétés | limite de débit OpenAI | `--set classifiers.gpt4o_judge.workers=2` |
| iNaturalist très lent | limite d'1 requête/s | c'est normal (≈ 30 min pour les 2 jeux iNat) |
| `species not in TreeOfLife names` | espèce invasive ou rare absente de TreeOfLife-10M | normal, signalé dans le log ; ces images ont `p_true = nan` |
