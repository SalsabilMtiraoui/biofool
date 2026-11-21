# BioFool : plan du projet (semaines 10 à 14)

## 1. Le problème, formellement

Pour une espèce $y$, on produit une image $x$ de deux façons :
- **texte → image** (jeu `synth80`) : $x = G_\phi(\text{prompt}(y, m), z)$ avec Stable Diffusion 1.5, où $m$ est un
  modificateur de contexte et $z$ le bruit initial ;
- **image → image** (jeux `inat80`, `invasive`, `rare`) : on part d'une **vraie photo** de $y$, on la bruite jusqu'à
  une force $s$ et on la régénère avec un modificateur (SDEdit).

Trois rôles de modèles :

| Rôle | Modèle | Sert à |
|---|---|---|
| Cible $T$ | **BioCLIP comme dans sa démo** : zero-shot sur les ~450 000 espèces de TreeOfLife-10M (`bioclip_tol`) | ce qu'on veut tromper |
| Oracle de récompense $O$ | BioCLIP-2 (ensemble fermé) | fidélité **dans** la récompense (recherche, RAFT, DDPO) |
| Oracle d'évaluation $E$ | GPT-4o via l'API OpenAI (oui/non) + humains | fidélité **dans les chiffres du rapport** |

**Attaque réussie :** $E$ dit « c'est bien $y$ » **et** le top-1 de $T$ n'est pas $y$.

**Récompense** (`reward.py`) : $r = (1 - p_T(y)) \cdot p_O(y)$. L'ablation `fool_only` retire le facteur de
fidélité, pour montrer le reward hacking (comme l'exercice 2b du Lab 4).

## 2. Les 4 jeux de données (critère « Extensiveness »)

| Jeu | Contenu | Pourquoi |
|---|---|---|
| `synth80` | images générées de 80 espèces, 4 groupes (insectes, oiseaux, plantes, champignons), 21 paires de sosies | jeu principal ; split A pour entraîner, split B pour évaluer |
| `inat80` | vraies photos iNaturalist des mêmes 80 espèces | précision de BioCLIP sur du réel + attaques img2img |
| `invasive` | vraies photos de 40 espèces invasives (liste de l'UE) | le cas d'usage critique : rater une espèce invasive |
| `rare` | benchmark **Rare Species** du papier BioCLIP (espèces UICN exclues de son entraînement) | le test le plus dur, et le jeu de référence des auteurs |

En plus de ces 4 jeux, on compare **≥ 12 variantes du générateur** et **5 classifieurs** (BioCLIP ouvert et fermé,
BioCLIP-2, CLIP, SigLIP). La condition « 4+ datasets *or* distinct model variants » est donc remplie deux fois.

## 3. Les méthodes et leur hypothèse

| # | Méthode | Hypothèse testée | Commande |
|---|---|---|---|
| M0 | Baseline | BioCLIP reconnaît les images générées « propres » (sinon l'espèce sort de l'ensemble attaquable) | `python run.py baseline` |
| M1 | Modificateurs de contexte (6 familles) | BioCLIP utilise des raccourcis : fond, lumière, angle, stade de vie… | `python run.py templates` |
| M2 | Ciblé (sosies + voisins dans l'espace texte de BioCLIP) | les erreurs suivent la structure de l'embedding | `python run.py targeted` |
| M3 | Attaquant LLM (OPRO, Lab 9) | un LLM qui ne voit que des scores trouve de meilleurs modificateurs que nos templates | `python run.py llm` |
| M4 | Best-of-N (8, ou 6 avec le profil t4) | plus de calcul à l'inférence donne plus d'attaques | `python run.py bon` |
| M5 | Best-of-N (8, ou 6 avec le profil t4) + **rejet précoce** | l'aperçu x̂₀ (TAESD) à 40 % des étapes classe déjà bien les candidats : −45 % de calcul | `python run.py bon_early early` |
| M6 | img2img, forces 0,3 / 0,5 / 0,7 | partir d'une vraie photo garde l'espèce ; la force règle le compromis fidélité / ASR | `python run.py i2i` |
| M7 | img2img + best-of-N + rejet précoce | la recherche marche aussi sur de vraies photos | `python run.py i2i_bon` |
| M8 | RAFT (LoRA SFT sur les meilleures attaques de M4, Labs 3 et 5) | distiller la recherche dans le générateur | `python run.py raft eval_lora` |
| M9 | **DDPO + avantages de groupe (GRPO, Lab 4)** | le RL généralise à des espèces jamais vues (split B) | `python run.py ddpo eval_lora` |
| M10 | DDPO sans fidélité | ablation : reward hacking | `python run.py ddpo_ablation` |
| M11 | Accélération : DDIM 50/25/10, DPM-Solver++, LCM-LoRA, TAESD, fp16, `torch.compile` | gain ×5–10 pour une perte de fidélité mesurée | `python run.py bench vae` |

**Analyses d'interprétabilité (Lab 3) :** t-SNE des embeddings de BioCLIP, dérive couche par couche du token CLS
entre la vraie photo et sa version attaquée, saillance SmoothGrad, entropie et « erreurs confiantes » (`python run.py interpret`).

## 4. Correspondance avec la grille (viser 9–10 partout)

| Critère | Ce qui le justifie |
|---|---|
| Nouveauté | Attaques sémantiques contre l'outil **réel** (démo BioCLIP en ensemble ouvert) ; fidélité jugée par un oracle indépendant ; ASR par rang taxonomique ; espèces invasives et rares comme cas d'usage |
| Difficulté d'adaptation | Adaptation **nouvelle** (DDPO/GRPO sur LoRA avec récompense à deux oracles) **+** amélioration **à l'inférence** (rejet précoce) **+** optimisations classiques (étapes, LCM, TAESD, fp16, compile, cache), toutes chiffrées |
| Évaluation | 4 jeux de données, ≥ 12 variantes du générateur, 5 classifieurs, IC par bootstrap, étude humaine avec κ de Cohen, analyses d'interprétabilité |
| Rapport et code | Tests en CI, `python run.py <étape>` pour chaque figure, README, configs versionnées, seeds fixes, brouillon du rapport déjà rédigé |

## 5. Ce qui est prêt, et ce qu'il vous reste à faire

**Prêt dans le dépôt :** tout le code jusqu'à la semaine 13 (génération, attaques, entraînement, benchmark,
scoring, analyse, figures, interprétabilité, application d'annotation), les tests, et le brouillon du rapport.

**Ce qu'il reste à faire (impossible depuis ma machine : pas de GPU, pas d'accès à Hugging Face) :**

| Semaine | À faire | Qui |
|---|---|---|
| **10** | `python run.py test check taxonomy smoke`, puis `python run.py data baseline` (voir `docs/RUN_GUIDE.md`) ; regarder 20 images à la main ; corriger les noms de modèles si `check` échoue | 1 personne avec le GPU |
| **11** | `python run.py templates targeted llm bon bon_early early i2i i2i_bon` (≈ 4–5 h GPU) | GPU |
| **12** | `python run.py raft ddpo ddpo_ablation eval_lora` (≈ 4–6 h GPU) ; vérifier les courbes de `checkpoints/ddpo/train_log.csv` | GPU |
| **13** | `python run.py bench vae score analyze interpret human_sample` ; **annotation humaine** : chaque membre lance `python scripts/09_human_eval.py annotate --annotator <prénom>` (~210 images, 20 min), puis `... report` | tout le monde |
| **14** | Remplir les chiffres du rapport (`report/`), figure de synthèse, 20 min de présentation | tout le monde |

`python run.py all` enchaîne toutes les étapes sans humain (≈ 10–14 h sur une A100, 2 à 3 fois plus sur un T4).

**Si on manque de temps, on coupe dans cet ordre :** M3 (LLM), M2 (ciblé), `torch.compile`, puis le jeu `invasive`.
On ne coupe **jamais** : l'oracle d'évaluation indépendant, le split A/B, l'étude humaine, M5 et M9.

## 6. Répartition (à adapter)

| Personne | Responsabilité | Fichiers |
|---|---|---|
| A | Post-training (RAFT, DDPO, ablation) | `raft.py`, `ddpo.py`, `04/05_*.py` |
| B | Évaluation (oracles, métriques, étude humaine, analyse) | `classifiers.py`, `metrics.py`, `03/08/09_*.py` |
| C | Context engineering + jeux réels + img2img | `prompts.py`, `01_*.py`, `07_*.py`, `02_generate.py` |
| D | Inférence (best-of-N, rejet précoce, benchmark), interprétabilité, dépôt | `search.py`, `generator.py`, `06/10/11_*.py` |

Remplissez `docs/CONTRIBUTIONS.md` au fil de l'eau : la contribution individuelle compte pour 20 % de la note.

## 7. Questions à poser à l'enseignant / l'assistant

1. Est-ce qu'un projet d'*audit adversarial* (le « produit », c'est le générateur d'attaques et le benchmark) compte
   bien comme « application GAI » ?
2. Un juge GPT-4o validé par environ 210 annotations humaines (κ de Cohen) suffit-il pour prouver la fidélité ?
3. Les 4 jeux (`synth80`, `inat80`, `invasive`, `rare`) comptent-ils comme « 4+ datasets » ?
4. Le cours fournit-il un GPU ? Sinon, Colab T4 en plusieurs sessions suffit-il (profil réduit `t4`) ?
5. Faut-il ajouter de la quantification à la partie « accélération de l'inférence » ?
6. Faut-il une section éthique dédiée ? Peut-on montrer des photos iNaturalist et Rare Species dans le rapport en
   citant leur licence ?

## 8. Risques connus

| Risque | Parade |
|---|---|
| SD 1.5 dessine mal certaines espèces | ensemble attaquable (`--attackable-from`) ; les attaques img2img partent de vraies photos |
| GPT-4o juge mal | κ avec les humains ; on rapporte aussi la fidélité selon BioCLIP-2 |
| DDPO instable | RAFT reste la version sûre ; `clip_range` et `lr` faibles ; point de contrôle toutes les 10 époques |
| Taxonomie de `species.csv` ≠ celle de BioCLIP | `python run.py taxonomy` réaligne le fichier sur les noms de TreeOfLife |
| Une espèce invasive ou rare absente de TreeOfLife-10M | `p_true = nan`, l'espèce compte comme « inconnue » ; le script le signale |
| Coût ou limite de l'API OpenAI | réponses en cache (jamais payées deux fois) ; `workers` plus bas en cas de `RateLimitError` |
| Un identifiant de modèle a changé | `python run.py check` charge tout dès la semaine 10 |
