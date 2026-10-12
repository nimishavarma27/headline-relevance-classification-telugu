# Telugu News Headline Relevance Classification

Our reproduction of the TeClass paper (Kanumolu et al., LREC-COLING 2024) with one extra experiment the paper doesn't have: a zero-shot evaluation of Sarvam-105b.

Course project for CSE413 Speech and Natural Language Processing.

## Task

A Telugu article/headline pair, three possible labels:

- **HREL** - headline faithfully summarises the article's main event
- **MREL** - partly related, side event or weak link
- **LREL** - misleading, sensational, or clickbait

26,178 pairs. Three annotators per pair. Fleiss Kappa 0.77.

We ran four tracks:

1. Classical ML (TF-IDF + hand-crafted features)
2. Fine-tuned BERT (MuRIL and mDeBERTa)
3. Binary (HREL+MREL vs LREL)
4. Zero-shot LLM with 5 prompt variants  *(not in the paper)*

## Results at a glance

| System | Weighted F1 | Macro F1 |
|---|---|---|
| LR + 5 features | 0.57 | 0.58 |
| MuRIL | 0.61 | 0.61 |
| **mDeBERTa** | **0.62** | **0.62** |
| Sarvam-105b, anti-bias zero-shot (full test) | 0.35 | 0.35 |
| mDeBERTa, binary task | 0.80 | 0.75 |

A 278M parameter mDeBERTa fine-tuned on 18k Telugu labels beats a 105B parameter LLM by 27 F1 points. Labelled Telugu data still wins over model size.

## Dataset

70 / 15 / 15 split, stratified by label:

| Split | HREL | MREL | LREL | Total |
|---|---|---|---|---|
| Train | 5,962 | 7,105 | 5,257 | 18,324 |
| Dev | 1,277 | 1,523 | 1,127 | 3,927 |
| Test | 1,278 | 1,523 | 1,126 | 3,927 |

CSVs are in `teclass_dataset/`. Columns: `id, url, article, headline, label`.

Preprocessing: custom Telugu tokenizer (danda, double-danda, dates, repeated punctuation treated as single tokens) then SentencePiece BPE with vocab 2,945. Same recipe as the paper.

## Full results

### Classical ML

5-fold stratified CV, weighted F1.

| Features | Classifier | Ours | Paper |
|---|---|---|---|
| None | LR | 0.54 | 0.55 |
| None | SVM | 0.50 | 0.53 |
| None | MLP | 0.54 | 0.54 |
| Cosine | LR | 0.55 | 0.55 |
| Cosine | SVM | 0.51 | 0.54 |
| Cosine | MLP | 0.54 | 0.53 |
| 3-feature | LR | 0.56 | 0.58 |
| 3-feature | SVM | 0.52 | 0.57 |
| 3-feature | MLP | 0.56 | 0.56 |
| 5-feature | LR | 0.56 | 0.58 |
| 5-feature | SVM | 0.52 | 0.57 |
| 5-feature | MLP | 0.56 | 0.56 |

LR and MLP are within 1 to 2 F1 of the paper. SVM is 3 to 5 points behind, which is a LinearSVC feature-scaling thing (raw TF-IDF stacked with scalar features). L2 normalising the stacked matrix fixes it, but we left it as-is so the pipeline matches the paper's description.

Best setup: LR + 5 features. On the held-out test set that gets W-F1 0.57 and macro F1 0.58.

### BERT

Test set.

| Model | HREL | MREL | LREL | W-F1 | Macro F1 |
|---|---|---|---|---|---|
| MuRIL | 0.67 | 0.55 | 0.64 | 0.61 | 0.61 |
| mDeBERTa | 0.67 | 0.58 | 0.62 | 0.62 | 0.62 |

mDeBERTa confusion matrix (3,927 test rows):

```
             pred HREL  pred MREL  pred LREL
true HREL:      876        342        60
true MREL:      381        933       209
true LREL:       94        409       623
```

Pattern: HREL and MREL get confused with each other. LREL also leaks into MREL. Nothing wild misclassified as the opposite extreme.

### Binary task

HREL and MREL collapsed into one RELEVANT class, LREL stays as IRRELEVANT. Same mDeBERTa recipe with a 2-way head.

| Class | F1 |
|---|---|
| Relevant | 0.87 |
| Irrelevant | 0.63 |
| Weighted F1 | 0.80 |
| Macro F1 | 0.75 |

So going from 3-class to binary gains 18 F1 points. That tells you most of the pain in the 3-class task is on the HREL vs MREL boundary.

### Zero-shot LLM

Sarvam-105b-conversations, accessed via the Sarvam HTTP API, temperature 0.

| Prompt | HREL | MREL | LREL | W-F1 | Macro F1 |
|---|---|---|---|---|---|
| Zero-shot (naive) | 0.54 | 0.15 | 0.04 | 0.24 | 0.24 |
| Chain-of-thought | 0.54 | 0.12 | 0.11 | 0.26 | 0.26 |
| Few-shot (3 examples) | 0.58 | 0.25 | 0.04 | 0.29 | 0.29 |
| Anti-bias zero-shot (n=400) | 0.59 | 0.27 | 0.22 | 0.36 | 0.36 |
| Anti-bias few-shot (n=400) | 0.62 | 0.31 | 0.16 | 0.36 | 0.36 |
| **Anti-bias zero-shot, full test (n=3,927)** | **0.56** | **0.27** | **0.21** | **0.35** | **0.35** |

A few things we noticed while running this:

- Naive zero-shot just predicts HREL 87% of the time. Basically useless.
- One extra sentence in the prompt (the anti-bias one, see below) adds +12 F1 points. By far the biggest single thing that helped.
- Both anti-bias variants (zero-shot and few-shot) landed at 0.36 on the 400-sample subset. Different per-class profiles, same overall score. So we think this is a model ceiling, not a prompt problem.
- Full test re-run gave 0.35, which is within noise of 0.36.
- Even with the anti-bias instruction the model still predicts HREL for roughly 77% of inputs. The prior is sticky.

## The anti-bias prompt

This is the one that helped. Nothing fancy:

```
IMPORTANT: Classes are approximately balanced in the test set
(roughly 1/3 HREL, 1/3 MREL, 1/3 LREL). Do NOT default to HREL.
Look carefully for LREL cues like sensational language or
claims unsupported by the article.
```

The full prompt that goes to Sarvam looks like:

```
Classify the relevance between this Telugu news article and headline.

{DEFS}            # baseline or anti-bias version of the category definitions
{EXAMPLES}        # 3 labelled pairs from training set, few-shot only

Article: {article text, truncated to ~2200 chars}
Headline: {headline text}

Reply with ONE label only: HREL, MREL, or LREL.
```

## Running it

### ML baselines

Open `teclass_ml_all_baselines.ipynb` in Colab (CPU is fine). Point the three CSV paths at your copy of the dataset. Run all cells. Takes maybe 10 minutes.

### BERT

Open `TeClass_mDeBERTa_improved.ipynb` or `TeClass_MuRIL.ipynb` in Colab with a T4 GPU. Mount Drive. Set the output path to somewhere on Drive so your checkpoints survive a disconnect. One run is around 90 minutes on T4.

Hyperparameters we used (same for both models):

```
learning_rate   = 1.4e-5
weight_decay    = 0.043
warmup_ratio    = 0.085
dropout         = 0.11
batch_size      = 8
max_length      = 512
fp16            = True
epochs          = up to 4, early stop on dev macro F1
```

These came out of Optuna, trial 2 was the one that stuck.

### LLM

Get a Sarvam key from sarvam.ai. In Colab, add it as a secret called `SARVAM_API_KEY`. Then open any of the `teclass_llm_sarvam_*` notebooks and run.

Cost note: at the current pricing, 400 samples is roughly Rs. 3. The full-test run (3,927 samples) was around Rs. 30. The free credit on signup covers all the subset runs plus most of a full-test run.

### Live demo

`teclass_live_demo.ipynb` loads the fine-tuned mDeBERTa from Drive, fits a quick LR baseline on the training set, and sets up the Sarvam API call. Paste any (headline, article) pair into the demo cell and it prints predictions from all three systems. We used this during the presentation.

## What we didn't do

- Only tested one LLM. Would have been nice to also run GPT-4o or Claude but credit was limited.
- Four of the five prompts were on 400 samples. Only the best one got rerun on the full 3,927 test.
- No per-domain analysis. The dataset has politics, sports, films, etc. and we didn't break results down by that.
- No LLM fine-tuning. A LoRA pass on Sarvam would probably close most of the gap but we didn't have time or GPU budget for it.
- The fine-grained sub-labels in the paper's Table 5 (FME, STC, FSE, WKC, SEN, CBT, MLC, USO) aren't in the public release, so we couldn't reproduce that experiment.

## Things we'd try next

- Retrieval-augmented few-shot: for each test pair, pull the k nearest training examples instead of using a fixed set
- Use the LLM's label as an extra feature inside the classical classifier, see if ensembling helps
- Train on Hindi or Tamil TeClass-like data and see how much transfers to Telugu
- LoRA fine-tune Sarvam-105b on TeClass

## Team

| Name | Roll |
|---|---|
| C. A. Nimisha Varma | 2023BCD0052 |
| K. Neharika | 2023BCD0053 |
| L. Shruthi | 2023BCD0050 |
| Murali | 2023BCD0008 |

## Reference

G. Kanumolu, L. Madasu, N. Surange, M. Shrivastava. **TeClass: A Human-Annotated Relevance-based Headline Classification and Generation Dataset for Telugu**. LREC-COLING 2024, pp. 15711-15720.
