# Final Writeup — Detailed Outline

**Paper title (working):** *How Far Does a Small Closed-Book Encoder Get on ARC? A Controlled
Study of Backbone Choice and Fine-Tuning Sensitivity Under a Two-GPU-Hour Budget*

Every number below is already computed and traceable to a file in `results/`. The
"Number bank" (§7) maps each claim to its source so writing becomes copy-editing.
Read §9 (*Gaps to close*) before you start writing — two of them are ~10 minutes of
GPU time and materially strengthen the paper.

---

## 1. Word budget

The rubric counts **only Related Work + Methods + Results** toward 1500–2000 words.
Abstract, Introduction, Discussion, Limitations, Conclusion and References are free.

| Section | Words | Paragraphs | Notes |
|---|---:|---:|---|
| Related Work | 450 | 4 | ~110 w each |
| Methods | 600 | 5 | ~120 w each |
| Results | 700 | 6 | ~115 w each |
| **Counted total** | **1750** | 15 | leaves 250 w of slack inside the band |
| Abstract | 120 | 1 | not counted |
| Introduction | 200 | 2 | not counted |
| Discussion + Limitations | 280 | 4 | not counted |
| Conclusion | 80 | 1 | not counted |

Writing rule of thumb: **one paragraph ≈ 6–8 sentences ≈ 110–120 words.** Each bullet
in §3–§5 below is meant to become roughly one sentence. Do not add sections; the
rubric names these three and grades them.

---

## 2. Front matter (not counted)

**Abstract (~120 w).** One sentence of task, one of method, three of results
(75.0 / 56.1 test; DeBERTa > BERT by 9–20 pts on validation; negation questions
−24 pts), one of the takeaway (a 184M closed-book encoder clears the 2018 baselines
by ~29 points on Challenge but stays ~7 points below a 2020 retrieval ensemble, and
its accuracy moves more with backbone choice than with any hyperparameter).

**Introduction (~200 w, 2 paragraphs).**
1. ARC as a benchmark designed so that surface retrieval fails; why "grade-school
   science" is harder than it sounds; the question this paper asks.
2. Contributions, as three bullets: (i) a controlled backbone comparison in which
   only the pretrained weights change; (ii) a 63-run hyperparameter study showing
   optimal fine-tuning settings do **not** transfer between backbones; (iii) an
   error analysis identifying negation and long answer options as the reliable
   failure modes, plus a calibration analysis showing overconfidence concentrated
   on Challenge.

---

## 3. Related Work (~450 words, 4 paragraphs)

The rubric asks for three specific things: characterize the benchmark **with published
accuracy numbers**, organize prior work **by idea family** and synthesize what each
family got right/wrong, and name **the gap your method builds on**. One paragraph per
family, gap statement in the last paragraph.

### RW-1 — The benchmark and why its split matters (~110 w)
- ARC (Clark et al., 2018): 7,787 grade-school science questions, natural-language
  multiple choice, 3–5 options.
- The split is **adversarial by construction**: a question lands in Challenge iff
  *both* a retrieval (IR) solver *and* a word co-occurrence (PMI) solver answer it
  wrong; everything else is Easy.
- Sizes (state them here, they anchor every CI later): Easy 2,251 / 570 / 2,376 and
  Challenge 1,119 / 299 / 1,172 (train / val / test).
- Consequence to state explicitly: Easy and Challenge are not "two difficulty levels
  of one task" — Challenge is defined as the residual where lexical-overlap methods
  fail, so any method that is a better lexical matcher gains on Easy and not on
  Challenge. This sentence sets up your own R3 finding.
- Note that ARC ships a **labeled public test set**, which is why you can report test
  numbers at all (many benchmarks of this era do not).

### RW-2 — Retrieval + entailment, the pre-transformer family (~110 w)
- Numbers from Clark et al. (2018), Table 6, **test** accuracy, Challenge / Easy:
  IR over the ARC Corpus 20.26 / 62.55; TupleInference 23.83 / 60.81;
  DecompAttn 24.34 / 58.27; DGEM-OpenIE 26.41 / 57.45; BiDAF 26.54 / 50.11;
  DGEM 27.11 / 58.97. Random guessing is 25.02.
- The synthesis sentence: **no member of this family beat random by more than about
  two points on Challenge**, while the same systems reached 50–63% on Easy.
- What the family got right: retrieval over a science corpus is genuinely sufficient
  for most Easy questions — the 62.55% IR number is not a weak result, it is the
  ceiling of lexical matching, and it is what the Challenge filter was defined to
  exclude.
- What it got wrong: stacking a trained entailment model (DGEM, DecompAttn) on top of
  retrieved sentences added almost nothing, because the missing ingredient was
  background knowledge and inference, not sentence-pair scoring.

### RW-3 — Pretrained transformers, with and without retrieval (~120 w)
- Aristo (Clark et al., 2020) — retrieval plus an ensemble of solvers including a
  RoBERTa-large reader — reaches **88.3% Easy / 63.0% Challenge (test)**. That is a
  ~36-point Challenge jump over the 2018 baselines in two years.
- The synthesis sentence: essentially all of that jump came from large-scale
  **pretraining**, not from better symbolic inference — the retrieval and inference
  machinery around the reader was already present in the 2018 systems.
- Within this family, the standard *closed-book* recipe is a pretrained encoder with a
  multiple-choice scoring head (`BertForMultipleChoice` and its descendants), fine-tuned
  on the ARC training set with no retrieved context, so the model must answer from
  parametric knowledge alone. BERT, RoBERTa and DeBERTa-v3 are the usual backbones.
- Cross-format training (UnifiedQA and later instruction-tuned models) is the other
  branch: train one model over many QA formats, then transfer to ARC.
- ⚠️ **Verify before citing** — I could not confirm a specific published closed-book
  base-size-encoder ARC number, or UnifiedQA's exact ARC figures, from a primary
  source. Either verify one (then cite it) or write this paragraph without a number;
  do **not** cite a figure you have not seen in the paper itself. See §9.4.

### RW-4 — The evaluation-format critique, and your gap (~110 w)
- "In Case You Missed It: ARC 'Challenge' Is Not That Challenging" (Findings of ACL
  2025) shows the Easy–Challenge gap shrinks up to six-fold when a model is shown all
  answer options in one context ("options") rather than scoring each option
  independently ("separation"): Mistral Large 67 → 95%, Qwen 2.5 72B 63 → 95%,
  Llama 3.1 70B 64 → 93%, Mixtral 8×7B 66 → 85%.
- Why this belongs in *your* Related Work: `AutoModelForMultipleChoice` is a
  **separation-style scorer** — each option is encoded in its own sequence and never
  attends to its competitors. Your architecture sits on the side of this debate that
  the paper argues is penalised, which is a caveat you can own rather than be caught by.
- **The gap (state it as one explicit sentence):** prior work characterises either
  retrieval-augmented ensembles or 70B-scale models; what is under-documented is how
  far a *single, small, retrieval-free encoder* gets under a fixed and small compute
  budget, and how much of that result is determined by the choice of pretrained
  backbone versus the fine-tuning hyperparameters. That is what this paper measures.

---

## 4. Methods (~600 words, 5 paragraphs)

Rubric asks for: task formulation + input representation + preprocessing + exact
splits; architecture + key hyperparameters + training procedure; experimental design
(baselines, ablations, what is held fixed, metric).

### M-1 — Casting multiple choice as a learning problem (~130 w)
- A question with *n* options becomes *n* independent sequences
  `[CLS] question [SEP] option_i [SEP]`.
- All *n* pass through the **same** encoder (shared weights); each pooled `[CLS]`
  vector goes through **one** linear layer `hidden_size → 1`, producing *n* scalars.
- The *n* scalars are stacked into a logit vector, softmaxed **across options**, and
  trained with cross-entropy against the gold option index. So the classification is
  over positions-within-a-question, not over a fixed label set — this is what lets the
  same head handle 3-, 4- and 5-option questions.
- Head size: 769 parameters for a 768-dim backbone (768 weights + 1 bias), randomly
  initialised; everything else is pretrained. Worth one sentence — it makes the point
  that the paper is measuring pretrained representations, not a learned classifier.
- Cross-reference RW-4 in half a sentence: options are encoded independently.

### M-2 — Data and preprocessing (~130 w)
- Source: `allenai/ai2_arc` (HuggingFace), configs `ARC-Easy` and `ARC-Challenge`.
- Give the split table again as numbers you actually used: Easy 2,251/570/2,376;
  Challenge 1,119/299/1,172; combined 3,370/869/3,548. These are the official splits,
  unmodified — no re-splitting, no data added.
- Two normalisation steps the dataset forces:
  (i) **label schemes are mixed** — some questions label options `A–E`, others `1–5`;
  both are mapped to a 0-based index;
  (ii) **option count varies (3/4/5)** — within a batch, examples are padded to the
  batch maximum and a boolean `choice_mask` marks real options; masked logits are set
  to the dtype minimum before the softmax so padded slots receive zero probability.
- One implementation detail worth a sentence because it is a real trap: padded slots
  are filled with a **copy of the first real option**, not with all-padding rows — an
  all-padding row makes the attention softmax operate over an entirely masked row and
  produce NaN.
- Tokenisation: `truncation="only_first"` (the question is truncated, the option never
  is), dynamic padding to the longest sequence in the batch, `max_length = 128`.
  ⚠️ Insert the truncation rate at 128 from `notebooks/eda.ipynb` — see §9.3.

### M-3 — Architectures compared (~110 w)
- `bert-base-uncased`: 110M parameters, 30k WordPiece vocabulary, pretrained with
  masked-LM + next-sentence prediction on ~16GB of text.
- `microsoft/deberta-v3-base`: 184M total parameters (≈86M transformer backbone plus
  ≈98M embedding parameters, a consequence of the 128k SentencePiece vocabulary),
  disentangled attention with explicit relative positions, pretrained with
  ELECTRA-style replaced-token detection on ~160GB of text.
- The controlled-comparison sentence, which is the methodological point of the paper:
  **identical head, identical preprocessing, identical training loop, identical grid
  protocol — the only variable is the pretrained checkpoint.**

### M-4 — Training procedure and hyperparameters (~120 w)
- Optimiser AdamW with decoupled weight decay 0.01, applied to all parameters
  **except** biases and LayerNorm weights.
- Schedule: linear warmup over the first 10% of updates, then linear decay to zero.
- Batch size 8 with gradient accumulation 2 → effective batch 16; gradient-norm
  clipping at 1.0.
- Mixed precision: fp16 autocast with a gradient scaler on CUDA, keeping fp32 master
  weights (the DeBERTa-v3 checkpoint ships in fp16 and is cast to fp32 at load).
- Checkpointing: validation accuracy is computed after every epoch and the best-epoch
  weights are kept; the final-epoch weights are never used for evaluation.
- Hardware: a single NVIDIA GPU on UCSD DataHub. One sentence flagging that an Apple
  MPS run of the identical code silently produced chance-level validation accuracy —
  full detail goes in the Discussion, but Methods should say which device produced the
  reported numbers.

### M-5 — Experimental design (~110 w)
- **Baselines.** Random guessing, computed both analytically as the mean of 1/*n* over
  the split (options counts vary, so this is not exactly 25%) and empirically over
  1,000 trials; and a majority-position baseline that always predicts the most frequent
  gold position, which tests whether the dataset leaks a positional prior.
- **Grid / ablations.** Learning rate × epochs × training regime. BERT lr ∈
  {2, 3, 5, 7}×10⁻⁵, DeBERTa lr ∈ {1, 1.5, 2, 3}×10⁻⁵ (centred on each family's known
  working range), epochs ∈ {3, 4, 5}, and three training regimes: Easy-only,
  Challenge-only, and combined Easy+Challenge. 72 cells planned, **63 completed**.
- **Held fixed:** seed within a sweep, batch size, accumulation, max_length, weight
  decay, warmup ratio, scheduler, clipping.
- **Metric:** accuracy, reported with 95% Wilson score intervals; differences tested
  with a two-proportion *z*-test.
- **Protocol sentence (say it explicitly, it is a credibility marker):** every model
  and hyperparameter decision was made on validation; the test sets were evaluated
  exactly once, at the end, with a single selected checkpoint.

---

## 5. Results (~700 words, 6 paragraphs)

Rubric asks for: performance **compared to prior literature with similar approaches**,
error analysis naming **which question types fail** and whether that matches prior
work, ≥1 plot, ≥1 table.

### R-1 — Headline numbers → **Table 1** (~130 w)
- Test, DeBERTa-v3-base combined champion (lr 2e-5, 5 epochs, best epoch by val):
  **Easy 75.0%** [73.3, 76.7] (n=2,376), **Challenge 56.1%** [53.2, 58.9] (n=1,172),
  **combined 68.8%** [67.2, 70.3] (n=3,548).
- Baselines on test: random 25.0 / 25.0 / 25.0; majority-position 24.6 / 26.5 / 25.3 —
  i.e. the positional baseline is worth nothing, the dataset leaks no position prior.
- Validation champions, both backbones: DeBERTa 76.1 / 52.2 / 70.0 vs BERT
  58.2 / 42.8 / 50.4 (Easy / Challenge / combined).
- Note the val→test consistency for DeBERTa (76.1 → 75.0 Easy, 70.0 → 68.8 combined):
  a ~1-point drop, i.e. the sweep did not badly overfit the validation set. Challenge
  actually goes *up* (52.2 val → 56.1 test) — but the val champion there is a different,
  Challenge-only-trained model, so say this carefully (see §10).
- **Table 1 is placed here.** It is the rubric's required table.

### R-2 — Where this sits relative to prior literature (~120 w)
- Challenge test 56.1% vs the best 2018 baseline (DGEM, 27.11%): **+29.0 points**, and
  vs random 25.02%: +31.1 points.
- Easy test 75.0% vs the strongest 2018 baseline on Easy (IR over the ARC Corpus,
  62.55%): **+12.5 points**.
- Below Aristo's 88.3 / 63.0 — by 13.3 points on Easy and 6.9 on Challenge. The
  honest, interesting framing: Aristo is a *retrieval-augmented ensemble* built around
  a RoBERTa-**large** reader, while this is **one 184M encoder, closed-book, 103 GPU-
  minutes for the entire 63-run study**. The ordering is exactly what prior work
  predicts; the contribution is quantifying the price of dropping retrieval and scale.
- One sentence connecting to RW-2's synthesis: the 2018 family's Challenge scores
  clustered at random because lexical matching was excluded by construction; a
  pretrained encoder with no retrieval at all clears that bar by 29 points, which
  locates the missing ingredient in **pretraining**, not in retrieval.

### R-3 — Backbone comparison (~120 w)
- Validation, DeBERTa − BERT: Easy **+17.9 pts** (p < 0.0001), Challenge **+9.4 pts**
  (p = 0.0219), combined **+19.6 pts** (p < 0.0001). All significant at α = 0.05.
- Because the head, data pipeline, training loop and grid protocol are identical, the
  gap is attributable to the pretrained checkpoint. And it is not merely parameter
  count: 184M vs 110M is 1.7×, and most of DeBERTa's extra parameters sit in the
  embedding table rather than the transformer stack.
- The subtle observation worth making: **the Challenge gap is the smallest and the
  least significant of the three.** A better backbone buys the most where lexical and
  factual competence pays off (Easy) and the least on the adversarially-filtered
  subset — consistent with RW-1's point that Challenge is the residual where
  surface-level competence was already excluded.

### R-4 — Hyperparameter sensitivity and transfer → **Figure 1** (~120 w)
- Champion configurations: BERT peaks at lr 7e-5 on Easy (58.2, 5 epochs) and combined
  (50.4, 4 epochs) and at 3e-5 on Challenge (42.8, 5 epochs); DeBERTa peaks at 3e-5 on
  Easy (76.1, 5 epochs) and Challenge (52.2, 4 epochs) and at 2e-5 on combined
  (70.0, 5 epochs).
- **Optimal learning rates do not transfer across backbones** — BERT's best settings
  sit at the top of its range and DeBERTa's near the bottom of a range that is already
  2–3× lower. A default lifted from a BERT tutorial costs DeBERTa real accuracy. This
  is the paper's cleanest secondary finding and Figure 1 is what shows it.
- DeBERTa's grid is also visibly **flatter** — it is less sensitive to the choice, not
  just better on average, which matters for anyone with one shot at a hyperparameter.
- **Combined vs separate training** (an ablation, worth its own two sentences):
  weighting the two separate champions 570:299 predicts a combined-validation accuracy
  of 67.9% for DeBERTa, and the combined-trained model reaches 70.0% (**+2.1**); the
  same calculation for BERT predicts 52.9% against an actual 50.4% (**−2.5**). The
  stronger backbone converts additional out-of-distribution training data into gains
  while the weaker one is diluted by it.
- **Figure 1 (the heatmaps) is placed here.** It is the rubric's required plot.

### R-5 — Error analysis: which question types fail → **Table 2** (~140 w)
This paragraph is explicitly required by the rubric — give it the most space.
- **Negation / "except" questions are the single reliable failure mode, on both
  subsets.** Easy: 50.7% (n=73) vs 75.1% overall, **−24.4 pts**, p < 0.0001.
  Challenge: 43.0% (n=79) vs 56.1% overall, **−13.0 pts**, p = 0.0158. It is the
  weakest category in both, and the only category that is weakest in both.
- **Long answer options hurt monotonically on Easy:** ≤2 words 80.0% (n=1,060, +5.0),
  3–5 words 74.1%, 6–10 words 68.9% (−6.1, p < 0.0001), >10 words 62.5%
  (n=64, −12.5, p = 0.019). Challenge shows the same tail (>10 words 49.1%) plus a
  short-option penalty (≤2 words 50.6%, −5.4, p = 0.022).
- **Long questions hurt on Easy:** >30 words 68.7% (−6.4, p = 0.0015).
- **No positional bias:** predicted answer-position distribution deviates from gold by
  at most 1.2 pts (Easy) and 1.8 pts (Challenge) — the model is not exploiting a prior,
  which corroborates the majority-position baseline in R-1.
- **Does this match prior work?** (the rubric asks for this comparison explicitly) —
  yes on two counts: negation insensitivity is a long-documented weakness of
  MLM-pretrained encoders, and the option-length effect is what the separation-scoring
  critique from RW-4 predicts, since a long option is scored in isolation and never
  compared against its competitors. The 128-token truncation is a competing explanation
  for the long-question effect and should be named as such.
- **Caveat, stated in the paper, not hidden:** these categories are an automatic
  **lexical proxy** (regular expressions over the question text), not the hand-labelled
  knowledge/reasoning taxonomy of Clark et al. (2018).

### R-6 — Calibration and efficiency → **Figure 2** (~120 w)
- **Overconfidence is systematic and concentrated on Challenge.** In the [0.9, 1.0)
  confidence bin, Easy is 91.3% accurate at 97.6% mean confidence (gap 6.3 pts) while
  Challenge is 77.5% accurate at 96.7% (gap **19.2 pts**). The [0.7, 0.9) bin: Easy gap
  12.6 vs Challenge gap **23.4**. Easy is nearly calibrated below 0.7 (gaps of
  0.0 and −0.3); Challenge is miscalibrated everywhere.
- **87 Challenge predictions (7.4% of the set, 16.9% of all its errors) are wrong at
  confidence above 0.9** — the model does not know what it does not know, precisely
  on the subset where it is worst.
- **Efficiency:** BERT 17.3 s/epoch vs DeBERTa 20.8 s/epoch — only ~20% slower despite
  1.7× the parameters, because the extra parameters are embedding lookups, not
  attention FLOPs. 63 completed runs consumed **103 GPU-minutes in total**, so the
  entire study reproduces in under two GPU-hours.
- **Figure 2 (calibration + weakest categories) is placed here.**

---

## 6. Back matter (not counted)

**Discussion (~180 w, 3 threads).**
1. *Overfitting and why best-epoch checkpointing matters.* Champions plateau at epoch
   3–4 while training loss continues toward zero (`fig_training_dynamics.png`). More
   epochs buy nothing after the plateau; the last-epoch weights would have been worse
   than the ones reported.
2. *The separation-scoring hypothesis.* The Easy−Challenge test gap here is 18.9
   points, in the range the Findings-ACL 2025 work attributes partly to evaluation
   format rather than question difficulty. The natural next experiment is a
   joint-context reranker that sees all options at once — a concrete, cheap
   follow-up, and a good closing line for the lightning talk.
3. *A reproducibility incident worth reporting.* Identical code and seed on Apple's
   MPS backend produced falling training loss with validation accuracy pinned at
   chance, while CUDA reached 55.8%. Silent numerical failure in an accelerator
   backend is indistinguishable from a modelling bug without a cross-device control —
   which is why every reported number comes from CUDA.

**Limitations (~100 w).** Single seed per sweep cell, so within-sweep gaps of ~±2 pts
are noise; the BERT-Easy sweep completed only 3 of 12 cells after a disk-quota failure;
**no BERT test numbers**, so all cross-backbone claims are validation-only; every test
cell comes from one checkpoint (the combined champion), so a Table 1 row is not one
model's profile; error categories are lexical proxies rather than a hand-labelled
taxonomy.

**Conclusion (~80 w).** Restate the three contributions with the headline number.

---

## 7. Figures and tables — manifest

| Label | Source file in `results/analysis/` | Where | Caption seed |
|---|---|---|---|
| **Table 1** *(required)* | `t1_main_results.tex` / `.md` | R-1 | Accuracy (%) with 95% Wilson intervals on ARC Easy/Challenge/combined, validation and test, against random and majority-position baselines. |
| **Figure 1** *(required)* | `fig_hparam_heatmaps.png` | R-4 | Validation accuracy across the learning-rate × epochs grid for each backbone and training regime; champion cell outlined, failed cells grey. |
| **Figure 2** | `fig_error_analysis.png` | R-6 | Left: confidence-bin accuracy vs mean confidence (calibration). Right: accuracy by question category, deviation from subset mean. |
| Table 2 | `t5_error_categories.md` — **trim to the significant rows** | R-5 | Accuracy by question category with n and two-proportion p vs the rest of the subset. |
| Figure 3 *(if space)* | `fig_training_dynamics.png` | Discussion | Per-epoch validation accuracy and training loss for the six sweep champions. |
| Appendix A | `t2_all_runs.md` | Appendix | All 63 completed runs. |
| Appendix B | `t3_efficiency.md`, `t4_calibration.md` | Appendix | Throughput and full calibration bins. |

Trim Table 2 to the rows that carry the argument: negation/except (both subsets),
the answer-option-length ladder, and Easy question-length >30 words. The full
28-row table goes to the appendix.

---

## 8. Number bank (every figure you will cite, and where it came from)

| Claim | Value | Source |
|---|---|---|
| Test, DeBERTa combined champion | Easy 75.0 / Challenge 56.1 / combined 68.8 | `results/eval_*_test.json` |
| Test 95% CIs | [73.3, 76.7] / [53.2, 58.9] / [67.2, 70.3] | `analysis/t1_main_results.md` |
| Val champions, DeBERTa | 76.1 / 52.2 / 70.0 | `analysis/analysis.md` |
| Val champions, BERT | 58.2 / 42.8 / 50.4 | `analysis/analysis.md` |
| Champion configs | BERT 7e-5/5ep (easy), 3e-5/5ep (chal), 7e-5/4ep (comb); DeBERTa 3e-5/5ep, 3e-5/4ep, 2e-5/5ep | `analysis/t2_all_runs.md` |
| DeBERTa − BERT gaps + p | +17.9 (p<0.0001) / +9.4 (p=0.0219) / +19.6 (p<0.0001) | `analysis/analysis.md` |
| Combined-vs-separate | DeBERTa 67.9→70.0 (+2.1); BERT 52.9→50.4 (−2.5) | `analysis/analysis.md` |
| Negation, Easy | 50.7%, n=73, −24.4, p<0.0001 | `analysis/t5_error_categories.md` |
| Negation, Challenge | 43.0%, n=79, −13.0, p=0.0158 | `analysis/t5_error_categories.md` |
| Option-length ladder, Easy | 80.0 / 74.1 / 68.9 / 62.5 | `analysis/t5_error_categories.md` |
| Question length >30w, Easy | 68.7%, −6.4, p=0.0015 | `analysis/t5_error_categories.md` |
| Position bias | max deviation 1.2 (Easy) / 1.8 (Challenge) pts | `analysis/analysis.md` |
| Calibration gaps, [0.9,1.0) | Easy 6.3 pts / Challenge 19.2 pts | `analysis/t4_calibration.md` |
| Confidently wrong | 87 Challenge (7.4% of set, 16.9% of errors); 104 Easy | `analysis/analysis.md` |
| Throughput | BERT 17.3 s/ep, DeBERTa 20.8 s/ep; 103 GPU-min total | `analysis/t3_efficiency.md` |
| Runs completed | 63 of 72 | `analysis/analysis.md` |
| Clark 2018 Challenge baselines (test) | IR 20.26, TupleInf 23.83, DecompAttn 24.34, random 25.02, DGEM-OpenIE 26.41, BiDAF 26.54, DGEM 27.11 | ARC paper, Table 6 |
| Clark 2018 Easy baselines (test) | IR 62.55, TupleInf 60.81, DGEM 58.97, DecompAttn 58.27, DGEM-OpenIE 57.45, BiDAF 50.11 | ARC paper, Table 6 |
| Aristo (test) | Easy 88.3 / Challenge 63.0 | Clark et al. 2020 |
| Separation vs options | Mistral Large 67→95, Qwen 2.5 72B 63→95, Llama 3.1 70B 64→93 | Findings ACL 2025 |

---

## 9. Gaps to close before writing

1. **No BERT test numbers.** Every cross-backbone claim is currently validation-only.
   Retraining the BERT combined champion (lr 7e-5, 4 epochs — ~2 minutes) and running
   two test evaluations turns R-3 into a test-set comparison and removes the biggest
   limitation. **Highest value per minute of anything left on the list.**
2. **Single seed everywhere.** `configs/final.json` runs the DeBERTa combined champion
   across seeds 42/43/44 (~3 × 2.5 min). It converts "the champion scored 70.0" into
   "70.0 ± sd" and lets you state that within-sweep differences are noise with evidence
   rather than assertion.
3. **Truncation rate at `max_length = 128`** — one number from `notebooks/eda.ipynb`,
   needed for a Methods sentence and as the competing explanation in R-5.
4. **Unverified citation.** No confirmed published number for a closed-book base-size
   encoder on ARC, or for UnifiedQA. Verify from the primary paper or write RW-3
   without a number.
5. **Optional, high rubric value.** Hand-label 30–50 Challenge errors into Clark et
   al.'s knowledge/reasoning categories. The rubric asks whether your failure types
   "match findings in prior work"; a hand-labelled sample answers that directly, while
   the regex proxy only approximates it. Budget ~45 minutes.

---

## 10. Rubric traceability

| Requirement | Where it is satisfied | Status |
|---|---|---|
| 1500–2000 words across RW + Methods + Results | §1 budget = 1750 | planned |
| Characterise ARC, cite published accuracies | RW-1, RW-2, RW-3 | ready |
| Organise prior work by idea family, synthesise | RW-2/3/4, one family each with a "got right / got wrong" sentence | ready |
| Name the gap your method builds on | RW-4, final sentence | ready |
| Multiple choice as a learning problem, input representation, preprocessing, exact splits | M-1, M-2 | ready |
| Architecture, key hyperparameters, training procedure | M-3, M-4 | ready |
| Baselines, ablations, held-fixed, metric | M-5 | ready |
| Compare to prior literature with similar approaches | R-2 | ready |
| **Error analysis: which question types fail** | R-5 | ready |
| Whether errors match prior work | R-5, final two sentences | ready |
| ≥1 plot of a core finding | Figure 1 (heatmaps) | ready |
| ≥1 table of core metrics | Table 1 | ready |
| Code submitted | repo | ready |

---

## 11. Sentences *not* to write

- ✗ "DeBERTa outperforms BERT on the test set." No BERT test run exists — say
  *validation* until §9.1 is done.
- ✗ "Table 1's test row shows our model's profile across subsets." The three test cells
  all come from **one** checkpoint (the combined champion) while the validation cells
  come from three different per-subset champions. Say which checkpoint produced which
  cell.
- ✗ "The champion configuration is significantly better than the runner-up." Within-sweep
  gaps of ~2 points on n=570 or n=299 are not significant, and the sweeps are single-seed.
- ✗ "Our error categories show the model fails at multi-hop reasoning." The categories
  are regex over question text. Call them a lexical proxy every time.
- ✗ "Challenge test (56.1) beats Challenge validation (52.2), so the model generalises."
  Different checkpoints and different-sized sets; if you mention it at all, attribute it
  to the combined-trained model having seen ~3× the training data of the Challenge-only
  champion.
- ✗ Any accuracy number for UnifiedQA or a specific published closed-book encoder until
  §9.4 is resolved.

---

## 12. Lightning talk (3–5 min) — slide skeleton

1. **The benchmark, in one slide.** ARC, the adversarial Easy/Challenge filter, and the
   2018 result that nothing beat random on Challenge. (40 s)
2. **The question.** How far does one small closed-book encoder get, and what actually
   determines the answer? (20 s)
3. **Setup.** One diagram of the multiple-choice head; the controlled-comparison line.
   (40 s)
4. **Table 1.** 75.0 / 56.1 on test; +29 over the 2018 Challenge baselines; −6.9 vs a
   retrieval ensemble. (50 s)
5. **Figure 1.** Optimal learning rate does not transfer between backbones. (50 s)
6. **Figure 2.** Negation questions −24 points; overconfident precisely on Challenge.
   (50 s)
7. **Close.** Options-in-context reranking as the next experiment; the MPS
   reproducibility anecdote as the one-liner people remember. (30 s)

---

## References to cite

- Clark, P. et al. (2018). *Think you have Solved Question Answering? Try ARC, the AI2
  Reasoning Challenge.* arXiv:1803.05457.
- Clark, P. et al. (2020). *From 'F' to 'A' on the N.Y. Regents Science Exams: An
  Overview of the Aristo Project.* AI Magazine 41(4). arXiv:1909.01958.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding.* NAACL.
- He, P. et al. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training
  with Gradient-Disentangled Embedding Sharing.* ICLR.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
- *In Case You Missed It: ARC 'Challenge' Is Not That Challenging.* Findings of ACL 2025.
- Khashabi, D. et al. (2020). *UnifiedQA: Crossing Format Boundaries with a Single QA
  System.* Findings of EMNLP. *(cite only if §9.4 is resolved)*
