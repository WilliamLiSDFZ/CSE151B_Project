# Fine-Tuning BERT and DeBERTa-v3 on the AI2 Reasoning Challenge

**Yuze Li** — CSE 151B, Summer 2026, Model-Building Track

*Related Work + Methods + Results: 1,964 words (limit 1500–2000).*

---

## Abstract

We fine-tune two pretrained encoders, BERT-base and DeBERTa-v3-base, on the AI2
Reasoning Challenge (ARC), a dataset of grade-school science exam questions. We use no
retrieval, so the model has to answer using only the knowledge it picked up during
pretraining. We ran 63 training runs over a grid of learning rates, epoch counts and
training sets. Our best model reaches 75.0% accuracy on the Easy test set and 56.1% on
the Challenge test set. On validation, DeBERTa-v3-base beats BERT-base by between 9.4 and
19.6 points depending on the subset, even though the two were trained with exactly the
same code. We also found that the best learning rate is very different for the two
models, and that the most common failure case is questions containing a negation word
such as "except", where accuracy drops by up to 24 points.

## 1. Introduction

ARC is a dataset of science questions written for grade-school exams. At first this
sounds like an easy problem, because the questions are short and the vocabulary is
simple. It is not easy. The authors of the dataset deliberately separated out the
questions that simple search-based methods could not answer, and when the dataset was
released in 2018 none of the systems they tested could beat random guessing on that part
of the data.

In this project we wanted to find out how far a single small pretrained language model
gets on ARC without any retrieval system helping it. Most of the strong published results
either search a science corpus for supporting text, or use a very large language model,
or both. We wanted to see what a base-sized encoder can do on its own, and what actually
determines the final accuracy. Our contributions are:

1. A controlled comparison of two pretrained backbones, in which the classification head,
   the data pipeline and the training loop are identical and only the pretrained weights
   change.
2. A 63-run hyperparameter study showing that the best fine-tuning settings for one
   backbone are not the best settings for the other.
3. An error analysis showing which kinds of questions the model gets wrong, and a
   calibration analysis showing that it is most overconfident exactly on the subset where
   it is weakest.

## 2. Related Work

**The ARC dataset.** ARC (Clark et al., 2018) is 7,787 multiple-choice science questions
with three to five options each. The split is what makes it interesting: the authors ran
two simple solvers on every question, one retrieval-based and one using word
co-occurrence, and every question both got wrong went into Challenge, the rest into Easy.
Easy has 2,251 / 570 / 2,376 and Challenge 1,119 / 299 / 1,172 train / validation / test
questions. Challenge is therefore not just "the harder questions" — it is where word
matching between question and answer fails by construction, so a better word matcher
should gain on Easy but not on Challenge. Test labels are public, so we can report test
accuracy.

**Retrieval and entailment systems.** The original paper's baselines all pair retrieval
with a sentence-level inference model. On Challenge test they score 20.26% (IR over the ARC
corpus) to 27.11% (DGEM) against 25.02% random, so none beats chance by more than about two
points; on Easy the same systems reach 36–63%, the best being IR at 62.55%. What this family
got right is that retrieval alone answers most Easy questions, and 62.55% is near the
ceiling of word matching. What it got wrong is that stacking an entailment model on the
retrieved sentences barely helped: the missing ingredient was background knowledge, not a
better sentence-pair scorer.

**Pretrained transformers.** Aristo (Clark et al., 2020) pairs retrieval with an ensemble
of solvers, one a RoBERTa-large reader, and reaches 87.0% Easy and 64.3% Challenge — about
37 points on Challenge in two years, almost all of it from large-scale pretraining rather
than better symbolic inference, since the retrieval machinery already existed in 2018. This
family also contains the closed-book recipe we use: a multiple-choice head on a pretrained
encoder, fine-tuned with no retrieved context.

**How the options are scored, and our gap.** Borchmann (2025) argues that much of the
Easy–Challenge gap comes from the evaluation setup: scoring each option separately does far
worse than letting all options share one context. Re-measuring the same checkpoints under
both schemes, Qwen 14B goes from 47.3% to 86.6%, Llama 2 70B from 57.4% to 79.6% and
Llama 3.1 70B from 64% to 93%. These are large models scored without fine-tuning, so we
borrow the mechanism, not the magnitude. This matters for us because the standard
multiple-choice head encodes each option in its own sequence, so our model never sees the
other options while scoring one. The gap we address: prior work covers retrieval-augmented
very large models, but not how far a single small closed-book encoder gets on a small
compute budget, or how much of that is decided by the backbone rather than the
hyperparameters.

## 3. Methods

### 3.1 Multiple choice as a learning problem

A question with *n* options becomes *n* sequences `[CLS] question [SEP] option_i [SEP]`.
All *n* pass through the same encoder, and each pooled `[CLS]` vector goes through one
linear layer down to a single number. The *n* scalars are softmaxed across options and
trained with cross-entropy against the index of the correct option. Predicting a position
inside the question rather than a fixed label is what lets one head handle three, four or
five options. The head is tiny — 769 parameters for a 768-dimensional encoder — and randomly
initialised; everything else comes from pretraining. Options are encoded independently and
never attend to each other.

### 3.2 Data and preprocessing

We use `allenai/ai2_arc` with the `ARC-Easy` and `ARC-Challenge` configurations and the
official splits, unmodified: 2,251 / 570 / 2,376, 1,119 / 299 / 1,172, and 3,370 / 869 /
3,548 combined. Two things need normalising. Some questions label options `A`–`E` and others
`1`–`5`, so we map both to a zero-based index. And the option count varies, so within a batch
we pad to the largest count and keep a boolean mask; before the softmax the padded logits
are set to the dtype minimum. Padded slots must copy a real option rather than stay empty:
an all-padding row makes the attention softmax return NaN, which cost us a lot of debugging. We tokenise with `truncation="only_first"` so the question is truncated but the
option never is, pad dynamically per batch, and cap length at 128 tokens.
[TODO: insert the truncation rate at 128 from `notebooks/eda.ipynb`.]

### 3.3 Models

`bert-base-uncased` has 110M parameters, a 30k WordPiece vocabulary, and masked-language-
model plus next-sentence pretraining on about 16GB of text. `microsoft/deberta-v3-base` has
184M, but only about 86M in the transformer stack — the other 98M are embeddings, because of
its 128k SentencePiece vocabulary. It uses disentangled attention with explicit relative
positions and ELECTRA-style replaced-token-detection pretraining on about 160GB. Head,
preprocessing, training loop and sweep protocol are identical for both, so the only variable
is the starting checkpoint.

### 3.4 Training procedure

We train with AdamW, decoupled weight decay 0.01 on everything except biases and LayerNorm
weights, linear warmup over the first 10% of updates then linear decay to zero, batch size 8
with gradient accumulation 2 (effective batch 16), and gradient-norm clipping at 1.0. On
CUDA we use fp16 autocast with a gradient scaler and fp32 master weights; the DeBERTa-v3
checkpoint ships in fp16 and we cast it to fp32 on load. We evaluate every epoch and keep
the best-epoch weights, so the final-epoch weights are never reported. Every number comes
from one NVIDIA GPU on UCSD DataHub; we also had an Apple MPS device but stopped using it,
for the reason in Section 5.

### 3.5 Experimental design

Our two baselines are random guessing (computed analytically as the mean of 1/*n* and
empirically over 1,000 trials) and a majority-position baseline that always predicts the
most common correct position, which tests whether the dataset leaks a positional pattern.
The grid varies learning rate, epochs and training set. BERT used 2, 3, 5, 7 × 10⁻⁵ and
DeBERTa 1, 1.5, 2, 3 × 10⁻⁵, the ranges usually recommended for each family; epochs 3, 4 or
5; regimes Easy only, Challenge only, combined. That is 72 configurations, of which 63
completed — the rest failed on a server disk quota. Batch size, accumulation, max length,
weight decay, warmup, scheduler and clipping were fixed everywhere, and the seed was fixed
within each sweep. We report accuracy with 95% Wilson intervals and compare pairs with a
two-proportion *z*-test. Every decision was made on validation; we evaluated on test exactly
once, at the end, with one checkpoint.

## 4. Results

### 4.1 Main results

Table 1 gives our main numbers. The selected model is the DeBERTa-v3-base champion trained
on the combined set (learning rate 2 × 10⁻⁵, 5 epochs, best epoch chosen on validation). On
test it reaches 75.0% Easy, 56.1% Challenge, 68.8% overall. Both baselines sit at chance —
random 25.0% everywhere, majority-position 24.6% Easy and 26.5% Challenge — so the dataset
leaks no positional pattern. On validation the DeBERTa champions score 76.1 / 52.2 / 70.0
and the BERT champions 58.2 / 42.8 / 50.4 (Easy / Challenge / combined). On Easy and
combined, DeBERTa drops about a point from validation to test, so the sweep did not badly
overfit validation.

**Table 1.** Accuracy (%) with 95% Wilson intervals. Validation cells come from each subset's own sweep champion; all three test cells come from the single combined DeBERTa champion, so the test row is not three separate models.

| Model | Easy val | Easy test | Chal. val | Chal. test | Comb. val | Comb. test |
|:---|:---|:---|:---|:---|:---|:---|
| Random guess | 25.0 [21.6, 28.7] | 25.0 [23.3, 26.8] | 25.1 [20.5, 30.3] | 25.0 [22.6, 27.6] | 25.0 [22.3, 28.0] | 25.0 [23.6, 26.5] |
| Majority position | 26.7 [23.2, 30.4] | 24.6 [22.9, 26.4] | 24.4 [19.9, 29.6] | 26.5 [24.1, 29.1] | 25.9 [23.1, 28.9] | 25.3 [23.9, 26.7] |
| BERT-base | 58.2 [54.2, 62.2] | — | 42.8 [37.3, 48.5] | — | 50.4 [47.1, 53.7] | — |
| DeBERTa-v3-base | **76.1** [72.5, 79.5] | **75.0** [73.3, 76.7] | **52.2** [46.5, 57.8] | **56.1** [53.2, 58.9] | **70.0** [66.8, 72.9] | **68.8** [67.2, 70.3] |

### 4.2 Comparison with previous work

Our 56.1% on Challenge is 29.0 points above the best baseline in the original ARC paper
(DGEM, 27.11%) and 31.1 above random; our 75.0% on Easy is 12.5 above the best 2018 Easy
baseline (IR, 62.55%). We are still below Aristo by 12.0 points on Easy and 8.2 on
Challenge, but that is not like-for-like: Aristo is a retrieval-augmented ensemble around a
RoBERTa-large reader, ours is one 184M encoder with no retrieval, and our whole 63-run study
used 103 minutes of GPU time. The informative part is that the 2018 systems scored at chance
on Challenge because word matching was excluded by construction, and a closed-book encoder
clears that bar by 29 points — pretraining, not retrieval, was the missing ingredient.

### 4.3 BERT compared with DeBERTa

On validation DeBERTa beats BERT by 17.9 points on Easy (*p* < 0.0001), 9.4 on Challenge
(*p* = 0.0219) and 19.6 on combined (*p* < 0.0001), all significant at 0.05. Since head,
preprocessing, training loop and sweep protocol were identical, we attribute this to the
pretrained checkpoint, not to size: 184M against 110M is 1.7×, and most of the extra
parameters are embeddings. The Challenge gap is the smallest and least significant of the
three, which fits Section 2 — a better backbone helps most where lexical and factual
competence pays off, least on the subset filtered to exclude it. We have no BERT test
numbers, so these comparisons are validation-only.

### 4.4 Hyperparameter sensitivity

Figure 1 shows validation accuracy across the learning rate × epochs grid. BERT peaks at
7 × 10⁻⁵ on Easy and combined and 3 × 10⁻⁵ on Challenge; DeBERTa peaks at 3 × 10⁻⁵ on Easy
and Challenge and 2 × 10⁻⁵ on combined. The best learning rate does not transfer between
backbones — BERT wants the top of its range, DeBERTa the bottom of a range already two to
three times lower — so copying a value from a BERT tutorial would have cost several points.
DeBERTa's grid is also flatter where it matters: on Challenge its best accuracy varies by
3.0 points across learning rates against BERT's 7.4, so it is less sensitive as well as
better. Our other ablation is the training regime: weighting the two per-subset champions by validation size
predicts 67.9% combined for DeBERTa against an actual 70.0% (+2.1), and 52.9% for BERT
against an actual 50.4% (−2.5). The stronger backbone turns extra out-of-distribution data
into a gain; the weaker one is hurt by it.

### 4.5 Which question types fail

Table 2 and the right panel of Figure 2 break accuracy down by question category. The
clearest result is negation: questions containing "not" or "except" are the weakest category
on both subsets, and the only category weakest on both — 50.7% on Easy against a 75.0%
average (−24.4, *p* < 0.0001) and 43.0% on Challenge against 56.1% (−13.0, *p* = 0.0158).
The second pattern is answer-option length. On Easy, options of two words or fewer score
80.0%, falling steadily to 74.1%, 68.9% and 62.5% for 3–5, 6–10 and more than 10 words;
Challenge shows the same tail (49.1% above 10 words) plus a penalty at the short end
(50.6%). Easy questions longer than 30 words score 68.7%, 6.4 below average (*p* = 0.0015).
Predicted answer positions track the true distribution to within 1.8 points. Both patterns
match
previous work: insensitivity to negation is a known weakness of masked-language-model
encoders, and the length effect is what the separation-versus-options argument from
Section 2 predicts, since a long option is scored alone and never compared against its
competitors. We should not over-claim: our categories come from regular expressions over the
question text, so they are a rough lexical proxy, and the 128-token cap is a competing
explanation for the long-question result.

**Table 2.** Selected question categories for the combined DeBERTa champion on the test sets. Δ is the difference from that subset's overall accuracy and *p* is a two-proportion test against the rest of the subset. The full 28-row table is in Appendix A.

| Subset | Category | n | Accuracy % | Δ | *p* |
|:---|:---|---:|---:|---:|---:|
| Easy | negation / except | 73 | 50.7 | −24.4 | <0.0001 |
| Easy | options ≤ 2 words | 1060 | 80.0 | +5.0 | <0.0001 |
| Easy | options 6–10 words | 634 | 68.9 | −6.1 | <0.0001 |
| Easy | options > 10 words | 64 | 62.5 | −12.5 | 0.0188 |
| Easy | question > 30 words | 386 | 68.7 | −6.4 | 0.0015 |
| Challenge | negation / except | 79 | 43.0 | −13.0 | 0.0158 |
| Challenge | options ≤ 2 words | 320 | 50.6 | −5.4 | 0.0216 |
| Challenge | comparative | 38 | 73.7 | +17.6 | 0.0260 |
| Challenge | superlative | 427 | 60.0 | +3.9 | 0.0419 |

### 4.6 Confidence and running time

The left panel of Figure 2 compares confidence with accuracy. On Easy the model is close to
calibrated below 0.7 and only mildly overconfident at the top: the [0.9, 1.0) bin is 91.3%
accurate at 97.6% mean confidence, a gap of 6.3 points. On Challenge it is overconfident
everywhere and much more so — the same bin is 77.5% accurate at 96.7% confidence, a gap of
19.2 points, and the [0.7, 0.9) gap is 23.4 points against 12.6 on Easy. In total 87
Challenge predictions (7.4% of the subset, 16.9% of its errors) are wrong above 0.9
confidence, so the model is least able to tell when it is wrong on the subset where it is
wrong most often. On cost, BERT averaged 17.3 s/epoch and DeBERTa 20.8 — only 20% slower for
1.7× the parameters, since most extra parameters are embedding lookups; all 63 completed
runs used 103 GPU-minutes in total.

## 5. Discussion

**Overfitting.** Figure 3 shows validation accuracy and training loss for the six sweep
champions. In every case validation accuracy flattens after three or four epochs while the
training loss keeps falling toward zero. Training longer did not help, and the last-epoch
weights would have been worse than the best-epoch weights we used, so keeping the best
checkpoint mattered.

**The evaluation format.** Our Easy–Challenge gap on test is 18.9 points. Borchmann (2025),
discussed in Section 2, argues that a large part of a gap this size can come from
scoring each option in isolation instead of showing the model all options at once. Our
architecture does exactly the isolated kind of scoring, and our option-length results are
consistent with that explanation. The obvious next experiment is a second stage that sees
all the options in one context and reranks them. We did not have time to try it.

**A bug worth reporting.** We originally trained on a Mac using PyTorch's MPS backend.
The training loss went down normally but validation accuracy stayed at chance. We first
assumed our labels were misaligned. Running the identical code with the identical seed on
the CUDA GPU on DataHub reached 55.8% immediately, which showed the problem was the MPS
backend producing silently wrong numbers rather than a bug in our code. Without running
the same code on a second device we could not have told a numerical failure in an
accelerator backend apart from a modelling mistake. Every number in this paper comes from
CUDA.

## 6. Limitations

Each cell in our sweeps used a single random seed, so differences of about two points
within a sweep are probably noise. The BERT sweep on Easy completed only 3 of its 12
configurations before the server ran out of disk space. We have no test-set numbers for
BERT at all, so every comparison between the two backbones is on validation data only. All
three test numbers come from one checkpoint, the combined DeBERTa champion, while the
validation numbers come from three different per-subset champions, so a row of Table 1 is
not one model's profile across the three subsets. Finally, our error categories are found
with regular expressions and are only a proxy for the real question types.

## 7. Conclusion

A single 184M-parameter encoder with no retrieval reaches 75.0% on ARC-Easy and 56.1% on
ARC-Challenge, about 29 points above the best baseline from the original ARC paper on
Challenge but still below a retrieval-augmented ensemble. Which pretrained backbone we
start from matters far more than any hyperparameter we tuned, and the best hyperparameters
do not carry over from one backbone to the other. The clearest remaining weakness is
negation, and the model is most overconfident on the questions it is worst at.

## References

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O.
(2018). *Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge.*
arXiv:1803.05457.

Clark, P., Etzioni, O., Khot, T., Khashabi, D., Mishra, B., Richardson, K., et al. (2020).
*From 'F' to 'A' on the N.Y. Regents science exams: An overview of the Aristo project.*
AI Magazine, 41(4). arXiv:1909.01958.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep
bidirectional transformers for language understanding.* NAACL-HLT.

He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-style
pre-training with gradient-disentangled embedding sharing.* ICLR.

Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.

Borchmann, Ł. (2025). *ARC 'Challenge' is not that challenging.* Findings of the Association
for Computational Linguistics: ACL 2025, 2797–2804. Vienna, Austria: Association for
Computational Linguistics. doi:10.18653/v1/2025.findings-acl.144

## Figures

- **Figure 1** — `results/analysis/fig_hparam_heatmaps.png`. Validation accuracy over the
  learning rate × epochs grid, for each backbone and training regime. The champion cell is
  marked and failed runs are grey.
- **Figure 2** — `results/analysis/fig_error_analysis.png`. Left: accuracy against mean
  confidence in each confidence bin. Right: accuracy by question category relative to the
  subset average.
- **Figure 3** — `results/analysis/fig_training_dynamics.png`. Per-epoch validation
  accuracy and training loss for the six sweep champions.

## Appendices

- **Appendix A** — full question-category table (`results/analysis/t5_error_categories.md`).
- **Appendix B** — all 63 completed runs (`results/analysis/t2_all_runs.md`), throughput
  (`t3_efficiency.md`) and full calibration bins (`t4_calibration.md`).
