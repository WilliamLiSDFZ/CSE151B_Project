# Fine-Tuning BERT and DeBERTa-v3 on the AI2 Reasoning Challenge

**Yuze Li** — CSE 151B, Summer 2026, Model-Building Track

*Version 2 — expanded, no word limit. Related Work + Methods + Results: 9,348 words
(1,390 + 3,472 + 4,486), tables excluded; whole paper 13,096 words. The 1,992-word version is
retained separately as `PAPER_DRAFT_v1.md`. All numbers reflect the complete 72-run sweep.*

---

## Abstract

We fine-tune two pretrained encoders, BERT-base and DeBERTa-v3-base, on the AI2 Reasoning
Challenge (ARC), a dataset of grade-school science exam questions. We use no retrieval and
no external corpus, so the model must answer using only the knowledge it acquired during
pretraining plus whatever the fine-tuning set teaches it. We ran 72 training runs across a
grid of learning rates, epoch counts, and training regimes, holding the classification head,
the preprocessing pipeline, the optimiser, the schedule, and the evaluation protocol fixed
so that the pretrained checkpoint is the only variable that changes between the two model
families. Our selected model — the DeBERTa-v3-base champion trained on the combined
Easy + Challenge training set — reaches 75.0% accuracy on the ARC-Easy test set and 56.1%
on the ARC-Challenge test set, using 184M parameters and 110 GPU-minutes for the entire
study. On validation, DeBERTa-v3-base beats BERT-base by 18.1, 9.4, and 19.8 points on Easy,
Challenge, and combined respectively, all significant at the 0.05 level under a
two-proportion test. Three secondary findings follow. First, the optimal learning rate does
not transfer between the two backbones: BERT peaks at 5 × 10⁻⁵ on Easy and combined and at
3 × 10⁻⁵ on Challenge, while DeBERTa peaks at 3 × 10⁻⁵ or 2 × 10⁻⁵ — the top of a range that
is already two to three times lower in absolute terms. Second, the most damaging surface property we can measure is a negation
word such as "not" or "except", which costs up to 24.4 accuracy points; we stress throughout
that these categories are a lexical proxy, not a reasoning taxonomy. Third, the model is
badly calibrated exactly where it is weakest: on Challenge its most confident predictions
overstate accuracy by 19.2 points, against 6.3 points on Easy.

## 1. Introduction

ARC is a dataset of natural science questions drawn from grade-school standardised exams.
On the surface this sounds like a solved problem. The questions are short, the vocabulary is
ordinary, and a human fourth-grader is expected to answer them. It is not a solved problem.
The authors of the dataset deliberately partitioned it so that one half is adversarial to
shallow methods: they ran two simple solvers over every question, one based on information
retrieval and one based on pointwise mutual information between question and answer words,
and every question that *both* solvers got wrong was placed in the Challenge partition. The
remainder became Easy. When the dataset was released in 2018, none of the systems the
authors tested could beat random guessing on Challenge by more than about two points.

That construction is the reason ARC is a good testbed for a question that is otherwise hard
to isolate: how much of a question-answering system's competence comes from finding the
right text, and how much comes from what the model already knows? Most of the strong
published numbers on ARC come from systems that do one of three things — retrieve supporting
sentences from a science corpus, use a very large language model, or both. Comparatively
little published work reports what a *base-sized* encoder achieves on its own, closed-book,
on a compute budget a student can afford.

That is the question this project asks. We fine-tune two pretrained encoders with the same
multiple-choice head design, the same data pipeline, and the same training loop, changing
only the pretrained weights, and we run a hyperparameter sweep large enough to
separate "this backbone is better" from "this backbone happened to get a better learning
rate". Our contributions are:

1. **A controlled backbone comparison.** The head design, preprocessing, padding and masking
   scheme, optimiser, schedule, batch composition, precision policy, checkpoint selection
   rule, and evaluation code are shared between the two model families; only the pretrained
   checkpoint changes. Three things necessarily follow the checkpoint rather than our code —
   the tokeniser, one extra randomly initialised projection in DeBERTa's multiple-choice
   class, and a family-specific learning-rate range — and we state each explicitly in
   Section 3 rather than claiming a cleaner control than we have. This lets us attribute
   most of the observed gap to pretraining rather than to any downstream choice, and to argue
   that it is not simply a parameter-count effect.
2. **A complete 72-run hyperparameter study.** We show that the best fine-tuning settings for
   one backbone are not the best settings for the other, measure how much accuracy actually
   moves across each backbone's grid — on five of six model-by-subset grids, no more than the
   sampling noise on that validation split — and show that within a single sweep the champion is never
   significantly better than its runner-up, which is itself a result worth stating, because
   it bounds how much of any headline number is selection noise.
3. **An error and calibration analysis.** We break test accuracy down by lexical cue,
   question length, and answer-option length, and we relate confidence to accuracy bin by
   bin. The analysis shows both a specific linguistic weakness (negation) and a systematic
   reliability failure (the model is most overconfident on the subset where it is least
   accurate).

We also report, in Section 5, a hardware-level bug that cost us several days and that we
think is worth documenting: PyTorch's Apple MPS backend silently produced wrong results on
this workload, in a way that was indistinguishable from a label-alignment bug in our own
code until we ran the identical code on a second device.

## 2. Related Work

We organise prior work by the idea each family relies on, rather than chronologically, and
for each family we state what it got right and what it got wrong. All accuracies below are
quoted from the primary sources named in the References.

**The ARC dataset and its construction.** ARC (Clark et al., 2018) consists of 7,787
multiple-choice natural-science questions with three to five answer options each. What makes
it more than another exam benchmark is the partition rule described in Section 1: a question
is placed in Challenge if and only if both a retrieval solver and a word-co-occurrence
solver answer it incorrectly. Easy contains 2,251 / 570 / 2,376 and Challenge 1,119 / 299 /
1,172 train / validation / test questions; taken together, 3,370 / 869 / 3,548. The
consequence of the partition rule is easy to state and easy to forget: Challenge is not
merely "the harder questions", it is specifically the region where lexical overlap between
question and correct answer fails to identify the answer, *by construction*. A system that
improves its word matching should therefore gain on Easy and gain little or nothing on
Challenge. Test labels are public, which lets us report test accuracy directly rather than
through a leaderboard.

**Retrieval plus sentence-level entailment.** The baselines reported in the original ARC
paper pair a retrieval component with some form of sentence-pair inference or structured
reasoning. On the Challenge test set they span 20.26% (IR over the ARC corpus) to 27.11%
(DGEM), against a random-guess rate of 25.02% — so the *best* of them beats chance by about
two points, and several fall below it. On Easy the same systems reach roughly 36% to 63%, the
strongest being IR over the ARC corpus at 62.55%. Two further rows, IR-Waterloo (74.48%) and
PMI-Waterloo (77.82%), score higher on Easy, but they are close relatives of the two solvers
used to define the partition in the first place, so their Easy accuracy is inflated by
construction and is not a meaningful baseline — which is also why both drop to 1.02% and
2.03% on Challenge. What this family got right is that retrieval alone answers a large majority of Easy
questions, and that around 62% is close to the ceiling of what corpus word-matching can do
on that partition. What it got wrong is the assumption that the missing ingredient was a
better sentence-pair scorer: stacking DecompAttn, DGEM, TableILP, or TupleInference on top
of the retrieved sentences moved Challenge accuracy by only a few points in either
direction. The missing ingredient was background knowledge, not inference machinery.

**Retrieval-augmented pretrained transformers.** Aristo (Clark et al., 2020) is the system
that closed most of the gap. It combines retrieval with an ensemble of solvers, including
an AristoRoBERTa reader built on RoBERTa-large, and reaches 86.99% on Easy and 64.33% on
Challenge; its individual transformer solvers score 81.78 / 57.59 (AristoBERT) and 82.88 /
64.59 (AristoRoBERTa). That is roughly 37 points of Challenge accuracy gained in two years,
and almost none of it came from better symbolic inference — the retrieval machinery already
existed in 2018. Two ablations in that paper are directly relevant to our own error
analysis. First, when AristoRoBERTa is retrained on *only the answer options*, with no
question body and no retrieved knowledge, it still reaches 36.17% on Easy and 35.92% on
Challenge — well above the 25% chance rate, which means a non-trivial slice of ARC is
answerable from surface properties of the option set alone. Second, when the options are
made adversarially harder by expanding each question to eight options, accuracy falls to
65.7 / 47.7. Both results say that the option set, not just the question, carries signal.
The UnifiedQA line (Khashabi et al., 2020) makes the complementary point about scale and
multi-task pretraining: at 11B parameters, a T5 model fine-tuned on ARC reaches 83.8 / 65.4
closed-book and 90.0 / 69.7 with retrieved text, while the same model first pretrained on a
mixture of QA formats (UnifiedQA) reaches 86.4 / 75.0 closed-book and 92.0 / 78.5 with
retrieval — a gain of 9.6 points on Challenge from format-agnostic QA pretraining alone.
This family got right that pretraining scale, not retrieval, was the binding constraint. What
it leaves open is the cost question: every number above comes from a model between 355M and
11B parameters, usually with a retrieval stack attached.

**Retrieval-free fine-tuned encoders.** This is the family our own system belongs to, and
the one with the fewest published ARC numbers. Huang et al. (2022) report a clean set of
closed-book baselines — their experiments use no provided documents and no external corpora
— fine-tuning each backbone with a standard multiple-choice head and evaluating on the ARC
test sets. On Easy / Challenge test they obtain 52.32 / 34.85 for RoBERTa-base, 45.84 /
30.21 for ALBERT-base, 62.40 / 35.97 for RoBERTa-large, and 53.77 / 31.19 for ALBERT-large;
their own generate-then-select method, GenMC, reaches 58.82 / 39.00 with a T5-base backbone
and 69.01 / 47.41 with T5-large. What this family gets right is that it isolates the
variable we care about: with retrieval removed, differences between systems are differences
in what the pretrained weights already encode. What it gets wrong, or rather leaves
incomplete, is coverage of the encoder space — the strongest closed-book encoder reported
there is RoBERTa-large, and we are not aware of a published ARC result for DeBERTa-v3 of any
size, closed-book or otherwise. Note also that these systems are trained per-subset, whereas
our selected model is trained on the combined set; we return to this confound in Section 4.2.

**Very large models scored without fine-tuning.** Brown et al. (2020) report GPT-3 175B at
68.8 / 51.4 zero-shot, 71.2 / 53.2 one-shot, and 70.1 / 51.5 few-shot on ARC-Easy /
ARC-Challenge, against a fine-tuned state of the art they quote as 92.0 / 78.5. The
interesting feature of that row is how little the in-context examples help: one-shot gains
1.8 points on Challenge over zero-shot, and few-shot gains nothing over one-shot. This family
got right that a great deal of science knowledge is recoverable from pretraining alone,
without any task-specific gradient updates. What it got wrong — or at least what later work
suggests it under-examined — is the assumption that the resulting number measures reasoning
ability rather than scoring protocol, which brings us to the last family.

**How the options are scored, and our gap.** Borchmann (2025) argues that a substantial part
of the apparent Easy–Challenge gap is an artefact of the evaluation setup rather than a
property of the questions. Scoring each option in a separate forward pass — so that the
model never sees the alternatives while judging one of them — performs far worse than
presenting all options together in a single context and letting the model choose among them.
Re-measuring the same checkpoints under both protocols, Qwen 14B moves from 47.3% to 86.6%,
Llama 2 70B from 57.4% to 79.6%, and Llama 3.1 70B from 64% to 93%. These are large
generative models scored without fine-tuning, so we borrow the *mechanism* from this work
and explicitly not the magnitude. The mechanism matters to us because the standard
`AutoModelForMultipleChoice` head reshapes a batch of shape (batch, options, tokens) into
(batch × options, tokens) before the encoder runs, so attention cannot cross option
boundaries: each option is encoded in isolation, exactly like the weaker of Borchmann's two
protocols. The analogy is ours, not a claim Borchmann makes about encoder-based
multiple-choice heads. It is also incomplete in an important way, and we say so: unlike a
zero-shot generative model, our final softmax and cross-entropy *are* computed jointly
across the option set, so fine-tuning does supply a comparative training signal even though
the representations are built independently.

**The gap we address.** Prior work characterises retrieval-augmented systems well, and
characterises very large models scored zero-shot well. It characterises small closed-book
encoders thinly, and the strongest such published baseline we could find uses RoBERTa-large.
No published work we are aware of asks how far a base-sized closed-book encoder gets on ARC
under a fixed small compute budget, how much of that result is decided by the choice of
pretrained backbone as opposed to the fine-tuning hyperparameters, or which surface
properties of a question predict failure for such a model. Those are the three questions
Sections 3 and 4 answer.

## 3. Methods

### 3.1 Multiple choice as a learning problem

A question with *n* options is turned into *n* independent input sequences of the form
`[CLS] question [SEP] option_i [SEP]`. All *n* sequences pass through the same encoder — the
weights are shared, not merely tied — and the pooled `[CLS]` representation of each sequence
is projected by a single linear layer down to one scalar. The *n* scalars are then softmaxed
*across the option axis* and trained with cross-entropy against the index of the correct
option.

Two properties of this formulation matter for the rest of the paper. First, the model
predicts a *position within this question's option list*, not a fixed class label. That is
what allows one head to serve questions with three, four, or five options without any
per-question architectural change, and it is also why the answer-key normalisation described
in Section 3.2 is necessary. Second, the head is very small relative to the encoder, and it
is the only randomly initialised part of the model — everything else arrives from
pretraining. This asymmetry is the reason the learning-rate warmup discussed in Section 3.4
is not optional: for the first few hundred updates, a randomly initialised head is producing
meaningless gradients that flow back into weights we would rather not damage.

One asymmetry between the two backbones belongs here rather than in a footnote, because it
qualifies the "identical head" control we claim elsewhere. We instantiate both models through
`AutoModelForMultipleChoice`, which resolves to `BertForMultipleChoice` for BERT and to
`DebertaV2ForMultipleChoice` for DeBERTa-v3. For BERT the new parameters are exactly the
769 of the final linear layer (768 weights plus a bias), because BERT's pooler is part of the
pretrained checkpoint. DeBERTa-v3 ships no pooler, so its multiple-choice class instantiates
a `ContextPooler` containing a randomly initialised 768 × 768 linear layer on top of the
encoder, adding about 0.59M new parameters before the same 769-parameter classifier. The two
heads are therefore functionally the same design — pool the `[CLS]` position, project to one
scalar, softmax across options — but DeBERTa's has one more trainable layer, roughly 0.3% of
its total parameters, initialised from scratch. We did not control for this, and it is a
small confound in the backbone comparison of Section 4.3; we would not expect a single extra
768 × 768 projection to account for a 9- to 20-point gap, but we cannot rule out that it
contributes, and it is one more reason the two families' optimal learning rates might differ.

It is worth being precise about what is and is not shared between options, because
Section 2's last paragraph turns on it. Inside the encoder, nothing is shared: option *i*'s
tokens cannot attend to option *j*'s tokens, because the two live in different rows of the
batch. The comparison happens only at the very end, in the softmax and the loss. So the
representation of each option is built in ignorance of its competitors, while the *decision*
is made with full knowledge of them. A long, syntactically complex distractor is therefore
encoded exactly as it would be if it were the only option on the page.

### 3.2 Data and preprocessing

We use the `allenai/ai2_arc` dataset with the `ARC-Easy` and `ARC-Challenge` configurations
and the official splits, unmodified: 2,251 / 570 / 2,376 for Easy, 1,119 / 299 / 1,172 for
Challenge, and 3,370 / 869 / 3,548 for the combined regime, which is the concatenation of
the two. We do not resample, rebalance, filter, or augment.

Three properties of the raw data need normalising before training.

*Mixed answer-key schemes.* Some questions label their options `A`–`E` and others `1`–`5`.
The gold answer is given as a key in whichever scheme that question uses. We map both schemes
onto a zero-based integer index into the option list, so that the label the loss sees is
always a position. Getting this wrong is silent: the model still trains, the loss still
falls, and accuracy simply sits near chance.

*Variable option counts.* Most ARC questions have four options, but not all. In the test
sets, Easy contains 2,365 four-option, 7 three-option, and 4 five-option questions, and
Challenge contains 1,165 four-option, 4 three-option, and 3 five-option questions; the
validation sets follow the same pattern (Easy 567 / 1 / 2, Challenge 295 / 3 / 1 for four /
three / five options). More than 99% of questions are therefore four-way, which is why the
analytic random baseline lands at 25.0% rather than something further from it. Because the
counts vary, each batch is padded to the largest option count present in that batch and
carries a boolean choice mask. Before the softmax, the logits of padded slots are set to
`torch.finfo(logits.dtype).min` so they receive effectively zero probability under any
dtype, including fp16.

*Padded slots must contain real text.* This is the one implementation detail we would flag
to anyone reproducing this setup. A padded option slot cannot be left as an all-padding
sequence, because every position in such a row is masked out of the attention computation;
the attention softmax then normalises over an all `-inf` row and returns NaN, which
propagates through the whole batch and destroys the update. We therefore fill padded slots by
copying an existing real option from the same question, and rely on the logit mask to remove
the copy's contribution to the loss.

*Tokenisation.* We tokenise each (question, option) pair with the backbone's own tokeniser
using `truncation="only_first"`, so that when a pair exceeds the length budget the question
is truncated from the end and the option is never touched. Truncating the option instead
would be far more damaging, since the option is what the model is scoring; a truncated
question loses context, but a truncated option can lose the very word that makes it right or
wrong. Padding is dynamic — `padding="longest"` within each batch rather than to a global
maximum — which saves a substantial amount of computation, since ARC questions are short and
padding every batch out to 128 tokens would waste most of the encoder's work. What the
reshape from (batch × options, tokens) back to (batch, options) needs is only that all
sequences in a single collated batch have the same length; padding to a fixed global length
would satisfy that too. The maximum length is 128 tokens.

We did not log the exact fraction of examples truncated at 128, and the exploratory notebook
was committed without stored cell outputs, so we cannot recover it; we list this in
Section 6 as a measurement we are missing. What we can say from the data we do have is that
truncation is unlikely to be common. After the special tokens and the answer option are
accounted for, roughly 100 tokens remain for the question, which at typical WordPiece
expansion rates corresponds to somewhere in the region of 75–80 English words. In the test
sets, questions longer than 30 words make up 386 of 2,376 Easy questions (16.2%) and 266 of
1,172 Challenge questions (22.7%), and the >30-word band is the longest band we measured, so
we cannot bound the tail directly. We therefore treat the 128-token cap as a *possible*
competing explanation for the long-question result in Section 4.5, rather than dismissing it.

### 3.3 Models

`bert-base-uncased` (Devlin et al., 2019) and `microsoft/deberta-v3-base` (He et al., 2023)
are deliberately matched in the one dimension that is easiest to confound and separated in
almost every other. Both are 12-layer encoders with a hidden size of 768 and 12 attention
heads; both are the "base"-scale general-purpose checkpoint of their family, meant to be
fine-tuned rather than prompted; both consume a token sequence and expose a per-token hidden
state that our head reads at position zero. Layer for layer they are the same object. What
differs is where their parameters sit, how they encode position, how they were pretrained,
and how they segment text — and because DeBERTa-v3 changes all four at once relative to BERT,
these are worth separating before Section 4.3 reports what the change is worth.

*Where the parameters sit.* BERT-base has 110M parameters against DeBERTa-v3-base's 184M, a
ratio of 1.7 that would ordinarily make a comparison unfair. It is much less unfair than it
looks, because the two models put their parameters in different places. BERT's 30,522-token
WordPiece vocabulary occupies about 23.8M parameters (30,522 × 768 for the word embeddings,
plus 512 learned absolute position vectors, the two-entry token-type table, and a LayerNorm);
DeBERTa-v3's 128,100-token SentencePiece vocabulary occupies about 98.4M (128,100 × 768).
Subtract the embedding tables from both and the transformer stacks are within a couple of
percent of each other — roughly 85M against 86M. Embeddings are lookups rather than matrix
multiplications, so in the forward pass those 74M extra parameters cost memory and disk but
almost no arithmetic per token. Training is less forgiving than that sentence alone suggests:
the backward pass through an embedding layer materialises a dense gradient the size of the
entire table, and AdamW touches every parameter and both of its moment buffers on every step,
so a four-times-larger vocabulary does cost real time and memory once gradients are involved.
Even so, the two models are much closer in computation than in parameter count — Section 4.6
puts DeBERTa between 48% and 79% above BERT per training example, a range that straddles the 67% the
parameter counts would imply. This is part of the basis for our claim in Section 4.3 that the
backbone gap is not a size effect; the parameter accounting above is the rest of it.

*How position is encoded.* BERT adds a learned absolute position embedding to the token
embedding at the input, so from the first layer onward a single vector carries both "which
word" and "which slot", and every attention score is computed from vectors in which the two
are already mixed. DeBERTa keeps them apart. Each token is represented by a content vector
and a relative-position vector, and the attention score between two tokens is a sum of
separate content-to-content, content-to-position and position-to-content terms, computed over
the *relative* offset between the pair rather than their absolute indices. The practical
consequence is that the model can learn that a modifier attaches to the noun three tokens to
its left without having to learn that fact separately for every absolute position at which
the pattern occurs. Whether this buys anything on ARC specifically we cannot say from our
design: our inputs are short — one question and one option, under 128 word pieces — and short
inputs are where absolute position embeddings are least strained.

*How they were pretrained.* BERT was trained with masked language modelling plus
next-sentence prediction on BooksCorpus (800M words) and English Wikipedia (2,500M words),
about 16GB of uncompressed text by Liu et al.'s (2019) accounting. Two of those three
choices were later revised by the field: Liu et al. found next-sentence prediction to be of
little or negative value, and 16GB became small. DeBERTa-v3 is trained instead with
ELECTRA-style replaced-token detection — a small generator model proposes substitute tokens
and the main model classifies *every* position as original or replaced — on roughly 160GB.
The difference in signal density is the part worth noting: masked language modelling produces
a learning signal at the 15% of positions that were masked, while replaced-token detection
produces one at 100% of them, so a token of text does more work. DeBERTa-v3's specific
contribution over plain ELECTRA is gradient-disentangled embedding sharing, which stops the
generator's and the discriminator's gradients from pulling the shared embedding table in
opposite directions. Between the objective and the tenfold data increase, DeBERTa-v3 has seen
far more supervision per parameter than BERT before either of them sees a single ARC question.

*How they segment text.* The tokeniser is not a free choice — it ships with the checkpoint —
but it is a real difference between the models rather than an artefact of our setup, and it
plausibly matters on this dataset. ARC is grade-school science, so its vocabulary is dense in
terms like *photosynthesis*, *condensation*, *precipitation* and *thermometer*. A 128k
SentencePiece vocabulary is likelier to hold such a word, or a long piece of it, as a single
unit, where a 30k lowercased WordPiece vocabulary is likelier to shatter it into three or four
fragments whose embeddings must be composed by the encoder. We did not measure fertility
(pieces per word) on ARC and so cannot quantify this; we flag it as a plausible contributor
to the gap in Section 4.3 rather than a demonstrated one.

These four differences arrive as a bundle, and our experiment measures only their sum. When
Section 4.3 reports that swapping BERT for DeBERTa-v3 is worth 9.4 to 19.8 validation points,
that number is the joint effect of disentangled attention, a better pretraining objective, ten
times the pretraining data, and a four-times-larger vocabulary — we cannot apportion it among
them. Doing so would need at minimum a DeBERTa-v1 run to hold the architecture fixed while
reverting the objective, and a BERT-large run to separate scale from method; both were outside
our compute budget. We note this because the honest claim ("the checkpoint you start from
dominates the hyperparameters you tune") is weaker and more useful than the claim our data
cannot support ("disentangled attention is what makes the difference").

Against those intrinsic differences, our own setup adds as little as we could manage.
Preprocessing, collation, masking, optimiser, schedule, precision policy, checkpoint
selection, and evaluation run through the same code path for both models: the only thing our
code changes is the string passed to `from_pretrained`. Three consequences of that string
are nevertheless not identical, and we name them rather than claim a cleaner control than we
have. The tokeniser is tied to the checkpoint, as just discussed. The multiple-choice class
resolved by `AutoModelForMultipleChoice` differs, so DeBERTa gets one extra randomly
initialised projection (Section 3.1) — the class is named `DebertaV2ForMultipleChoice`, but
"V2" here identifies the shared architecture implementation, not the weights, since DeBERTa-v3
reuses v2's computation graph and changes only the pretraining recipe and vocabulary. And the
learning-rate grid is deliberately family-specific (Section 3.5), because the two families'
usable ranges do not coincide. Everything else is shared.

### 3.4 Training procedure

We optimise with AdamW. Weight decay is 0.01 and is decoupled from the gradient, following
Loshchilov and Hutter (2019): the decay term is applied directly to the parameters rather
than folded into the gradient, so that it does not get rescaled by Adam's per-parameter
second-moment normalisation. Biases and LayerNorm gains are excluded from decay, since
shrinking them toward zero has no regularising interpretation and does measurably distort
the normalisation statistics.

The learning rate warms up linearly over the first 10% of total optimiser updates and then
decays linearly to zero over the remainder. Warmup matters here for the reason given in
Section 3.1: the classification head is randomly initialised while the encoder is not, so the
first updates are computed from a head that is producing noise. Taking those updates at full
learning rate pushes large, uninformative gradients into a pretrained encoder before the head
has learned anything worth propagating. Because the schedule is defined over total updates
and total updates depend on the epoch count, the two swept hyperparameters — learning rate
and epochs — are not fully independent: a 3-epoch run and a 5-epoch run at the same peak
learning rate follow differently-shaped decay curves, not merely truncated versions of the
same curve.

The batch size is 8 with gradient accumulation over 2 steps, for an effective batch of 16.
The per-step batch is small because each *question* expands into up to five sequences, so a
nominal batch of 8 questions is up to 40 sequences of 128 tokens through the encoder. Using
accumulation rather than simply raising the batch size keeps peak activation memory at the
8-question level while giving the optimiser the gradient statistics of a 16-question batch.

On CUDA we train under fp16 autocast with a gradient scaler and fp32 master weights. Two
details are worth recording. First, the DeBERTa-v3 checkpoint is distributed in fp16, and if
it is loaded and left in fp16 as the master copy, the parameter updates at these learning
rates are small enough to round away to nothing against the fp16 mantissa — the loss barely
moves. We therefore cast the loaded weights to fp32 explicitly and let autocast handle the
per-operation precision, so that the *accumulator* is always fp32 even though the matmuls run
in half precision. Second, gradient norms are clipped to a maximum of 1.0, and the clip is
applied *after* `scaler.unscale_` rather than before. The gradient scaler multiplies the loss
by a large factor to keep small gradients inside fp16's representable range; clipping before
unscaling would therefore be clipping the scaled gradients against a threshold meant for
unscaled ones, and the effective clip threshold would silently change every time the scaler
adjusted its scale factor.

We evaluate on the validation split after every epoch and retain the weights from the
best-scoring epoch, so the final-epoch weights are never the ones reported. Section 5
quantifies how much this mattered. Every number in this paper was produced on a single NVIDIA
GPU on UCSD DataHub. We also had access to an Apple MPS device and stopped using it, for the
reason described in Section 5.

### 3.5 Experimental design

*Baselines.* We compare against two reference points. The **random-guess** baseline is the
mean of 1/*n* over the evaluation set, which for ARC's near-uniform four-option structure
comes to 25.0%; we also confirm it empirically over 1,000 simulated random-guessing trials,
which reproduces the analytic value to within a standard deviation of 0.85 to 2.5 points
depending on split size (for example 24.97 ± 0.85 on Easy test against an analytic 25.02).
The **majority-position** baseline always predicts whichever answer position is most frequent
in the training split, which is index 1 (the second option) for both subsets, occurring in
26.1% of Easy and 26.5% of Challenge training questions. This baseline exists to test whether
the dataset leaks an exploitable positional prior; it reaches 24.6% on Easy test and 26.5% on
Challenge test, so it does not.

*The grid.* We sweep three axes. Learning rate uses the range conventionally recommended for
each family — 2, 3, 5, 7 × 10⁻⁵ for BERT and 1, 1.5, 2, 3 × 10⁻⁵ for DeBERTa — which means
the two grids are deliberately *not* the same set of values, because using BERT's range on
DeBERTa would test only the divergent tail of DeBERTa's usable range. Epochs are 3, 4, or 5.
The training regime is Easy only, Challenge only, or combined, always evaluated on the
matching validation split. That is 4 × 3 × 3 = 36 configurations per model and 72 in total,
all of which completed. An earlier attempt at the BERT / Easy cell was lost to a server disk
quota after 3 of its 12 configurations; that cell was re-run in full, and every number in this
paper comes from the complete grid.

*What is held fixed.* Across all 72 runs we fix the batch size (8), gradient accumulation
(2), effective batch (16), maximum sequence length (128), weight decay (0.01), warmup ratio
(0.1), scheduler shape (linear warmup, linear decay), gradient clipping (1.0), precision
policy (fp16 autocast with fp32 masters), optimiser (AdamW), checkpoint-selection rule (best
validation epoch), and the classification head. The random seed is 77 in all six sweeps, so
every cell in the study is seed-matched to every other. One caveat we can put a number on,
because the accident of the disk failure let us measure it: all three configurations that had
completed before the failure returned different accuracies when re-run at the same nominal
seed. The cleanest instance is the configuration that had won the truncated sweep, 7 × 10⁻⁵
for 5 epochs, which scored 58.25 the first time and 57.37 the second — a drift of 0.88 points
with nothing changed but the wall clock. Seeding fixes the data order, the shuffling and the
head initialisation, but it does not make the run bit-reproducible; non-deterministic GPU
kernel selection and fp16 reduction order are the likely causes, and on this split they move a
validation accuracy by roughly a point. That figure is worth carrying into every comparison
below, and Section 6 returns to it.

*Metric and inference.* The metric is accuracy, defined as exact agreement between the
argmax over masked option logits and the gold option index. We report 95% Wilson score
intervals rather than normal-approximation intervals, since the Wilson interval behaves
correctly for the small validation splits (n = 299 for Challenge) and near the boundaries.
Pairwise comparisons use a two-sided two-proportion *z*-test. Every design decision — model
selection, learning rate, epoch count, training regime — was made on validation data. The
test sets were touched exactly once, at the end, with a single checkpoint.

## 4. Results

### 4.1 Main results

The selected model is the DeBERTa-v3-base champion trained on the combined Easy + Challenge
training set, with a peak learning rate of 2 × 10⁻⁵ and a 5-epoch schedule, with the
best-validation epoch (epoch 3) retained. Evaluated once on test, it reaches **75.0% on Easy**
(n = 2,376), **56.1% on Challenge** (n = 1,172), and **68.8% overall** (n = 3,548).

Both baselines sit at chance on every split: random guessing is 25.0% everywhere, and the
majority-position baseline reaches 24.6% on Easy test and 26.5% on Challenge test. The
dataset therefore leaks no usable positional pattern, and the model's own predicted-position
distribution confirms this from the other direction — on Challenge test it predicts positions
0–3 with frequencies 23.9 / 27.4 / 24.7 / 24.1% against a gold distribution of 22.7 / 26.5 /
26.5 / 24.3%, a maximum deviation of 1.8 points, and on Easy the maximum deviation is 1.2
points.

On validation, the per-subset DeBERTa champions score 76.1 / 52.2 / 70.0 and the per-subset
BERT champions 58.1 / 42.8 / 50.2 (Easy / Challenge / combined). On Easy and combined,
DeBERTa's test accuracy is about one point below its validation accuracy (75.0 against 76.1,
68.8 against 70.0), which suggests the 12-configuration sweep did not badly overfit the
validation split — the selection penalty is smaller than the width of the validation
confidence interval.

One comparison in Table 1 is tempting and invalid, so we name it explicitly. Challenge test
(56.1) is higher than Challenge validation (52.2), but these are two different models: the
validation figure comes from the Challenge-only champion, trained on 1,119 questions, while
the test figure comes from the combined champion, trained on 3,370. The difference is a
training-set difference, not evidence about generalisation.

**Table 1.** Accuracy (%) with 95% Wilson intervals. Validation cells come from each subset's
own sweep champion — three different models. All three test cells come from the *single*
combined DeBERTa champion. A row of this table is therefore not one model's profile across
the three subsets, and the validation and test columns are not paired.

| Model | Easy val | Easy test | Chal. val | Chal. test | Comb. val | Comb. test |
|:---|:---|:---|:---|:---|:---|:---|
| Random guess | 25.0 [21.6, 28.7] | 25.0 [23.3, 26.8] | 25.1 [20.5, 30.3] | 25.0 [22.6, 27.6] | 25.0 [22.3, 28.0] | 25.0 [23.6, 26.5] |
| Majority position | 26.7 [23.2, 30.4] | 24.6 [22.9, 26.4] | 24.4 [19.9, 29.6] | 26.5 [24.1, 29.1] | 25.9 [23.1, 28.9] | 25.3 [23.9, 26.7] |
| BERT-base | 58.1 [54.0, 62.1] | — | 42.8 [37.3, 48.5] | — | 50.2 [46.9, 53.5] | — |
| DeBERTa-v3-base | **76.1** [72.5, 79.5] | **75.0** [73.3, 76.7] | **52.2** [46.5, 57.8] | **56.1** [53.2, 58.9] | **70.0** [66.8, 72.9] | **68.8** [67.2, 70.3] |

### 4.2 Comparison with previous work

Against the 2018 baselines, our 56.1% on Challenge is 29.0 points above the best of them
(DGEM, 27.11%) and 31.0 points above random, and our 75.0% on Easy is 12.5 points above the
best 2018 Easy result from a corpus-based retrieval system (IR over the ARC corpus, 62.55%).
It also edges past the two Waterloo solvers on Easy (74.48% and 77.82% — we are above the
first and below the second), but as noted in Section 2 those two are not meaningful Easy
baselines. The informative comparison is on Challenge, where the 2018
systems scored at or below chance precisely because word matching had been excluded by
construction, and where a closed-book encoder with no retrieval at all clears that bar by 29
points. Pretraining, not retrieval, was the missing ingredient.

Against the closed-book fine-tuned encoders of Huang et al. (2022) — the family our system
actually belongs to — our model is ahead of every entry in their table on both subsets. Their
strongest encoder baseline, RoBERTa-large, reaches 62.40 / 35.97, and their own GenMC with a
T5-large backbone reaches 69.01 / 47.41, against our 75.0 / 56.1. On Challenge in particular,
our margin over RoBERTa-large is 20.1 points and over GenMC (T5-large) is 8.6. Three
qualifications belong with that comparison and we would not report it without them. First,
their systems are trained per-subset while ours is trained on the combined set, which gives
our Challenge-test model 3,370 training questions instead of 1,119 — that is a genuine
confound, not a stylistic caveat, and Section 4.4 shows the combined regime is worth about
+2.1 points to DeBERTa on validation. Second, their numbers are means over multiple seeds
with standard deviations reported (up to ±2.2 on Challenge), while ours is a single seed.
Third, RoBERTa-large is a 355M-parameter model and ours is 184M, so the comparison is not
unflattering to us on that axis, but ALBERT and T5 differ from us in more ways than size.

Against retrieval-augmented and very large systems we remain behind, as expected. Aristo
reaches 86.99 / 64.33, putting us 12.0 points behind on Easy and 8.3 behind on Challenge;
UnifiedQA at 11B reaches 86.4 / 75.0 closed-book and 92.0 / 78.5 with retrieval. Neither is
like-for-like: Aristo is a retrieval-augmented ensemble built around a RoBERTa-large reader,
UnifiedQA is roughly sixty times our parameter count, and our entire 72-run study consumed
110 GPU-minutes. The comparison we find most striking is with GPT-3 175B, which scores 68.8 /
51.4 zero-shot and 71.2 / 53.2 one-shot: our 184M fine-tuned encoder is ahead of it on
Challenge by between 2.9 and 4.7 points. We do not read that as evidence that our model knows
more science than GPT-3. We read it, following Section 2, as evidence that task-specific
fine-tuning on 3,370 examples buys a great deal on a benchmark whose difficulty is partly a
matter of scoring protocol — our model is trained with a loss that is explicitly comparative
across the option set, which a zero-shot scored model is not.

### 4.3 BERT compared with DeBERTa

On validation, DeBERTa-v3-base beats BERT-base by 18.1 points on Easy (76.1 against 58.1,
*p* < 0.0001), 9.4 points on Challenge (52.2 against 42.8, *p* = 0.0219), and 19.8 points on
combined (70.0 against 50.2, *p* < 0.0001). All three are significant at the 0.05 level under
a two-proportion test, and the Easy and combined intervals in Table 1 do not come close to
overlapping. We have no BERT test numbers — not because the weights are gone, but because the
test evaluation was never run against them (Section 6) — so **every BERT-versus-DeBERTa
comparison in this paper is a validation comparison.** All three gaps are an order of
magnitude larger than the roughly one-point run-to-run variation measured in Section 3.4, so
none of them is at risk from non-determinism.

Because head design, preprocessing, optimiser, schedule, precision, checkpoint rule, and
sweep protocol were shared — with the three checkpoint-linked exceptions listed in
Section 3.3 — we attribute the bulk of the gap to the pretrained checkpoint. We do not think
it is primarily a size effect. The parameter ratio is 1.7×, but roughly 98M of DeBERTa's 184M
are vocabulary embeddings, leaving transformer stacks of about 86M against BERT's roughly
86M non-embedding parameters — that is, the two models have similar amounts of *computation*
and differ mainly in vocabulary, pretraining objective, position encoding, and pretraining
data volume (about 160GB against about 16GB). We rest this argument on the parameter
accounting alone and not on the clock. The timings in Section 4.6 do not adjudicate it either
way: DeBERTa costs between 48% and 79% more per training example depending on the subset, a
range that straddles the 67% a naive reading of the parameter counts would predict rather than
falling clearly short of it, and those measurements are in any case too noisy to carry a
claim. Nothing in the throughput data contradicts the reading above; it simply does not
support it.

The structure of the gap is itself informative. The Challenge gap (9.4) is roughly half the
Easy gap (18.1) and is the least statistically secure of the three. That is what Section 2
predicts. A stronger backbone brings better lexical and factual competence, and Challenge is
the partition constructed by removing the questions where lexical competence suffices. The
subset where a better encoder helps least is the subset defined by excluding what a better
encoder is best at.

### 4.4 Hyperparameter sensitivity and the training regime

Figure 1 shows best validation accuracy across the learning-rate × epochs grid for each
backbone and regime. The clearest finding is that the optimal learning rate does not transfer
between backbones. BERT peaks at 5 × 10⁻⁵ on Easy and combined and at 3 × 10⁻⁵ on Challenge;
DeBERTa peaks at 3 × 10⁻⁵ on Easy and Challenge and at 2 × 10⁻⁵ on combined. BERT wants the
middle of its recommended range while DeBERTa wants the top of a range that is already two to
three times lower in absolute terms; the two grids overlap in only two values, 2 × 10⁻⁵ and
3 × 10⁻⁵. We should be careful about how much this buys, since the paragraphs below show that
most of the movement across a grid is inside sampling noise. Two transfers we can actually
check are small: DeBERTa at 3 × 10⁻⁵, BERT's best Challenge setting, reaches 69.7 on combined
against its own best of 70.0, and BERT at 2 × 10⁻⁵, DeBERTa's best combined setting, reaches
47.5 against its own best of 50.2. Neither gap is significant on its own. The one place where
the learning rate demonstrably matters is BERT on Challenge, where moving inside BERT's own
grid from 3 × 10⁻⁵ to 7 × 10⁻⁵ costs 7.4 points; even moving to 5 × 10⁻⁵, which is exactly
where the same backbone peaks on the other two subsets, costs 3.0. The practical lesson is
therefore about ranges rather than about single values: the two families want different
neighbourhoods, and the cost of being in the wrong neighbourhood is visible on the hardest
subset.

A tempting second claim is that DeBERTa's grid is also flatter — that it is not merely better
but less sensitive to the learning rate, and therefore cheaper to tune. The raw numbers seem
to support it, and now that both grids are complete they support it in the same direction on
all three subsets. Taking the best result at each learning rate and measuring the spread from
the worst of those four to the best, DeBERTa moves 3.0 points on Easy, 3.0 on Challenge and
1.8 on combined, against BERT's 4.2, 7.4 and 2.6. DeBERTa's grid is the flatter one every
time, and on Challenge it is flatter by more than a factor of two. We report the spreads but
decline the inference, for three reasons.

The first is that the two backbones were not swept over the same grid. BERT was searched over
{2, 3, 5, 7} × 10⁻⁵ and DeBERTa over {1, 1.5, 2, 3} × 10⁻⁵, chosen separately because each
family's usable range is different, so the two spreads are not measurements of the same
quantity: each is one model's behaviour over its own range, not two models on a common axis.
This is the weakest of the three objections, because in *ratio* terms the grids are close —
BERT's spans 3.5× from bottom to top and DeBERTa's 3.0× — so we are at least comparing
movement over comparable multiplicative widths. It still means the comparison is a
convenience, not a controlled one.

The second is that part of the effect is arithmetic rather than behavioural. A spread measured
in accuracy points is not scale-free: the sampling standard deviation of an accuracy is
√(p(1−p)/n), which is largest at p = 0.5 and falls away on either side of it. So even if the
two backbones were equally indifferent to the learning rate — even if every cell in both grids
were a draw from one fixed accuracy with no learning-rate effect at all — the model sitting
further from 50% would produce the tighter set of four draws. On Easy that favours DeBERTa
substantially, since 76.1% is much further from the midpoint than 58.1%: its expected spread
is 14% smaller than BERT's before any behavioural difference enters, and on combined 8%
smaller. We flag this rather than lean on it, because it does not explain the case that looks
most impressive. On Challenge the accuracies are 52.2% and 42.8%, and it is *DeBERTa* that
sits nearer 50%, so the arithmetic runs the other way: DeBERTa's expected spread there is 1%
*larger* than BERT's, while its observed spread is less than half. Whatever is happening on
Challenge, this objection does not account for it. What it does establish is that a flatness
comparison stated in raw accuracy points is not independent of the accuracy gap it is supposed
to be controlling for, and cannot be read as a clean statement about tuning behaviour.

The third is decisive, and it is the one that disposes of Challenge as well. Almost none of
these spreads exceeds what sampling noise on a validation set of this size would produce on
its own. Four independent draws from a binomial have an expected range of about 2.06σ;
evaluating that at each model's
own accuracy level and each split's size gives noise floors of 4.3 / 5.9 / 3.5 points for BERT
on Easy, Challenge and combined, and 3.7 / 6.0 / 3.2 for DeBERTa. Dividing the observed spreads
through by those floors gives 0.99, 1.25 and 0.76 for BERT and 0.81, 0.51 and 0.58 for
DeBERTa. Five of the six sit at or below their own floor. The only spread that clears its
noise floor is BERT's 7.4 on Challenge, and it clears it by 1.5 points on the smallest
validation split we have. Once the numbers are put in units that account for the accuracy
level, the flatness comparison stops having anything to compare: on five of six grids we
cannot establish that the learning rate does anything at all, and a difference between two
quantities that are both indistinguishable from zero is not a finding.

Two caveats cut in opposite directions here. The first makes the floors too high: the
configurations share a validation set and a seed, so their errors are positively correlated
and the true floor is somewhat below the independent-draw figure. We use the independent
version because it is the same conservative assumption behind the two-proportion tests in the
next paragraph, and because even the conservative version leaves only one spread standing. The
second makes them too low: Section 3.4 showed that re-running the same configuration at the
same seed moves validation accuracy by roughly a point, so run-to-run non-determinism sits
underneath the binomial floor as a second, independent source of movement. Both combined
spreads (2.6 and 1.8) are only two or three times that figure.

One asymmetry belongs with this, and it has reversed since the BERT / Easy cell was re-run.
BERT's grid now brackets its optimum on both sides on all three subsets — 5 × 10⁻⁵ on Easy and
combined and 3 × 10⁻⁵ on Challenge are all interior points — so BERT's spreads measure the
width of a basin. DeBERTa's do not: its champion sits at 3 × 10⁻⁵, the top value searched, on
both Easy and Challenge, so on two of three subsets its spread measures one flank of a curve
whose peak may lie outside the grid. If anything this cuts against the flatness claim, since
the truncated grid is the one that looks flatter, and extending it upward could only widen
DeBERTa's spread or reveal a decline. We did not run those configurations.

What survives is narrower and, we think, more useful than the flatness claim: on five of the
six model-by-subset grids, moving the learning rate anywhere within the family's usable range
changes validation accuracy by no more than the noise on that split. The exception is BERT on
Challenge, where the choice does appear to matter. That is a statement about how much tuning
buys, not about which backbone is easier to tune.

We also want to state a negative result clearly, because it bounds how much of the headline
numbers is selection noise: **within every sweep, the champion is not significantly better
than its runner-up.** With both grids now complete, five of the six champion-to-runner-up gaps
are under a point: DeBERTa combined 69.97 against 69.74 (0.23), DeBERTa Challenge 52.17
against 51.84 (0.33), DeBERTa Easy 76.14 against 75.61 (0.53), BERT Easy 58.07 against 57.37
(0.70), and BERT combined 50.17 against 49.37 (0.81). Every one of those five is smaller than
the roughly one-point run-to-run variation we measured in Section 3.4, which means we could
not reproduce the ranking by rerunning the two configurations, let alone defend it
statistically; the two-proportion *p*-values run from 0.74 to 0.93. The one larger gap is BERT
on Challenge, 42.81 against 39.80 (3.01 points on n = 299), and it is not significant either
(*p* ≈ 0.45) — 3.0 points on n = 299 sits well inside the ±5.6-point half-width of that
split's confidence interval. Naming a champion is therefore a practical necessity for choosing
one checkpoint to evaluate, not a claim that the winning configuration is genuinely best. This
is worth stating plainly because it bounds the whole hyperparameter exercise: the sweep buys
us a defensible way to pick one model out of twelve, and almost nothing else.

Our second ablation is the training regime. Weighting each backbone's two per-subset
champions by validation set size (570 Easy : 299 Challenge) gives the combined-validation
accuracy we would expect from routing each question to a specialist. For DeBERTa that
predicts 67.9% against an actual combined-trained 70.0%, a gain of 2.1 points; for BERT it
predicts 52.8% against an actual 50.2%, a loss of 2.6 points. The two backbones respond in
opposite directions to the same extra data. The stronger backbone converts additional
out-of-distribution examples into a gain — 2,251 Easy questions help it on Challenge — while
the weaker one is hurt by them, which is consistent with BERT having less capacity to spare
for a broader distribution at the point where it is already struggling to fit Challenge at
all (see the loss curves in Section 5).

### 4.5 Which question types fail

Table 2 and the right panel of Figure 2 break test accuracy down by question category for
the combined DeBERTa champion. Before reading them, the caveat that applies to every number
in this subsection: our categories are assigned by regular expressions over the question
text, so they are a **lexical proxy** for question type, not a semantic taxonomy and not the
hand-labelled knowledge-and-reasoning taxonomy of Clark et al. (2018). A question is assigned
to the first matching cue in the fixed order negation → superlative → comparative →
which-question, falling through to "other", so a question containing both "not" and "most"
is counted as negation only. The negation pattern also matches "at least", which is not a
negation in the relevant sense. Answer-option length bands use the *mean* number of words per
option, and question-length bands use the raw question word count.

The clearest and most robust result is negation. Questions matching `not`, `except`, `least`,
or `never` are the weakest category on both subsets, and the only category that is weakest on
both: 50.7% on Easy against a 75.0% subset average (Δ = −24.4, n = 73, *p* < 0.0001) and
43.0% on Challenge against 56.1% (Δ = −13.0, n = 79, *p* = 0.0158). On Easy this is by far the
largest deviation of any category in the 26-row table, and it drops the model to within 26
points of chance on questions the rest of the subset finds straightforward.

The second pattern is answer-option length, and it is monotone on Easy. Questions whose
options average two words or fewer score 80.0% (n = 1,060), falling to 74.1% for 3–5 words,
68.9% for 6–10 words, and 62.5% for more than 10 words (n = 64) — a 17.5-point span, with the
top and bottom ends both significant (*p* < 0.0001 and *p* = 0.0188). Challenge shows the same
long-option tail (49.1% above 10 words) but adds a penalty at the short end (50.6% at ≤2
words, Δ = −5.4, *p* = 0.0216), which is what one would expect if short Challenge options are
short precisely because they are minimally-differing near-synonyms.

The third pattern is question length, and it is weaker. On Easy, questions longer than 30
words score 68.7% against 75.0% overall (Δ = −6.4, *p* = 0.0015), while the 11–20 word band is
slightly *above* average (77.2%, *p* = 0.036). On Challenge no question-length band is
significant. As noted in Section 3.2, the 128-token cap is a competing explanation for the
long-question effect on Easy that we cannot rule out.

Finally, two cue categories are *above* average on Challenge: comparative questions score
73.7% (Δ = +17.6, n = 38, *p* = 0.026) and superlative questions 60.0% (Δ = +3.9, n = 427,
*p* = 0.0419). The comparative result rests on 38 questions and we would not build an argument
on it. The superlative result is on a large enough sample to take seriously, and suggests
that questions asking for the *most* or *best* of something are, despite their apparent
requirement for comparison, among the easier Challenge questions for this model — plausibly
because a superlative frame narrows the plausible answer type.

Do these failures match previous work? Two of the three do. Insensitivity to negation is a
long-documented weakness of masked-language-model encoders, and it is exactly the kind of
failure that survives scale, so finding it in a base-sized model is unsurprising but worth
confirming. The option-length effect is what the separation-versus-options argument of
Section 2 would predict: a long option is encoded in isolation and never placed alongside its
competitors during representation-building, so the head must judge its plausibility in
absolute terms rather than relative ones, and the longer and more specific the option, the
more room there is for that absolute judgement to go wrong. It also rhymes with the
answer-only ablation in Clark et al. (2020), where a model given nothing but the options
still reaches 36.2 / 35.9 — the option set carries independent signal, and the length pattern
is one visible form of it. The question-length effect is the one we cannot separate from a
purely mechanical cause.

**Table 2.** Selected question categories for the combined DeBERTa champion on the test sets.
Δ is the difference from that subset's overall accuracy, and *p* is a two-sided
two-proportion test against the rest of that subset. Categories are a lexical proxy assigned
by regular expression. The full 26-row table is in Appendix A.

| Subset | Dimension | Category | n | Accuracy % | Δ | *p* |
|:---|:---|:---|---:|---:|---:|---:|
| Easy | lexical cue | negation / except | 73 | 50.7 | −24.4 | <0.0001 |
| Easy | option length | ≤ 2 words | 1060 | 80.0 | +5.0 | <0.0001 |
| Easy | option length | 6–10 words | 634 | 68.9 | −6.1 | <0.0001 |
| Easy | option length | > 10 words | 64 | 62.5 | −12.5 | 0.0188 |
| Easy | question length | > 30 words | 386 | 68.7 | −6.4 | 0.0015 |
| Challenge | lexical cue | negation / except | 79 | 43.0 | −13.0 | 0.0158 |
| Challenge | lexical cue | comparative | 38 | 73.7 | +17.6 | 0.0260 |
| Challenge | lexical cue | superlative | 427 | 60.0 | +3.9 | 0.0419 |
| Challenge | option length | ≤ 2 words | 320 | 50.6 | −5.4 | 0.0216 |

### 4.6 Confidence, calibration, and running time

The left panel of Figure 2 plots accuracy against mean confidence within each of five
confidence bins, where confidence is the maximum post-softmax probability over the masked
option logits. On Easy the model is close to calibrated across the lower bins and only mildly
overconfident at the top: the [0.5, 0.7) bin is 59.3% accurate at 59.0% mean confidence (a gap
of −0.3 points, i.e. very slightly *under*-confident), and the [0.9, 1.0) bin is 91.3%
accurate at 97.6% mean confidence, a gap of 6.3 points. On Challenge the model is
overconfident in every bin and much more so: [0.7, 0.9) is 57.6% accurate at 80.9% confidence
(a 23.4-point gap, against 12.6 on Easy) and [0.9, 1.0) is 77.5% accurate at 96.7% confidence
(19.2 points, against 6.3).

The mass of the distribution shifts too, and in the direction that makes the calibration
failure consequential. On Easy, 50.4% of all predictions fall in the top confidence bin and
are 91.3% accurate; on Challenge only 32.9% do, and they are 77.5% accurate. At the other
end, 11.7% of Easy predictions fall below 0.5 confidence (41.0% accurate within that region)
against 19.5% of Challenge predictions (34.6% accurate). Mean confidence on correct answers
is 85.7% on Easy and 79.0% on Challenge; on incorrect answers it is 65.3% and 63.9%
respectively — note that the confidence attached to *wrong* answers is nearly identical
across the two subsets, so the calibration gap comes almost entirely from the model being
less confident when it is right, not more confident when it is wrong.

The practical form of this is the confidently-wrong count. On Challenge test, 87 predictions
(7.4% of the subset, and 16.9% of all its errors) are wrong at above 0.9 confidence; on Easy,
104 predictions (4.4% of the subset, 17.5% of its errors). So on the subset where the model
is least accurate it is also least able to signal that it is about to be wrong — which is the
worst combination for any downstream use that would want to abstain or escalate.

On cost, the aggregate figures are 15.1 seconds per epoch for BERT against 20.8 for DeBERTa;
the 36 BERT runs used 45.4 GPU-minutes and the 36 DeBERTa runs 64.9, for **110.3 GPU-minutes**
across the entire study. Now that both models have run the same 36 configurations, the two
aggregate means finally average over the same workload mix, which removes the largest
objection to comparing them. We still report the per-epoch comparison with more caution than a
single ratio would carry, because it is not a controlled benchmark.

Normalising by training-set size, using the per-epoch train and eval times each run stores in
its JSON, DeBERTa costs 1.79× BERT per training example on Easy, 1.48× on Challenge and 1.52×
on combined — a range of 48% to 79%, straddling rather than undercutting the 1.67× the
parameter counts would suggest. We draw no conclusion from that agreement, because the second
problem is unfixed and disqualifying: the six sweeps ran on three different days on a shared
campus GPU, and the corresponding *evaluation*-time ratios come out at 1.35, 0.41 and 0.38,
which would have DeBERTa performing inference more than twice as fast as BERT on two subsets
out of three. That cannot be a property of the models. Worse, DeBERTa's own per-example
evaluation cost varies from 2.35 ms on combined to 8.15 ms on Easy — a 3.5× swing on the same
model doing the same operation — which is larger than any between-model difference we are
trying to measure. Some part of the between-sweep variation is therefore contention on a
shared device rather than anything we set, and it is large enough to swamp the signal.

What survives is only the qualitative claim: DeBERTa is more expensive per step than BERT.
The measured margin — 48% to 79% depending on the subset — brackets the 67% its 1.7×
parameter ratio would suggest, so on this data we cannot say whether it costs more or less
than parameter counting predicts, and we do not use the timings to argue either way in
Section 4.3. Two mechanisms would account for a cost above the parameter ratio if one exists. Its disentangled
attention computes three score terms per layer — content-to-content, content-to-position and
position-to-content — instead of one, together with a projection of the relative-position
table and a gather at every layer; and in the version of `transformers` we use, `BertModel`
declares support for PyTorch's fused scaled-dot-product attention while the DeBERTa-v2
implementation does not, so BERT's attention dispatches to a fused kernel and DeBERTa's runs
as ordinary eager operations that materialise the full attention matrix. The larger embedding
table contributes as well, not through the forward pass but through the optimiser: the
backward pass materialises a dense gradient the size of the whole table, and AdamW updates
184M parameters and their two moment buffers every step against BERT's 110M. The full per-run
leaderboard is in Appendix B.

## 5. Discussion

**Overfitting and why best-epoch checkpointing mattered.** Figure 3 shows per-epoch
validation accuracy and training loss for all six sweep champions. The pattern is the same in
every case: validation accuracy rises steeply for two to three epochs, flattens, and then
drifts down, while training loss continues falling. Every run starts at a training loss of
about 1.39, which is ln 4 to three decimals — confirming that the randomly initialised head
begins genuinely uninformative on a near-uniformly four-option dataset — and three of the six
champions end below 0.14, with BERT on Easy reaching 0.049 and DeBERTa on combined 0.131. The
other three are instructive. BERT on Challenge only reaches 0.848 and DeBERTa on Challenge
0.545, so on the adversarial partition even the *training* set is not fit, which is a different
failure from the overfitting seen elsewhere. BERT on combined lands between the two regimes at
0.311, which is what one would expect from the weaker backbone on a training set that is a
third Challenge questions: it fits the Easy portion and drags the Challenge portion behind it.

Concretely, four of the six champions peak *before* their final epoch. Had we taken
final-epoch weights instead of best-epoch weights, we would have lost 1.8 points on BERT/Easy,
4.0 on BERT/Challenge, 1.0 on DeBERTa/Challenge and 1.2 on DeBERTa/combined, with BERT/combined
and DeBERTa/Easy peaking exactly at their last epoch and losing nothing — a mean of 1.3 points
across all six, and enough on its own to change which configuration won a sweep. Training longer mostly did not help: in the sweeps, 5-epoch runs
beat 3-epoch runs mainly by giving the best epoch more chances to occur, not by being better
at epoch 5. The two champions that peak at their last epoch are the mild counter-example, and
they are also the reason we cannot rule out that a 6- or 7-epoch run would have done better on
those two cells; the grid stops at 5 and we did not test past it.

**The evaluation format.** Our Easy–Challenge gap on test is 19.0 points. Borchmann (2025),
discussed in Section 2, argues that a large part of a gap of this magnitude can be an artefact
of scoring each option in isolation rather than presenting the full option set at once. Our
architecture does exactly the isolated kind of representation-building, and our option-length
results are consistent with that reading: the longer and more content-bearing an option, the
more the head has to judge it on absolute plausibility rather than by comparison, and the
worse it does. We want to be careful about how far we push this. Borchmann's measurements are
on large generative models scored without fine-tuning, and our setup differs in a way that
should *reduce* the effect: our cross-entropy is computed jointly over the option set, so
gradient descent does deliver a comparative signal even though attention never crosses option
boundaries. The honest summary is that we have a mechanism-level analogy and a consistent
piece of indirect evidence, not a measurement.

The obvious next experiment, which we did not have time to run, is to make the comparison
direct: keep this model as a first stage, then add a second stage that receives the question
and *all* options in a single sequence and reranks them, and measure how much of the 19.0-point
gap closes. A cheaper partial version would be to re-run our own evaluation with the options
concatenated into one context and a per-option marker, which requires no retraining of the
first stage and would isolate the representation-building effect from the loss.

**A bug worth reporting.** We originally trained on a Mac using PyTorch's MPS backend. The
training loss fell normally, epoch after epoch, and validation accuracy stayed pinned at
chance. Every symptom pointed at our own code — a label misalignment between the answer-key
normalisation of Section 3.2 and the option ordering is exactly what that looks like — and we
spent a long time auditing the collator and the label mapping. What resolved it was running
the identical code with the identical seed and configuration on the CUDA GPU on DataHub,
where it immediately reached 55.8% on ARC-Easy after three epochs. The code was correct; the
MPS backend was silently producing wrong numerical results on this workload. We record this
because the general lesson generalises past our project: a numerical failure inside an
accelerator backend and a modelling bug in your own code produce *the same observable
signature* — loss decreasing, metric flat — and no amount of reading your own code
distinguishes them. Running the same code on a second device is the cheapest available
discriminator, and it should be the first thing tried rather than the last. Every number in
this paper comes from CUDA.

## 6. Limitations

*Single seed, and not even a reproducible one.* Every cell in every sweep used one random
seed. All six sweeps now use seed 77, so the study is at least internally seed-matched, which
an earlier version of this work could not claim. But the re-run of the BERT/Easy cell gave us
an accidental measurement of what a fixed seed actually guarantees, and the answer is: less
than we assumed. All three configurations that had completed before the disk failure moved
when run again at the same nominal seed, in both directions; the one we can pin down exactly
is the truncated sweep's champion at 7 × 10⁻⁵ for 5 epochs, which fell from 58.25 to 57.37.
Fixing the seed fixes the shuffling and the initialisation, but fp16 accumulation order and
non-deterministic CUDA kernel selection leave roughly a point of slack on a 570-example
split. Every within-sweep
difference in this paper smaller than about a point should therefore be read as unmeasured
rather than small, and Section 4.4 shows that five of six champion-to-runner-up gaps fall in
exactly that band. A three-seed configuration exists in the repository (`configs/final.json`)
and has not been run; running it is the single cheapest improvement available to this study,
at roughly six GPU-minutes.

*Timings are not a controlled benchmark.* The seconds-per-epoch figures come from six sweeps
run on three different days on a shared campus GPU, with no attempt to pin the device, fix the
clock, or repeat a configuration under matched load. Within a sweep the numbers are tight —
mean and median per-example costs agree to three decimal places — but across sweeps they are
not comparable: the evaluation-time ratios between the two models come out at 1.35, 0.41 and
0.38, which would make DeBERTa faster than BERT at inference on two of three subsets, and
that is not possible. A cleaner symptom of the same problem is that DeBERTa's own per-example
evaluation cost varies by 3.5× across the three sweeps while the model and the operation are
identical. We therefore treat the throughput comparison in Section 4.6 as indicative of a
direction and not as a measurement, and we do not use it to support any quantitative claim —
in particular we do not use the fact that the training-time ratios now land near the parameter
ratio as evidence for anything, since a measurement this unstable would have agreed with
whatever we expected.

*The two learning-rate grids are not the same grid.* BERT was swept over {2, 3, 5, 7} × 10⁻⁵
and DeBERTa over {1, 1.5, 2, 3} × 10⁻⁵, overlapping in only two values. The grids were chosen
separately, before the sweeps, from each family's commonly reported range, which is the right
thing to do if the goal is to give each backbone its best shot but the wrong thing to do if
the goal is to compare their sensitivity to the learning rate. DeBERTa's grid is additionally
truncated at the high end: its champion sits at 3 × 10⁻⁵, the top value searched, on both Easy
and Challenge, so on those two subsets its optimum is not bracketed on both sides and its
spread measures one flank of a curve rather than the width of a basin. BERT's grid does
bracket its optimum on all three subsets. Section 4.4 reports the resulting spreads and
explains why we do not read a flatness comparison out of them; a clean version of that
comparison would need both models swept over a common grid wide enough to bracket both peaks,
which we did not run.

*No BERT test numbers.* This is a gap in the experiment, not in the artefacts. Every sweep ran
with `keep_champion_last_pt: true`, so each cell's champion weights were retained and BERT's
three champion checkpoints should still exist on the training server; what was never run is
the evaluation pass against the test split. Producing the missing row of Table 1 is therefore a
matter of minutes of inference rather than of retraining, and we simply did not do it before
the deadline. Until it is done, every BERT-versus-DeBERTa statement here is validation-only,
and the reader should note that this is the single largest hole in the paper: the headline
backbone comparison rests entirely on 869 validation questions that were also used to select
both models.

*One checkpoint behind three test cells.* All three test numbers in Table 1 come from the
same combined DeBERTa champion, while the three validation numbers come from three different
per-subset champions. A row of Table 1 is therefore not a single model's profile across
subsets, and the val/test pairs in that row are not paired measurements of the same system.

*Error categories are a lexical proxy.* The categories in Section 4.5 come from regular
expressions over question text, are assigned first-match in a fixed order, and include at
least one known false positive ("at least" matching the negation pattern). They describe
surface form. They do not identify reasoning type, and we make no claim about which
*reasoning* operations the model fails at.

*Unmeasured truncation rate.* We did not log the fraction of examples truncated at 128
tokens, and the exploratory notebook was committed without outputs, so the 128-token cap
remains an unquantified competing explanation for the long-question result in Section 4.5.

*Scope.* We compare two backbones at base size, on one dataset, with no retrieval, under one
head architecture. We do not test large variants of either model, do not test whether the
learning-rate transfer failure holds at other sizes, and do not evaluate on any dataset other
than ARC, so the generality of the hyperparameter finding is untested.

## 7. Conclusion

A single 184M-parameter encoder with no retrieval and no external corpus reaches 75.0% on
ARC-Easy and 56.1% on ARC-Challenge — about 29 points above the best baseline reported in the
original ARC paper on Challenge, ahead of every published closed-book encoder baseline we
could find, and ahead of GPT-3 175B scored zero-shot on Challenge, while remaining well below
retrieval-augmented ensembles and 11B-scale fine-tuned models. The whole study cost 110
GPU-minutes across 72 completed runs.

Three things determined that result more than anything we tuned. Which pretrained backbone we
started from was worth 9.4 to 19.8 validation points, far more than any hyperparameter in the
grid, and it was not a parameter-count effect. The best hyperparameters do not carry over
between backbones, and the ranges do not even overlap in their optima. And the remaining
errors are structured rather than random, at least along the surface dimensions our regex and
length bins can see: questions whose wording our negation pattern matches cost up to 24.4
points, questions whose answer options are long cost up to 12.5, and the model's confidence is
least trustworthy on exactly the subset where its accuracy is lowest. These categories are a
lexical proxy, not a reasoning taxonomy — they tell us where accuracy drops, not why. The most promising next step, following Section 5, is not a
bigger encoder but a second stage that sees all the options at once.

## References

Borchmann, Ł. (2025). *ARC 'Challenge' is not that challenging.* Findings of the Association
for Computational Linguistics: ACL 2025, 2797–2804. Vienna, Austria: Association for
Computational Linguistics. doi:10.18653/v1/2025.findings-acl.144

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., et al. (2020).
*Language models are few-shot learners.* Advances in Neural Information Processing Systems 33.
arXiv:2005.14165.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O.
(2018). *Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge.*
arXiv:1803.05457.

Clark, P., Etzioni, O., Khot, T., Khashabi, D., Mishra, B., Richardson, K., et al. (2020).
*From 'F' to 'A' on the N.Y. Regents science exams: An overview of the Aristo project.*
AI Magazine, 41(4). arXiv:1909.01958v2.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep
bidirectional transformers for language understanding.* NAACL-HLT. arXiv:1810.04805.

He, P., Gao, J., & Chen, W. (2023). *DeBERTaV3: Improving DeBERTa using ELECTRA-style
pre-training with gradient-disentangled embedding sharing.* ICLR.

Huang, Z., Wu, A., Zhou, J., Gu, Y., Zhao, Y., & Cheng, G. (2022). *Clues before answers:
Generation-enhanced multiple-choice QA.* NAACL 2022. arXiv:2205.00274.

Khashabi, D., Min, S., Khot, T., Sabharwal, A., Tafjord, O., Clark, P., & Hajishirzi, H.
(2020). *UnifiedQA: Crossing format boundaries with a single QA system.* Findings of the
Association for Computational Linguistics: EMNLP 2020, 1896–1907.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., et al. (2019). *RoBERTa: A robustly
optimized BERT pretraining approach.* arXiv:1907.11692.

Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR.

Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.*
Journal of the American Statistical Association, 22(158), 209–212.

## Figures

- **Figure 1** — `results/analysis/fig_hparam_heatmaps.png`. Best validation accuracy over the
  learning-rate × epochs grid, as a 2 × 3 array of heatmaps (two backbones × three training
  regimes). The champion cell in each panel is marked. All six panels are full 4 × 3 grids
  with no missing cells; note that the two backbones' rows are labelled with different learning
  rates ({2, 3, 5, 7} × 10⁻⁵ for BERT, {1, 1.5, 2, 3} × 10⁻⁵ for DeBERTa), so the panels are
  not directly stackable along the vertical axis, and each panel's colour scale is set within
  the panel.
- **Figure 2** — `results/analysis/fig_error_analysis.png`. Left: accuracy against mean
  confidence within each confidence bin, for Easy and Challenge test, with the
  perfect-calibration diagonal shown. Right: accuracy by question category, plotted as a
  deviation from the subset average, for both subsets.
- **Figure 3** — `results/analysis/fig_training_dynamics.png`. Left: per-epoch validation
  accuracy for the six sweep champions, with the 25% random-guess rate marked. Right: training
  loss curves for the same six runs, showing the per-epoch sawtooth and the two Challenge runs
  that never approach zero.

## Appendices

- **Appendix A** — full 26-row question-category table
  (`results/analysis/t5_error_categories.md`).
- **Appendix B** — all 72 completed runs with model, subset, learning rate, epochs, seed, best
  validation accuracy, epochs completed, mean seconds per epoch, and wall time
  (`results/analysis/t2_all_runs.md`); throughput summary (`t3_efficiency.md`); full
  five-bin calibration table for both subsets (`t4_calibration.md`).
- **Appendix C** — champion configurations and per-epoch trajectories, reproduced below.
- **Appendix D** — reproduction notes. The `configs/` directory holds two files, not one per
  sweep: `sweep.json` is a single template that pairs a backbone and subset with a `grid`
  block of learning rates and epoch counts and a `fixed` block of everything held constant
  (batch size 8, gradient accumulation 2, maximum length 128, weight decay 0.01, warmup ratio
  0.1, `keep_champion_last_pt: true`, `keep_all_checkpoints: false`), and the six sweeps in
  this paper were run by editing `model_name`, `subset` and `grid` in that one file rather
  than by committing six separate configurations. As checked in, it holds the DeBERTa /
  combined sweep (`sweep_deberta_combined`, seed 77, lr ∈ {1, 1.5, 2, 3} × 10⁻⁵,
  epochs ∈ {3, 4, 5}), so the other five grids are recoverable only from
  `results/analysis/t2_all_runs.md`, which records every completed run's model, subset,
  learning rate, epoch count and seed. The second file, `final.json`, is a three-seed
  replication harness (`lr` 3 × 10⁻⁵, 3 epochs, seeds 42/43/44,
  `keep_all_checkpoints: true`) described in Section 6; it is committed but was never run, so
  no result in this paper depends on it. We note that its hyperparameters were written before
  the final sweeps and no longer match any champion — the BERT / combined champion is now
  5 × 10⁻⁵ for 4 epochs — so anyone running it as checked in would be replicating a
  configuration this paper does not report. The training entry point is `src/train.py` driven by
  `main.py`, evaluation is `src/evaluate.py --save_predictions`, baselines are
  `src/baselines.py`, and every table and figure in this paper is regenerated by
  `python src/analyze.py`. Note that the sweep runner deletes each run's checkpoint directory
  as soon as that run is beaten, retaining only the champion's weights; this is why no BERT
  test evaluation exists.

### Appendix C — champion configurations and per-epoch validation trajectories

All runs use batch size 8, gradient accumulation 2 (effective batch 16), maximum length 128,
weight decay 0.01, warmup ratio 0.1, fp16 autocast on CUDA. Bold marks the retained
best-validation epoch. Training loss is reported for the first and last epoch of the run.

| Model | Subset | lr | Ep. | Seed | Validation accuracy by epoch (%) | Train loss (first → last) | s/epoch | Wall (s) |
|:---|:---|:---|---:|---:|:---|:---|---:|---:|
| BERT-base | Easy | 5 × 10⁻⁵ | 5 | 77 | 52.63 / 54.39 / 56.49 / **58.07** / 56.32 | 1.395 → 0.049 | 14.7 | 94.2 |
| BERT-base | Challenge | 3 × 10⁻⁵ | 5 | 77 | 26.09 / 35.79 / **42.81** / 38.46 / 38.80 | 1.398 → 0.848 | 8.0 | 58.1 |
| BERT-base | Combined | 5 × 10⁻⁵ | 4 | 77 | 45.11 / 48.56 / 49.14 / **50.17** | 1.379 → 0.311 | 22.4 | 105.3 |
| DeBERTa-v3-base | Easy | 3 × 10⁻⁵ | 5 | 77 | 71.05 / 75.26 / 74.56 / 75.44 / **76.14** | 1.391 → 0.083 | 24.6 | 154.2 |
| DeBERTa-v3-base | Challenge | 3 × 10⁻⁵ | 4 | 77 | 42.81 / 51.17 / **52.17** / 51.17 | 1.387 → 0.545 | 9.9 | 63.9 |
| DeBERTa-v3-base | Combined | 2 × 10⁻⁵ | 5 | 77 | 63.52 / 67.43 / **69.97** / 69.39 / 68.82 | 1.386 → 0.131 | 27.8 | 168.1 |

Training set sizes are 2,251 (Easy), 1,119 (Challenge), 3,370 (combined); validation sizes are
570, 299, and 869. The selected model for all test evaluation is the last row, checkpoint
`checkpoints/sweep_deberta_combined_lr2e-05_epochs5_20260726_022137/best`.
