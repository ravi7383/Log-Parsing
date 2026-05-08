# Generalized Log Template Parser

Main notebook:

```text
Generalizing_log_parsing_.ipynb
```

This project builds a log template parser that can work across different log systems. The goal is simple: given a raw log line, the model should keep the stable event words and replace changing runtime values with `<*>`.

Example:

```text
Raw log content:
Failed password for root from 183.62.140.253 port 5212 ssh2

Predicted template:
Failed password for <*> from <*> port <*> ssh2
```

The final notebook is organized for thesis submission. Extra diagnostic outputs were removed so the results focus on the main evaluation, the ablation summary, the leave-one-out experiment, and the template examples.

## What The Notebook Produces

The notebook reports three main metrics.

| Metric | Meaning |
|---|---|
| TA | Token Accuracy. It checks whether each token is predicted correctly as static or variable. |
| PA | Parsing Accuracy. It checks whether the predicted template matches the ground-truth template closely enough. |
| GA | Grouping Accuracy. It checks whether log lines that belong to the same event type are placed in the same predicted group. |

The notebook also reports precision, recall, and F1-score for these measurements.

The main output files are:

| Output | Purpose |
|---|---|
| `outputs/reports/log_template_evaluation.xlsx` | Main seen and unseen evaluation results. |
| `outputs/reports/targeted_experiments.xlsx` | Targeted ablation and leave-one-out results. |
| `outputs/reports/model_results.json` | Compact metrics and configuration summary. |
| `outputs/figures/metric_heatmap.png` | Heatmap for TA, PA, and GA. |
| `outputs/figures/metric_bar_chart.png` | Average metric comparison. |
| `outputs/figures/precision_recall_f1.png` | Precision, recall, and F1 comparison. |
| `outputs/figures/grouping_behavior.png` | Predicted group counts by system. |
| `outputs/models/memaid_checkpoint.pt` | Saved model checkpoint from the final run. |


## Overall Workflow

```text
raw log line
  -> detect message content
  -> tokenize the message
  -> align ground-truth template when available
  -> encode the message with SentenceTransformer
  -> reduce 384 dimensions to 128 dimensions with SVD
  -> pass the vector through the MANN layer
  -> build token-level numeric features
  -> classify each token as static or variable
  -> decode the token labels into a template
  -> group similar predicted templates
  -> evaluate TA, PA, and GA
```

Ground truth is used for training and evaluation on benchmark data. For unseen systems, ground truth is not used to help prediction. It is used only after prediction to calculate accuracy.

## Five Example Logs

### OpenSSH

```text
Raw content:
Failed password for root from 183.62.140.253 port 5212 ssh2

Template:
Failed password for <*> from <*> port <*> ssh2
```

Here, `root`, the IP address, and the port number are runtime values. The surrounding words describe the event and should stay static.

### HDFS

```text
Raw content:
PacketResponder 1 for block blk_-1608999687919862906 terminating

Template:
PacketResponder <*> for block blk_<*> terminating
```

The response number and block id change between log lines. The event phrase stays the same.

### Spark

```text
Raw content:
Running task 2.0 in stage 0.0 (TID 2)

Template:
Running task <*> in stage <*> (TID <*>)
```

Task id, stage id, and TID are variables. The phrase `Running task ... in stage ...` identifies the event type.

### HPC

```text
Raw content:
Linkerror event interval expired

Template:
Linkerror event interval expired
```

This line has no changing value, so the whole message remains static.

### HealthApp

```text
Raw content:
calculateCaloriesWithCache totalCalories=1512

Template:
calculateCaloriesWithCache totalCalories=<*>
```

The method name and key are stable. The calorie value changes, so it becomes `<*>`.

## Main Methods

### 1. Configuration

The configuration cell stores the settings used by the whole notebook: dataset size, embedding dimension, memory dimension, number of memory slots, training epochs, support size, query size, and evaluation thresholds.

Current training setup:

```text
meta_epochs     = 100
tasks_per_epoch = 12
support_size    = 10
query_size      = 50
memory_slots    = 256
memory_dim      = 128
```


### 2. `load_system`

`load_system` reads one log system and converts it into a list of records. Each record keeps the raw line, detected message content, tokens, ground-truth template when available, token labels, and embedding fields.

Example record idea:

```text
raw_line  = "2015-10-18 INFO sshd: Failed password for root from ..."
content   = "Failed password for root from 183.62.140.253 port 5212 ssh2"
tokens    = ["Failed", "password", "for", "root", "from", ...]
template  = "Failed password for <*> from <*> port <*> ssh2"
labels    = [0, 0, 0, 1, 0, 1, 0, 1, 0]
```

### 3. Raw Metadata Boundary Detection

Raw logs often contain metadata before the real message. Timestamps, log levels, process ids, and component names usually appear at the start of the line. The model should parse the message, not the whole metadata header.

The final notebook uses a statistical boundary detector instead of a hand-written regex parser. It looks at token positions across many lines and estimates where the repeated message content begins.

Example:

```text
Raw line:
2015-10-18 18:01:48 INFO sshd: Failed password for root from 183.62.140.253 port 5212 ssh2

Detected content:
Failed password for root from 183.62.140.253 port 5212 ssh2
```

If the detector is not confident, it keeps more of the line instead of making an aggressive cut.

### 4. Ground-Truth Alignment

When benchmark templates are available, the notebook aligns the content tokens with the ground-truth template. This produces token labels:

```text
0 = static token
1 = variable token
```

Example:

```text
Content:
Failed password for root from 183.62.140.253 port 5212 ssh2

Template:
Failed password for <*> from <*> port <*> ssh2

Labels:
Failed=0, password=0, for=0, root=1, from=0, IP=1, port=0, 5212=1, ssh2=0
```

These labels are used to train the token classifier.

### 5. SentenceTransformer Encoding

The notebook uses `all-MiniLM-L6-v2` as a frozen sentence encoder. It turns each log message into a 384-dimensional vector.

For one log line:

```text
content -> (384,)
```

For 21,000 log lines:

```text
embedding matrix -> (21000, 384)
```

Rows are log lines. Columns are semantic embedding dimensions.

### 6. SVD Projection

The MANN layer works in 128 dimensions, but SentenceTransformer gives 384 dimensions. The notebook uses truncated SVD to learn a projection from 384 to 128 dimensions.

If the training embedding matrix is:

```text
X = (21000, 384)
```

the notebook first subtracts the training mean:

```text
X_centered = X - mean
```

Then SVD factorizes it:

```text
X_centered = U * S * Vt
```

For this shape, the economy SVD looks like:

```text
U  = (21000, 384)
S  = (384,)
Vt = (384, 384)
```

The first 128 rows of `Vt` are kept as the projection matrix:

```text
W_proj = (128, 384)
```

Projection then works like this:

```text
(21000, 384) @ (384, 128) = (21000, 128)
```

The notebook calculates retained variance from the fitted SVD values:

```text
sum(S[:128]^2) / sum(S^2)
```

In the final inspection cell, this is printed from the live object as:

```text
Variance retained from enc.svd_variance: 0.9432
```

The number can change slightly if the training data changes.

### 7. MAML Training

MAML helps the model learn parameters that can adapt quickly to a new log system.

Each training episode works like this:

```text
sample one training system
sample support lines
adapt on the support set
evaluate on the query set
update the original model from query performance
```

In this project:

```text
support set = 10 lines
query set   = 50 lines
```

This matters because unseen systems may have different formats. MAML teaches the model to adjust from a small support sample instead of depending only on one fixed global pattern.

### 8. MANN Memory Layer

The MANN layer gives the model external memory. It stores useful line representations and reads from similar previous patterns.

Main components:

| Component | Shape | Role |
|---|---|---|
| Input projection | `(1, 128)` | Puts the input vector into memory space. |
| Controller MLP | `(1, 128)` | Builds a hidden representation of the current log line. |
| Read query | `(1, 128)` | Searches memory for similar slots. |
| Memory matrix | `(256, 128)` | Stores 256 memory vectors. |
| Read scores | `(1, 256)` | Gives one similarity score per memory slot. |
| Read vector | `(1, 128)` | Weighted memory result. |
| Write gate | `(1, 128)` | Controls how much memory affects the final output. |
| EMA write head | memory update | Updates memory smoothly instead of replacing it suddenly. |
| Output projection | `(1, 128)` | Final line vector passed to token classification. |

The EMA update is:

```text
updated_slot = alpha * old_slot + (1 - alpha) * new_vector
```

This avoids noisy memory changes from one unusual log line.

### 9. Label Memory

The model also keeps a small memory of token-label behavior. For a new line, it checks similar stored examples and asks: in similar lines, was this position usually static or variable?

It contributes two values:

```text
memory_similarity
memory_label_score
```

These values become part of the token feature matrix.

### 10. Token Feature Matrix

For each token, the notebook builds an 11-dimensional feature vector. These are simple numeric descriptions of the token and its context.

| Feature | Meaning |
|---|---|
| digit_ratio | How much of the token is numeric. |
| alpha_ratio | How much of the token is alphabetic. |
| upper_ratio | How much is uppercase. |
| punctuation_ratio | How much is punctuation or symbols. |
| length_norm | Token length scaled to a fixed range. |
| position_norm | Token position inside the line. |
| previous_separator | Whether the previous token ends with separator-like punctuation. |
| next_separator | Whether the next token starts with separator-like punctuation. |
| mixed_alpha_digit | Whether letters and numbers appear together. |
| memory_similarity | Similarity to the nearest memory example. |
| memory_label_score | Memory-based static/variable suggestion. |

For a line with 5 tokens:

```text
feature matrix = (5, 11)
```

### 11. Token Classifier

The token classifier combines:

```text
MANN line vector + token feature vector
```

It outputs one logit per token:

```text
token logits = (number_of_tokens,)
```

After sigmoid, each token receives a probability of being variable.

### 12. Template Decoding

The decoder turns token labels into a template.

Example:

```text
Tokens:
["Failed", "password", "for", "root", "from", "183.62.140.253", "port", "5212", "ssh2"]

Predicted labels:
[0, 0, 0, 1, 0, 1, 0, 1, 0]

Template:
Failed password for <*> from <*> port <*> ssh2
```

The final prediction path is model-driven. It uses the classifier output and memory signals. It does not use a hand-written regex rule table to decide final static and variable tokens.

### 13. Template Reuse

After decoding, the notebook may reuse a stable template representative from the current evaluation run. This helps avoid group splitting when the model predicts a slightly less complete version of a template.

Example:

```text
Stable template:
jk2_init() Found child <*> in scoreboard slot <*>

New decoded template:
<*> Found child <*> in scoreboard slot <*>

Final reused template:
jk2_init() Found child <*> in scoreboard slot <*>
```

This step uses plain string comparison over static-token structure. It is meant to stabilize grouping, not to add dataset-specific parsing rules.

### 14. Evaluation

`evaluate` runs the model on a set of records and returns TA, PA, GA, precision, recall, F1, active memory slots, and line-level predictions.

PA uses a pattern-free flexible matcher. It allows small formatting differences when the main static meaning is preserved.

Accepted example:

```text
GT  : Progress of TaskAttempt attempt_<*> is : <*>.<*>
Pred: Progress of TaskAttempt <*> is: <*>
```

Rejected example:

```text
GT  : normal
Pred: gige temperature <*> <*> normal
```

The second prediction adds important static words, so it is a different event.

### 15. Grouping And GA

Predicted templates are grouped by their normalized template structure. GA compares those predicted groups with ground-truth groups.

A high GA means repeated event types are kept together. A low GA usually means the model split one real template into many predicted templates, or merged different real templates into one group.

The final visualization keeps only the group-count view. The older red line for flexible grouping behavior was removed because GA, GA precision, GA recall, and GA F1 already explain grouping quality more clearly.

### 16. Ablation Study

The targeted ablation compares three variants:

| Variant | Meaning |
|---|---|
| `full_existing` | Main model with MAML, memory, and token features. |
| `no_memory_maml` | MAML is kept, but the memory layer is removed. GA is still computed from predicted-template groups. |
| `memory_no_maml` | Memory is kept, but MAML adaptation and token-rule features are disabled. |

The final notebook prints the ablation metric table and the ablation summary by phase and variant. It no longer prints contribution deltas or winner counts.

### 17. Leave-One-Out Experiment

The targeted leave-one-out experiment trains without selected systems and evaluates on the held-out system. This checks whether the model can handle a format that was not present during training.

Example:

```text
train on: all selected systems except Hadoop
test on : Hadoop
```

This is useful because a parser that only works on systems it already saw is not enough for format-independent log parsing.

## Component Dimension Inspection

The final architecture cell prints only the dimension table and SVD path.

Expected output:

```text
Component dimensions derived from live notebook objects

SentenceTransformer embedding   (5, 384)
SVD projection matrix           (128, 384)
SVD-projected embeddings        (5, 128)
MANN controller MLP             (1, 128)
MANN read query                 (1, 128)
MANN memory matrix              (256, 128)
MANN read scores                (1, 256)
MANN write gate                 (1, 128)
MANN output projection          (1, 128)
Token feature matrix            (5, 11)
Token classifier logits         (5,)
Predicted labels                5 labels

SVD dimension reduction path
Function that fits projection: Encoder.fit_svd
Function that applies projection: project_embeddings
Raw embedding dimension from enc.encode(sample_contents).shape[-1]: 384
Projected dimension from project_embeddings(...).shape[-1]: 128
Projection matrix shape from W_proj.shape: (128, 384)
Variance retained from enc.svd_variance: 0.9432
```

## Important Functions

| Function or class | Main output | What it does |
|---|---|---|
| `load_system` | records | Loads one log system and prepares records. |
| `RawMetadataBoundaryDetector.detect` | content string | Finds the message part of a raw log line. |
| `attach_ground_truth` | labels and templates | Aligns benchmark templates with parsed tokens. |
| `Encoder.encode` | `(n, 384)` matrix | Creates SentenceTransformer embeddings. |
| `Encoder.fit_svd` | `(128, 384)` matrix | Learns the 384-to-128 projection. |
| `project_embeddings` | `(n, 128)` matrix | Applies the SVD projection. |
| `project_records` | updated records | Adds projected vectors to each record. |
| `MANN.forward` | line vector | Reads memory and returns a 128-dimensional representation. |
| `TokenClassifier.build_feat_matrix` | `(tokens, 11)` matrix | Builds numeric token features. |
| `TokenClassifier.forward` | token logits | Predicts one static/variable score per token. |
| `predict_line` | predicted template | Runs the model, decoder, and template reuse for one log line. |
| `evaluate` | metric dictionary | Calculates TA, PA, GA, precision, recall, and F1. |
| `export_evaluation_workbook` | Excel workbook | Saves main evaluation tables. |
| `run_targeted_leave_one_out` | experiment tables | Runs the leave-one-out experiment. |
| `run_targeted_ablation_existing_full` | experiment tables | Runs the targeted ablation study. |

## Final Notes

Use this notebook for final results:

```text
Generalizing_log_parsing_final_clean.ipynb
```

The core idea is:

```text
SentenceTransformer gives semantic vectors.
SVD compresses them from 384 to 128 dimensions.
MAML helps the model adapt to new formats.
MANN stores useful previous log patterns.
Token features describe each word locally.
The classifier predicts static or variable tokens.
The decoder builds the final template.
TA, PA, and GA measure the result.
```
