# Norwegian Embedding Fine-tuning
# Fine-tunes `perplexity-ai/pplx-embed-v1-0.6b` for Norwegian retrieval.
#
# Training data (all MTEB-safe):
#   - mMARCO-no — Norwegian MS MARCO translation
#   - NorNLI — Norwegian NLI pairs + contradiction triples
#   - NorSumm — Norwegian news article/summary pairs (dev split only)
#
# Eval data (never used in training):
#   - VG-RAG benchmark (held-out benchmark)
#   - NorQuad, SNL, MIRACL-no via MTEB
#
# Designed to run in Google Colab (T4 GPU, 16 GB VRAM).
# Install:
#   pip install sentence-transformers>=3.0.0 datasets transformers faiss-gpu mteb tqdm

# ── 1. Mount Google Drive (for saving checkpoints) ──────────────────────────
from google.colab import drive

drive.mount("/content/drive")

import os

SAVE_DIR = "/content/drive/MyDrive/nor-pplx-embed"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Checkpoints will be saved to: {SAVE_DIR}")

# ── 2. Config — adjust these before running ─────────────────────────────────

BASE_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"

# Colab T4 has 16GB VRAM — these settings are safe
MMARCO_MAX = 50_000  # increase to 200k on A100
BATCH_SIZE_S1 = 8  # Stage 1 — MNRL benefits from larger batches
BATCH_SIZE_S2 = 4  # Stage 2 — hard negatives, explicit triples
EPOCHS_S1 = 1
EPOCHS_S2 = 1
SKIP_TSDAE = True  # set False to run TSDAE warm-up first (slow)

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 3. Load training data ────────────────────────────────────────────────────

import random
import logging
from datasets import load_dataset
from sentence_transformers import InputExample

logging.basicConfig(level=logging.INFO)
random.seed(42)

# mMARCO-no
print("Loading mMARCO-no...")
mmarco_pairs = []
try:
    mmarco_ds = load_dataset("unicamp-dl/mmarco", "norwegian", split="train", streaming=True)
    for row in mmarco_ds:
        if len(mmarco_pairs) >= MMARCO_MAX:
            break
        query = row["query"].strip()
        positives = row.get("positive_passages", [])
        if positives:
            passage = positives[0]["text"].strip()
            if query and passage:
                mmarco_pairs.append(InputExample(texts=[query, passage]))
    print(f"  mMARCO-no: {len(mmarco_pairs):,} pairs")
except RuntimeError as e:
    print(f"  mMARCO-no not available ({e}) — skipping")
    mmarco_pairs = []

# NorNLI
print("Loading NorNLI...")
nornli_mnrl, nornli_triplets = [], []
try:
    nornli_ds = load_dataset("ltg/nornli", split="train")

    entailment, contradiction = {}, {}
    for row in nornli_ds:
        premise = row["premise"].strip()
        hyp = row["hypothesis"].strip()
        label = row["label"]  # 0=entailment, 1=neutral, 2=contradiction
        if label == 0:
            entailment.setdefault(premise, []).append(hyp)
        elif label == 2:
            contradiction.setdefault(premise, []).append(hyp)

    premises_with_both = set(entailment) & set(contradiction)
    for premise in premises_with_both:
        pos = random.choice(entailment[premise])
        neg = random.choice(contradiction[premise])
        nornli_triplets.append(InputExample(texts=[premise, pos, neg]))
        nornli_mnrl.append(InputExample(texts=[premise, pos]))
    for premise in set(entailment) - premises_with_both:
        for pos in entailment[premise]:
            nornli_mnrl.append(InputExample(texts=[premise, pos]))
    print(f"  NorNLI: {len(nornli_mnrl):,} MNRL pairs, {len(nornli_triplets):,} triplets")
except Exception as e:
    print(f"  NorNLI not available ({e}) — skipping")

# NorSumm (dev split only — test is held out)
print("Loading NorSumm...")
norsumm_pairs = []
try:
    norsumm_ds = load_dataset("SamiaT/NorSumm", "nb")
    for split in ["train", "validation"]:
        if split not in norsumm_ds:
            continue
        for row in norsumm_ds[split]:
            article = row.get("article", "").strip()
            if not article:
                continue
            # The row contains a 'summaries' key which is a list of summaries
            for summary_text in row.get("summaries", []):
                summary = summary_text.strip()
                if summary:
                    norsumm_pairs.append(InputExample(texts=[summary, article]))
    print(f"  NorSumm: {len(norsumm_pairs):,} pairs")
except Exception as e:
    print(f"  NorSumm not available ({e}) — skipping")

# Combine
all_mnrl_pairs = mmarco_pairs + nornli_mnrl + norsumm_pairs
random.shuffle(all_mnrl_pairs)
print(f"\nTotal MNRL pairs:  {len(all_mnrl_pairs):,}")
print(f"Total triplets:    {len(nornli_triplets):,}")

# ── 4. (Optional) TSDAE warm-up — skip if SKIP_TSDAE=True ──────────────────

if SKIP_TSDAE:
    print("Skipping TSDAE — using base model directly for Stage 1")
    stage1_init = BASE_MODEL
else:
    from datasets import load_dataset as ld
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import losses as st_losses
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    print("Loading HPLT Norwegian for TSDAE...")
    hplt_ds = ld("HPLT/HPLT2.0_cleaned", "nob_Latn", split="train", streaming=True)
    hplt_sentences = []
    for row in hplt_ds:
        if len(hplt_sentences) >= 20_000:
            break
        text = row.get("text", "").strip()
        words = text.split()
        if 10 <= len(words) <= 256:
            hplt_sentences.append(text)
    print(f"  {len(hplt_sentences):,} sentences")

    tsdae_model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)
    tsdae_dataset = DenoisingAutoEncoderDataset(hplt_sentences)
    tsdae_loss = st_losses.DenoisingAutoEncoderLoss(
        tsdae_model,
        decoder_name_or_path=BASE_MODEL,
        tie_encoder_decoder=True,
    )
    tsdae_args = SentenceTransformerTrainingArguments(
        output_dir=f"{SAVE_DIR}/tsdae",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=3e-5,
        fp16=True,
        save_strategy="epoch",
        logging_steps=200,
    )
    SentenceTransformerTrainer(
        model=tsdae_model,
        args=tsdae_args,
        train_dataset=tsdae_dataset,
        loss=tsdae_loss,
    ).train()
    stage1_init = f"{SAVE_DIR}/tsdae"
    tsdae_model.save(stage1_init)
    print(f"TSDAE saved to {stage1_init}")

# ── 5. Stage 1 — MNRL + TripletLoss ─────────────────────────────────────────

from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def to_hf(examples, columns):
    return HFDataset.from_dict(
        {col: [ex.texts[i] for ex in examples] for i, col in enumerate(columns)}
    )


print(f"Loading model from: {stage1_init}")
model = SentenceTransformer(stage1_init, trust_remote_code=True)

mnrl_dataset = to_hf(all_mnrl_pairs, ["anchor", "positive"])

train_datasets = {"mnrl": mnrl_dataset}
train_losses = {"mnrl": losses.MultipleNegativesRankingLoss(model)}

if len(nornli_triplets) > 0:
    triplet_dataset = to_hf(nornli_triplets, ["anchor", "positive", "negative"])
    triplet_loss = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.15,
    )
    train_datasets["triplet"] = triplet_dataset
    train_losses["triplet"] = triplet_loss

s1_args = SentenceTransformerTrainingArguments(
    output_dir=f"{SAVE_DIR}/stage1",
    num_train_epochs=EPOCHS_S1,
    per_device_train_batch_size=BATCH_SIZE_S1,
    gradient_accumulation_steps=2,
    warmup_ratio=0.05,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
    dataloader_num_workers=2,
)

SentenceTransformerTrainer(
    model=model,
    args=s1_args,
    train_dataset=train_datasets,
    loss=train_losses,
).train()

model.save(f"{SAVE_DIR}/stage1/final")
print(f"Stage 1 saved to {SAVE_DIR}/stage1/final")

# ── 6. Hard negative mining ──────────────────────────────────────────────────

import faiss
import numpy as np
from tqdm.auto import tqdm

queries = [p.texts[0] for p in all_mnrl_pairs]
positives = [p.texts[1] for p in all_mnrl_pairs]
corpus = list(set(positives))
corpus_idx = {text: i for i, text in enumerate(corpus)}

print(f"Encoding corpus ({len(corpus):,} passages)...")
corpus_emb = model.encode(
    corpus,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)

print(f"Encoding queries ({len(queries):,})...")
query_emb = model.encode(
    queries,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype(np.float32)

dim = corpus_emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(corpus_emb)

TOP_K = 20
print(f"FAISS search (top_k={TOP_K})...")
scores, indices = index.search(query_emb, TOP_K)

hard_triplets = []
for i, (query, positive) in enumerate(zip(queries, positives)):
    hard_neg = None
    for score, idx in zip(scores[i], indices[i]):
        candidate = corpus[idx]
        if candidate != positive:
            hard_neg = candidate
            break
    if hard_neg is None:
        hard_neg = corpus[np.random.randint(len(corpus))]
    hard_triplets.append(InputExample(texts=[query, positive, hard_neg]))

print(f"Mined {len(hard_triplets):,} hard-negative triplets")

# ── 7. Stage 2 — MNRL with hard negatives ───────────────────────────────────

hard_dataset = to_hf(hard_triplets, ["anchor", "positive", "negative"])
mnrl_loss2 = losses.MultipleNegativesRankingLoss(model)

s2_args = SentenceTransformerTrainingArguments(
    output_dir=f"{SAVE_DIR}/stage2",
    num_train_epochs=EPOCHS_S2,
    per_device_train_batch_size=BATCH_SIZE_S2,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    fp16=True,
    save_strategy="epoch",
    logging_steps=50,
    dataloader_num_workers=2,
)

SentenceTransformerTrainer(
    model=model,
    args=s2_args,
    train_dataset={"mnrl_hard": hard_dataset},
    loss={"mnrl_hard": mnrl_loss2},
).train()

model.save(f"{SAVE_DIR}/stage2/final")
print(f"Stage 2 saved to {SAVE_DIR}/stage2/final")

# ── 8. Evaluate on VG-RAG benchmark ─────────────────────────────────────────
# Adjust column names below to match your actual dataset schema

import numpy as np

print("Loading VG-RAG benchmark...")
vg_ds = load_dataset("zrmarine/vg-rag-benchmark", split="test")
print(f"  {len(vg_ds):,} examples")
print(f"  Columns: {vg_ds.column_names}")

# ← adjust these to match your dataset's actual column names
QUERY_COL = "question"
PASSAGE_COL = "context"

queries = vg_ds[QUERY_COL]
passages = vg_ds[PASSAGE_COL]

q_emb = model.encode(queries, normalize_embeddings=True, show_progress_bar=True)
p_emb = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)

sim = q_emb @ p_emb.T

hits1 = hits5 = hits10 = mrr = 0
n = len(queries)
for i in range(n):
    rank = int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1
    mrr += 1.0 / rank
    if rank == 1:
        hits1 += 1
    if rank <= 5:
        hits5 += 1
    if rank <= 10:
        hits10 += 1

print(f"\nVG-RAG Results (n={n})")
print(f"  Hits@1  : {hits1/n:.3f}")
print(f"  Hits@5  : {hits5/n:.3f}")
print(f"  Hits@10 : {hits10/n:.3f}")
print(f"  MRR     : {mrr/n:.3f}")

# ── 9. (Optional) MTEB Norwegian eval ──────────────────────────────────────

import mteb

MTEB_TASKS = ["NorQuadRetrieval", "SNLRetrieval", "MIRACLRetrieval"]

mteb_model = mteb.get_model(f"{SAVE_DIR}/stage2/final")
tasks = mteb.get_tasks(tasks=MTEB_TASKS, languages=["nob", "nno"])
results = mteb.evaluate(mteb_model, tasks)

for task, result in zip(MTEB_TASKS, results):
    print(f"{task}: {result.get_score():.3f}")
