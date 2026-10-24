"""Prompt set v2: 160 prompts, seeded, stratified over 5 sources; every 5th prompt of each source is held out (split=test).
usage: python ncm/build_prompts_v2.py OUT.json   (writes [{text, source, split}], texts capped at 240 chars at a word boundary)"""
import glob, json, random, sys
import pandas as pd

H = "/home/khing/.cache/huggingface/hub"
random.seed(1234)


def cap(s, n=240):
    s = " ".join(s.split()) if "\n" not in s else s.strip()
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0]


def dolly(n):
    f = glob.glob(f"{H}/datasets--databricks--databricks-dolly-15k/snapshots/*/*.jsonl")[0]
    rows = [json.loads(l) for l in open(f)]
    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    cats = sorted(by)
    out = []
    for i in range(n):  # round-robin over categories
        r = random.choice(by[cats[i % len(cats)]])
        out.append(cap(r["instruction"] + ("\n" + r["context"] if r["context"] else "")))
    return out


def oasst(n):
    f = glob.glob(f"{H}/datasets--OpenAssistant--oasst1/snapshots/*/data/train-*.parquet")[0]
    df = pd.read_parquet(f)
    df = df[(df.lang == "en") & (df.role == "prompter") & df.parent_id.isna()]
    return [cap(t) for t in df.text.sample(n, random_state=1234)]


def wiki(n):
    f = glob.glob(f"{H}/datasets--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet")[0]
    df = pd.read_parquet(f)
    paras = [t.strip() for t in df.text if len(t.strip()) > 300 and not t.strip().startswith("=")]
    return [cap(t) for t in random.sample(paras, n)]


def code(n):
    files = [f for pat in ("src/**/*.cpp", "pkg/**/*.go", "ncm/*.py", "src/**/*.cu", "include/**/*.hpp") for f in glob.glob(pat, recursive=True)]
    out = []
    while len(out) < n:
        lines = open(random.choice(files), errors="ignore").read().split("\n")
        i = random.randrange(0, max(1, len(lines) - 6))
        s = "\n".join(lines[i:i + 5]).strip()
        if len(s) > 60:
            out.append(cap(s))
    return out


def numeric(n):
    out = []
    for i in range(n):
        a, b = random.randint(12, 999), random.randint(12, 999)
        out.append([f"What is {a} * {b}? Answer with the number only.", f"Compute {a} + {b} - {b // 2}.",
                    f"Order number {random.randint(10**5, 10**6)} was shipped on 2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}. Repeat the order number:",
                    f"Write the first {random.randint(6, 12)} square numbers starting from {random.randint(2, 20)}^2."][i % 4])
    return out


if __name__ == "__main__":
    sets = [("dolly", dolly(50)), ("oasst", oasst(30)), ("wiki", wiki(30)), ("code", code(30)), ("numeric", numeric(20))]
    rows = []
    for src, texts in sets:
        for i, t in enumerate(texts):
            rows.append(dict(text=t, source=src, split="test" if i % 5 == 4 else "train"))
    json.dump(rows, open(sys.argv[1], "w"), indent=0)
    print(len(rows), "prompts;", sum(r["split"] == "test" for r in rows), "held out")
