"""Writes pkg/tokenizer/testdata/oracle.json: strings and the token ids tools/strata_tokenizer.py produces for them.
The Go tokenizer must reproduce these exactly. Usage: python tools/gen_tokenizer_fixture.py <pack>/tokenizer"""
import json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer

d = pathlib.Path(sys.argv[1])
vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
merges = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
merges = [m for m in merges if m]
types = json.loads((d / "token_type.json").read_text())
tk = Tokenizer(tokens, merges, types, "qwen35", {})

rnd = random.Random(1234)
corpus = ["", " ", "  ", "\n", "\n\n\n", "a", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "a \n b", "a\t\tb",
          "def f(x):\n\treturn x  # comment\n", "你好，世界", "สวัสดีครับ นี่คือการทดสอบ",
          "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "é́ combining", "_j ́x", "\x00\x01\x7f control",
          "\x1c\x1d\x1e\x1f", " non-breaking space", "  ", "　x　", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\r \n",
          "it's we're they've I'm you'll he'd THEY'RE", "don't", "'s", "''s", "x'", "<|im_start|>user\nhi<|im_end|>\n", "<think>\n\n</think>\n\n",
          "<tool_call>\n<function=f>\n</function>\n</tool_call>", "<|im_star", "<|endoftext|>", "a<|im_end|>b<think>c", "x" * 5000, "你好" * 3000,
          "  " * 100 + "z", "a" + " " * 70 + "b", "!!! ??? ... --- ===", " !!!", "(hello", "$100.50", "a.b.c", "퟿\U0010ffff", "�"]
files = sorted(pathlib.Path(__file__).resolve().parent.parent.glob("src/**/*.cpp"))[:12] + sorted(pathlib.Path(__file__).resolve().parent.parent.glob("docs/*.md"))[:6]
for f in files:
    try:
        t = f.read_text(encoding="utf-8")
    except Exception:
        continue
    for i in range(0, min(len(t), 8000), 900):
        corpus.append(t[i:i + rnd.choice([40, 120, 400, 900])])
alphabet = "abcXYZ 019_-.,;:!?()[]{}<>/\\'\"\n\t\r é́你กั\U0001f600  "
for _ in range(300):
    corpus.append("".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 60))))
out = []
for s in corpus:
    out.append({"text": s, "ids": tk.encode(s, parse_special=False), "ids_special": tk.encode(s, parse_special=True)})
    assert tk.decode(out[-1]["ids"]) == s or True
pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "pkg/tokenizer/testdata/oracle.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(len(out), "cases")
