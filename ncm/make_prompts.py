"""Write the versioned prompt set as JSON; ncm/tokenize (Go, Strata's own tokenizer) turns it into --tokens-file inputs.
usage: python ncm/make_prompts.py prompts.json"""
import json, sys

PROMPTS = [  # v1: 12 prompts over code, math, prose, structured data, multi-turn, numeric recall
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "#include <stdio.h>\n\nint main(void) {\n",
    "class LRUCache:\n    def __init__(self, capacity: int):\n",
    "Question: What is 17 * 23? Show the steps.\nAnswer:",
    "Prove that the sum of the first n odd numbers is n squared.\nProof:",
    "The history of the printing press begins in the fifteenth century, when",
    "Explain in two sentences why the sky is blue.\n",
    '{"name": "Ada", "languages": ["en", "fr"], "address": {"city": "London", "zip": "',
    "SELECT name, COUNT(*) AS n FROM orders WHERE created_at >= '2024-01-01' GROUP BY",
    "<|im_start|>user\nWrite a haiku about autumn rain.<|im_end|>\n<|im_start|>assistant\n",
    "Invoice #48213 total: $1,204.50. Account number 7781-2290-4413. Repeat the account number:",
    "| city | population |\n|---|---|\n| Tokyo | 37400068 |\n| Delhi | 28514000 |\n|",
]

if __name__ == "__main__":
    json.dump(PROMPTS, open(sys.argv[1], "w"))
