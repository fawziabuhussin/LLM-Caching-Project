# make_mixed_2k.py
import json, os, random
os.makedirs("data", exist_ok=True)

repeats = [
    ("hello world", 500),
    ("test prompt", 400),
    ("another example", 300),
    ("what's the weather today?", 200),
    ("Please summarize: " + "lorem "*120, 100),  # long, costly, repeated
]  # subtotal = 1500

uniq_n = 500
with open("data/mixed_2k.jsonl", "w", encoding="utf-8") as out:
    # repeats (interleaved a bit to make it realistic)
    for text, n in repeats:
        for _ in range(n):
            out.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
    # uniques
    for i in range(uniq_n):
        out.write(json.dumps({"prompt": f'unique prompt {i} ' + 'word '*(i%20)}, ensure_ascii=False) + "\n")
print("Wrote data/mixed_2k.jsonl")
