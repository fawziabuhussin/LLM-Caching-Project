"""Asynchronous replay of JSON‑L trace. Collects latency & hits.
Usage:
  python replay_driver.py sharegpt_eval.jsonl
Saves results to `results/*.csv`."""
import json, sys, asyncio, aiohttp, csv, time
import numpy as np
from tqdm import tqdm
from policy import CALRUCache

CACHE = CALRUCache(capacity=512)
LAT = []
HITS = 0
TOTAL = 0

async def fetch_llm(session, prompt):
    # Dummy endpoint — replace with real LLM call
    await asyncio.sleep(0.01)  # simulate 10 ms
    return prompt[::-1]

async def handle(session, prompt):
    global HITS, TOTAL
    t0 = time.perf_counter()
    # Define a synchronous fetch_fn for the cache, but use await for misses
    hit_result = {}
    def backend(key):
        # This will only be called if key is not in cache
        hit_result['val'] = None
        return None  # placeholder, will be replaced below
    # Try cache first
    val, hit = CACHE.get(prompt, backend)
    if not hit:
        # Actually fetch asynchronously and update cache
        val = await fetch_llm(session, prompt)
        feats = np.array([len(prompt), len(val)])
        cost_pred = CACHE.model.predict(feats)
        CACHE.model.partial_fit(feats, len(val.split()))
        CACHE._consider_eviction(cost_pred)
        CACHE.cache[prompt] = (val, cost_pred, time.time())
    t1 = time.perf_counter() - t0
    LAT.append(t1)
    TOTAL += 1
    if hit:
        HITS += 1

async def main(trace_path):
    global loop
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession() as session:
        with open(trace_path, 'r', encoding='utf-8') as f:
            prompts = (json.loads(l)['prompt'] for l in f)
            tasks = [handle(session, p) for p in prompts]
            for chunk in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await chunk
    # write results
    with open('results/latencies.csv', 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['latency_ms'])
        cw.writerows([[x*1000] for x in LAT])
    print(f"Hit‑Rate: {HITS/TOTAL:.2%} | Mean Lat: {np.mean(LAT):.3f}s")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python replay_driver.py <trace.jsonl>'); sys.exit(1)
    asyncio.run(main(sys.argv[1]))
