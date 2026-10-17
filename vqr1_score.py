"""VisualQuality-R1-7B (Wu et al., arXiv 2505.14460; Qwen2.5-VL-7B tuned with RL for IQA) as an NR metric.

Banana100 found it one of only two metrics that track iterative degradation, so it is the "proposed fix" we test.
Non-thinking prompt from the model card, with two departures so scores are reproducible:
greedy decoding instead of sampling, and NaN (counted) instead of the card's random 1-5 score when nothing parses.
8-bit weights: bf16 7B needs ~16.5 GB, more than our 16 GB card (Q-Align runs 8-bit for the same reason).
Writes <out>/scores/visualquality_r1.csv in score.py's format. --modes/--steps limit it to the images an analysis uses.
Usage: python vqr1_score.py --out out_ablate --modes clean sharpen@mild ...   |   python vqr1_score.py --test
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

MODEL = "TianheWu/VisualQuality-R1-7B"
PROMPT = ("You are doing the image quality assessment task. Here is the question: "
          "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
          "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. "
          "Please only output the final answer with only one score in <answer> </answer> tags.")


def parse(text):
    """Last <answer> block (or whole text if none), first number in it; NaN if there is no number."""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    n = re.search(r"\d+(\.\d+)?", m[-1] if m else text)
    return float(n.group()) if n else float("nan")


def load():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=BitsAndBytesConfig(load_in_8bit=True), dtype=torch.bfloat16, device_map="cuda")
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True)

    @torch.no_grad()
    def score(paths):
        x = proc(text=[text] * len(paths), images=[Image.open(p).convert("RGB") for p in paths],
                 padding=True, return_tensors="pt").to("cuda")
        y = model.generate(**x, max_new_tokens=128, do_sample=False)
        out = proc.batch_decode(y[:, x.input_ids.shape[1]:], skip_special_tokens=True)
        return [parse(o) for o in out], out
    return score


if __name__ == "__main__":
    assert parse("<answer>3.25</answer>") == 3.25 and parse("<think>1</think><answer> 4 </answer>") == 4.0
    assert parse("The rating is 2.5.") == 2.5 and parse("no idea") != parse("no idea")  # NaN != NaN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--modes", nargs="+")
    ap.add_argument("--steps", nargs="+", type=int)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--test", action="store_true", help="score clean vs damaged versions of 3 images and stop")
    a = ap.parse_args()
    score = load()
    if a.test:
        import time
        for i in ("001", "006", "008"):
            t = time.time()
            s, raw = score([f"out_ablate/clean/{i}/00.png", f"out_ablate/blur@strong/{i}/01.png",
                            f"out_ablate/noise@strong/{i}/01.png", f"out/naive/{i}/20.png"])
            print(i, "clean, blur_strong, noise_strong, chain_20:", s, f"{(time.time() - t) / 4:.2f} s/img", raw[:1])
        print(f"peak mem GB {torch.cuda.max_memory_allocated() / 1e9:.1f}")
        raise SystemExit
    imgs = sorted(p for p in Path(a.out).glob("*/*/*.png")
                  if (not a.modes or p.parts[-3] in a.modes) and (not a.steps or int(p.stem) in a.steps))
    rows, bad = [], 0
    for i in tqdm(range(0, len(imgs), a.bsz), desc="visualquality_r1", mininterval=30):
        b = imgs[i:i + a.bsz]
        s, raw = score(b)
        for p, v, r in zip(b, s, raw):
            bad += v != v
            if v != v:
                print("unparsed:", p, repr(r[:200]))
            rows.append(dict(mode=p.parts[-3], image=p.parts[-2], step=int(p.stem), metric="visualquality_r1",
                             score=v, lower_better=False))
    (Path(a.out) / "scores").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "scores" / "visualquality_r1.csv", index=False)
    print(f"done visualquality_r1 ({len(rows)}, unparsed {bad})")
