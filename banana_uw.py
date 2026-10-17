"""Banana100-style iterative editing on underwater images with Nano Banana (Gemini image models on Vertex AI).

Each step sends the previous output back with the same prompt in a fresh request (no chat history), as Banana100 did.
Step 00 = UIEB raw image as used everywhere else (out_ablate/clean). Outputs keep the model's native resolution, so
compare step i with step 1 (Banana100's delta), not with step 0, to avoid a resolution confound.
Layout matches score.py: out_banana/<model>_<prompt>/<image>/<step>.png ; token usage -> out_banana/usage.csv.
Resumable: an existing chain continues from its last saved step.
Auth: gcloud application-default login; GOOGLE_CLOUD_QUOTA_PROJECT must name the billed project.
Usage: python banana_uw.py --model gemini-3-pro-image --prompt replica --images 001 006 --steps 50
"""
import argparse
import csv
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROMPTS = {
    "replica": "Produce an exact replica of the provided image, with no alterations.",  # Banana100's main prompt
    "enhance": "Enhance this underwater photo: correct the colour cast and improve clarity and contrast, "
               "without changing its content.",  # re-restoring an already-restored image
}
SHORT = {"gemini-3-pro-image": "nbpro", "gemini-3.1-flash-image": "nb2"}
OUT = Path("out_banana")


def chain(client, model, prompt, img_id, steps, log):
    from google.genai import types
    d = OUT / f"{SHORT.get(model, model)}_{prompt}" / img_id
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "00.png").exists():
        Image.open(f"out_ablate/clean/{img_id}/00.png").save(d / "00.png")
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=types.ImageConfig(image_size="1K"))
    for s in range(1, steps + 1):
        if (d / f"{s:02d}.png").exists():
            continue
        src = (d / f"{s - 1:02d}.png").read_bytes()
        attempt = 0
        while True:
            try:
                r = client.models.generate_content(
                    model=model, config=cfg,
                    contents=[types.Part.from_bytes(data=src, mime_type="image/png"), PROMPTS[prompt]])
                parts = [p for p in (r.candidates[0].content.parts if r.candidates and r.candidates[0].content else [])
                         if p.inline_data]
                if parts:
                    break
                why = r.candidates[0].finish_reason if r.candidates else r.prompt_feedback
            except Exception as e:  # rate limit / transient server error
                why = e
            attempt += 1
            # 429 = Vertex shared-quota throttling (bursts last >20 min on gemini-3-pro-image): wait it out forever.
            # Anything else (safety block, no image, bad request) gives up after 8 tries.
            if attempt >= 8 and "429" not in str(why):
                print(f"STOP {d} step {s}: no image after {attempt} tries ({why})")
                return
            time.sleep(min(10 * 2 ** min(attempt, 5), 300))
        # write then rename: a host crash mid-save once left a 0-byte step that resume treated as done
        tmp = d / f"{s:02d}.tmp.png"
        Image.open(io.BytesIO(parts[0].inline_data.data)).convert("RGB").save(tmp)
        tmp.replace(d / f"{s:02d}.png")
        u = r.usage_metadata
        log([model, prompt, img_id, s, u.prompt_token_count, u.candidates_token_count])
    print(f"done {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3-pro-image")
    ap.add_argument("--prompt", choices=PROMPTS, default="replica")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from google import genai
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"], location="global")
    OUT.mkdir(exist_ok=True)
    f = open(OUT / "usage.csv", "a", newline="")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["model", "prompt", "image", "step", "input_tokens", "output_tokens"])

    lock = threading.Lock()

    def log(row):
        with lock:
            w.writerow(row)
            f.flush()
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda i: chain(client, a.model, a.prompt, i, a.steps, log), a.images))
