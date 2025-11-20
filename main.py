#!/usr/bin/env python3

import os
import argparse
import numpy as np
from io import BytesIO
from typing import Optional
import base64

import requests
from PIL import Image
import torch
from flask import Flask, request, jsonify
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPImageProcessor, CLIPProcessor, CLIPModel
import rembg

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
}

device = "cpu"

app = Flask(__name__)

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

safety_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")

interrogation_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
interrogation_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

def load_interrogation_lists():
    path = Path("ranking_lists")
    groups: Dict[str, List[str]] = {}

    for file in root.glob("*.txt"):
        with file.open("r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]

        if not lines:
            continue

        groups[path.stem] = lines

    return groups

def strip_background_image(url):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    image = Image.open(BytesIO(resp.content)).convert("RGB")

    session = rembg.new_session("u2net")
    output = rembg.remove(
        image,
        alpha_matting=True
    )
    
    del session
    return output

def interrogate_image(url):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    image = Image.open(BytesIO(resp.content)).convert("RGB")

    results: Dict[str, List[Tuple[str, float]]] = {}

    for group_name, prompts in groups.items():
        if not prompts:
            continue
    
    all_logits = []

    for start in range(0, len(prompts), 64):
        end = start + 64
        batch_texts = prompts[start:end]

        inputs = interrogation_processor(
            text=batch_texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            all_logits.append(logits.cpu())
    
    logits_file = torch.cat(all_logits, dim=1)[0]

    probs = logits_file.softmax(dim=-1)

    k = min(5, len(prompts))
    top_probs, top_indices = torch.topk(probs, k=k)

    file_results: List[Tuple[str, float]] = []
    for idx, p in zip(top_indices.tolist(), top_probs.tolist()):
        files_results.append((prompts[idx], float(p)))

    results[group_name] = file_results

    return results

def caption_image(url):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    image = Image.open(BytesIO(resp.content)).convert("RGB")
    inputs = caption_processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            min_new_tokens=20,
            max_new_tokens=50,
            top_p=0.9,
            repetition_penalty=1.4,
            do_sample=True,
            num_beams=7,
        )
    
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

def safety_check_image(url):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    image = Image.open(BytesIO(resp.content)).convert("RGB")

    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[None, ...]

    inputs = safety_processor(image, return_tensors="pt").to(device)

    _, has_nsfw_concept = safety_checker(
        images=image_np,
        clip_input=inputs.pixel_values,
    )

    return bool(has_nsfw_concept[0])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/caption", methods=["POST"])
def caption_endpoint():
    data = request.get_json(silent=False) or {}

    url = data.get("url")
    
    if not isinstance(url, str) or not url:
        return jsonify({"error": "JSON body must include a non-empty 'url' string"}), 400

    try:
        caption = caption_image(url)

        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/safetycheck", methods=["POST"])
def safetycheck_endpoint():
    data = request.get_json(silent=True) or {}

    url = data.get("url")
    
    if not isinstance(url, str) or not url:
        return jsonify({"error": "JSON body must include a non-empty 'url' string"}), 400

    try:
        safety_check = safety_check_image(url)

        return jsonify({"nsfw": safety_check})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/interrogate", methods=["POST"])
def interrogate_endpoint():
    data = request.get_json(silent=False) or {}

    url = data.get("url")
    
    if not isinstance(url, str) or not url:
        return jsonify({"error": "JSON body must include a non-empty 'url' string"}), 400

    try:
        interrogation = interrogate_image(url)

        return jsonify({"interrogation": 
            {
                group_name: {text: prob for text, prob in entries}
                for group_name, entries in interrogation.items()
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stripbackground", methods=["POST"])
def stripbackground_endpoint():
    data = request.get_json(silent=False) or {}

    url = data.get("url")
    
    if not isinstance(url, str) or not url:
        return jsonify({"error": "JSON body must include a non-empty 'url' string"}), 400

    try:
        image = strip_background_image(url)

        buffer = BytesIO()
        image.save(buffer, format="WebP", quality=95, method=6)

        return jsonify({"strip_background": "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(
        description="Caption an image URL (e.g. WebP) using BLIP."
    )
    parser.add_argument(
        "-i",
        nargs=1,
        help="IP Address",
    )
    parser.add_argument(
        "-p",
        nargs=1,
        help="Port",
    )

    args = parser.parse_args()

    app.run(host=args.i[0], port=args.p[0], debug=False)
        
if __name__ == "__main__":
    main() 