# === Step 1: Artifact-to-Group Mapping ===
artifact_to_group = {
    # Coherence/Reflection Group
    "Incorrect reflection mapping": "Reflection",
    "Distorted window reflections": "Reflection",
    "Unrealistic specular highlights": "Reflection",

    # Anatomy Group
    "Dental anomalies in mammals": "Anatomy",
    "Anatomically incorrect paw structures": "Anatomy",
    "Unrealistic eye reflections": "Anatomy",
    "Misshapen ears or appendages": "Anatomy",
    "Anatomically impossible joint configurations": "Anatomy",
    "Misaligned bilateral elements in animal faces": "Anatomy",
    "Asymmetric features in naturally symmetric objects": "Anatomy",

    # Geometry/Structure Group
    "Impossible mechanical connections": "Structure",
    "Incorrect wheel geometry": "Structure",
    "Floating or disconnected components": "Structure",
    "Non-manifold geometries in rigid structures": "Structure",
    "Irregular proportions in mechanical components": "Structure",
    "Scale inconsistencies within single objects": "Structure",

    # Texture/Surface Group
    "Improper fur direction flows": "Texture",
    "Over-smoothing of natural textures": "Texture",
    "Texture bleeding between adjacent regions": "Texture",
    "Texture repetition patterns": "Texture",
    "Aliasing along high-contrast edges": "Texture",
    "Jagged edges in curved structures": "Texture",
    "Metallic surface artifacts": "Texture",

    # Lighting/Rendering Group
    "Depth perception anomalies": "Rendering",
    "Ghosting effects: Semi-transparent duplicates of elements": "Rendering",
    "Artificial noise patterns in uniform surfaces": "Rendering",
    "Abruptly cut off objects": "Rendering",
    "Blurred boundaries in fine details": "Rendering",
    "Unnatural color transitions": "Rendering",
    "Incorrect perspective rendering": "Rendering"
}

# === Step 2: ReAct-Style CoT Prompt Generator ===
def generate_react_prompt(artifact_name, group):
    return f"""
    SYSTEM:
    You are a vision expert analyzing a generated image.
    Your task is to describe how the artifact \"{artifact_name}\" appears in the image.
    Use the ReAct style with Chain-of-Thought reasoning:

    Thought: Reason about how {group}-related artifacts usually manifest.
    Action: Examine the image and identify visual signs supporting or rejecting that artifact.
    Observation: Describe what was found visually.
    Repeat the cycle as needed.

    Final Answer: {{\"description\": \"[Insert final summary of the artifact manifestation here]\"}}
    """

# === Step 3: Prompt MOLMO with CoT + ReAct ===
def prompt_molmo_with_react(image_path, artifact):
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    import torch
    from PIL import Image
    import re

    group = artifact_to_group.get(artifact, "General")
    prompt = generate_react_prompt(artifact, group)

    img = Image.open(image_path).resize((128,128))
    processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    inputs = processor.process(images=[img], text=prompt)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items() if torch.is_tensor(v)}
    inputs["images"] = inputs["images"].to(torch.bfloat16)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        match = re.search(r'"description":\s*"([^"]*)"', text)
        return match.group(1) if match else "No description available"

# === Step 4: Scoring with G-Eval inspired evaluation ===
def g_eval_score(description, artifact_name):
    eval_criteria = [
        f"Does the explanation of '{artifact_name}' clearly match a visual property?",
        "Is the explanation specific and not vague?",
        "Does the explanation highlight image impact?"
    ]

    prompt = """
    SYSTEM:
    Evaluate the following artifact explanation against criteria step-by-step.
    Explanation: {description}
    Artifact: {artifact_name}
    For each step, respond Yes/No/Somewhat and give 1-sentence justification.
    Then assign a final score from 0 to 1.
    Format:
    - Step 1: [answer] - [justification]
    - Step 2: [answer] - [justification]
    - Step 3: [answer] - [justification]
    - Final Score: [float score from 0 to 1]
    """.format(description=description, artifact_name=artifact_name)

    from transformers import pipeline
    judge = pipeline("text-generation", model="gpt2")
    response = judge(prompt, max_new_tokens=200)[0]['generated_text']
    match = re.search(r"Final Score:\s*([0-9\.]+)", response)
    return float(match.group(1)) if match else 0.5

# === Step 4A: Multimodal Judge (LLaVA) ===
from transformers import pipeline as mm_pipeline
judge_mm = mm_pipeline(
    "multimodal-causal-lm",
    model="llava/llava-1_5-llava-13b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def multimodal_judge(image_path, artifact, description):
    messages = [
        {"role": "system", "content": (
            "You are an expert visual forensic assistant. "
            "Given an image, an artifact name, and a description, "
            "evaluate whether the described artifact truly appears in the image."
        )},
        {"role": "user", "content": (
            f"Artifact: {artifact}\n"
            f"Description: {description}\n\n"
            "Answer with:\n"
            "VERDICT: Yes or No\n"
            "CONFIDENCE: <0–1 float>\n"
            "JUSTIFICATION: <one-sentence reasoning>"
        )}
    ]

    output = judge_mm(image=image_path, prompt=messages)
    text = output["generated_text"]
    verdict = "No"
    confidence = 0.0
    justification = ""
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except:
                pass
        elif line.startswith("JUSTIFICATION:"):
            justification = line.split(":", 1)[1].strip()

    verified = (verdict.lower() == "yes") and (confidence >= 0.5)
    return verified, confidence, justification

# === Step 4B: LLM-based Paraphraser ===
def paraphrase_description(description, style="technical"):
    from transformers import pipeline
    prompt = {
        "technical": f"Rewrite this description in formal scientific language: '{description}'",
        "educational": f"Explain this description simply for students: '{description}'",
        "caption": f"Describe this like a visual caption: '{description}'"
    }[style]
    paraphraser = pipeline("text-generation", model="gpt2")
    output = paraphraser(prompt, max_new_tokens=100)[0]['generated_text']
    return output.strip()

# === Step 5: Full Integration ===
def analyze_image_with_full_pipeline(image_name, image_path, artifact_list, report_style="technical"):
    results = {}
    for artifact in artifact_list:
        try:
            print(f"\nAnalyzing artifact: {artifact}")
            description = prompt_molmo_with_react(image_path, artifact)
            print("Generated description:", description)
            score = g_eval_score(description, artifact)
            print(f"Score: {score:.2f}")
            verified, confidence, justification = multimodal_judge(image_path, artifact, description)
            print(f"Multimodal Verdict: {verified}, Confidence: {confidence:.2f}, Justification: {justification}")
            if verified:
                paraphrased = paraphrase_description(description, style=report_style)
            else:
                paraphrased = "Not verified."
            results[artifact] = {
                "description": description,
                "score": score,
                "verified": verified,
                "confidence": confidence,
                "justification": justification,
                "paraphrased": paraphrased
            }
        except Exception as e:
            results[artifact] = {"description": "Error", "score": 0.0, "verified": False, "confidence": 0.0, "justification": str(e), "paraphrased": ""}
            print(f"Error analyzing {artifact}: {e}")
    return {
        "image": image_name,
        "artifacts": results,
        "top_artifacts": sorted(
            [item for item in results.items() if item[1]["verified"]],
            key=lambda x: x[1]['score'], reverse=True)[:5]
    }

# === Step 6: Report Generation ===
def generate_summary_report(image_results):
    report = f"\nImage: {image_results['image']}\nTop Artifacts:\n"
    for artifact, info in image_results['top_artifacts']:
        report += f"- {artifact} (score={info['score']:.2f}, confidence={info['confidence']:.2f}): {info['paraphrased']}\n"
    return report
