from semantic_pipeline import SemanticForensicInterpreter


def vlm_reason_step(**kwargs):
    return "Observed repeated patterns, inconsistent gradients."


def vlm_action_step(**kwargs):
    return {"action": "inspect_patch_12", "stop": False}


def vlm_summarize_trace(**kwargs):
    return "Several superpixels show GAN-like inconsistencies, consistent with diffusion inpainting."


prompts = [
    "GAN-generated texture inconsistency",
    "diffusion inpainting trace artifacts",
    "copy-move duplication",
    "noise residual mismatch",
]

interpreter = SemanticForensicInterpreter(
    prompts=prompts,
    reason_fn=vlm_reason_step,
    act_fn=vlm_action_step,
    summarize_fn=vlm_summarize_trace,
)


def run_demo(image, patches_coarse, patches_fine, superpixels):
    evidence, scores = interpreter.run(
        image=image,
        patches_coarse=patches_coarse,
        patches_fine=patches_fine,
        superpixels=superpixels,
        artifact_prompts=prompts,
    )
    print(evidence)
