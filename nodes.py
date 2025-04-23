import copy
import io
from pathlib import Path
import folder_paths
import zipfile

from .utils import *


TOOLTIP_LIMIT_OPUS_FREE = "Limit image size and steps for free generation by Opus."

class PromptToNAID:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "text": ("STRING", { "forceInput":True, "multiline": True, "dynamicPrompts": False,}),
            "weight_per_brace": ("FLOAT", { "default": 0.05, "min": 0.05, "max": 0.10, "step": 0.05 }),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, text, weight_per_brace):
        nai_prompt = prompt_to_nai(text, weight_per_brace)
        return (nai_prompt,)

class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",) } }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, image):
        s = resize_to_naimask(image)
        return (s,)

class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["nai-diffusion-2", "nai-diffusion-furry-3", "nai-diffusion-3", "nai-diffusion-4-full", ], { "default": "nai-diffusion-4-full" }), #"nai-diffusion-4-curated" is not available
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, model, option=None):
        option = copy.deepcopy(option) if option else {}
        option["model"] = model
        return (option,)

class Img2ImgOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", { "default": 0.70, "min": 0.01, "max": 0.99, "step": 0.01, "display": "number" }),
                "noise": ("FLOAT", { "default": 0.00, "min": 0.00, "max": 0.99, "step": 0.02, "display": "number" }),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, strength, noise):
        option = {}
        option["img2img"] = (image, strength, noise)
        return (option,)

class InpaintingOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "add_original_image": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, mask, add_original_image):
        option = {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)

class VibeTransferOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "information_extracted": ("FLOAT", { "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
                "strength": ("FLOAT", { "default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, information_extracted, strength, option=None):
        option = copy.deepcopy(option) if option else {}
        if "vibe" not in option:
            option["vibe"] = []

        option["vibe"].append((image, information_extracted, strength))
        return (option,)

class NetworkOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ignore_errors": ("BOOLEAN", { "default": True }),
                "timeout_sec": ("INT", { "default": 120, "min": 30, "max": 3000, "step": 1, "display": "number" }),
                "retry": ("INT", { "default": 3, "min": 1, "max": 100, "step": 1, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, ignore_errors, timeout_sec, retry, option=None):
        option = copy.deepcopy(option) if option else {}
        option["ignore_errors"] = ignore_errors
        option["timeout_sec"] = timeout_sec
        option["retry"] = retry
        return (option,)


class GenerateNAID:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit_opus_free": ("BOOLEAN", {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE}),
                "width": ("INT", {"default": 832, "min": 64, "max": 1600, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 1216, "min": 64, "max": 1600, "step": 64, "display": "number"}),
                "positive": ("STRING", {"default": ", best quality, amazing quality, very aesthetic, absurdres", "multiline": True, "dynamicPrompts": False}),
                "negative": ("STRING", {"default": "lowres", "multiline": True, "dynamicPrompts": False}),
                "use_coords": ("BOOLEAN", {"default": True, "label": "Manual_positioning", "tooltip": "Manually specify character positions"}),
                "use_order": ("BOOLEAN", {"default": True, "label": "Keep_order", "tooltip": "Preserve character order"}),
                "legacy_uc": ("BOOLEAN", {"default": False, "label": "Legacy_UC", "tooltip": "Use legacy UC logic"}),
                "steps": ("INT", {"default": 28, "min": 0, "max": 50, "step": 1, "display": "number"}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number"}),
                "variety": ("BOOLEAN", {"default": False}),
                "decrisper": ("BOOLEAN", {"default": False}),
                "smea": (["none", "SMEA", "SMEA+DYN"], {"default": "none"}),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"], {"default": "k_euler"}),
                "scheduler": (["native", "karras", "exponential", "polyexponential"], {"default": "native"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number"}),
                "uncond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "display": "number"}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02, "display": "number"}),
                "keep_alpha": ("BOOLEAN", {"default": True, "tooltip": "Disable to further process output images locally"}),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    def generate(self, limit_opus_free, width, height, positive, negative, steps, cfg, decrisper, variety, smea, sampler, scheduler, seed, uncond_scale, cfg_rescale, keep_alpha, use_coords, use_order, legacy_uc, option=None):
        width, height = calculate_resolution(width*height, (width, height))

        # ref. novelai_api.ImagePreset
        params = {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": cfg,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": 1,
            "ucPreset": 3,
            "qualityToggle": False,
            "sm": (smea == "SMEA" or smea == "SMEA+DYN") and sampler != "ddim",
            "sm_dyn": smea == "SMEA+DYN" and sampler != "ddim",
            "dynamic_thresholding": decrisper,
            "skip_cfg_above_sigma": None,
            "controlnet_strength": 1.0,
            "add_original_image": False,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
            "legacy_v3_extend": False,
            "uncond_scale": uncond_scale,
            "negative_prompt": negative,
            "prompt": positive,
            "reference_image_multiple": [],
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": [],
            "extra_noise_seed": seed,
            "v4_prompt": {
                "use_coords": use_coords,
                "use_order": use_order,
                "legacy_uc": legacy_uc,
                "caption": {
                    "base_caption": positive,
                    "char_captions": []
                }
            },
            "v4_negative_prompt": {
                "use_coords": False,
                "use_order": False,
                "caption": {
                    "base_caption": negative,
                    "char_captions": []
                }
            }
        }
        model = "nai-diffusion-4-full"
        action = "generate"

        if option:
            if "img2img" in option:
                action = "img2img"
                image, strength, noise = option["img2img"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["strength"] = strength
                params["noise"] = noise
            elif "infill" in option:
                action = "infill"
                image, mask, add_original_image = option["infill"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["mask"] = naimask_to_base64(resize_to_naimask(mask, (width, height), "4" in model))
                params["add_original_image"] = add_original_image

            if "vibe" in option:
                for vibe in option["vibe"]:
                    image, information_extracted, strength = vibe
                    params["reference_image_multiple"].append(image_to_base64(resize_image(image, (width, height))))
                    params["reference_information_extracted_multiple"].append(information_extracted)
                    params["reference_strength_multiple"].append(strength)

            if "model" in option:
                model = option["model"]

            # Handle V4 options
            if "v4_prompt" in option:
                params["v4_prompt"].update(option["v4_prompt"])

            if "v4_negative_prompt" in option:
                params["v4_negative_prompt"].update(option["v4_negative_prompt"])

        timeout = option["timeout"] if option and "timeout" in option else None
        retry = option["retry"] if option and "retry" in option else None

        if limit_opus_free:
            pixel_limit = 1024*1024 if model in ("nai-diffusion-2", "nai-diffusion-furry-3", "nai-diffusion-3", "nai-diffusion-4", "nai-diffusion-4-curated-preview", "nai-diffusion-4-full") else 640*640
            if width * height > pixel_limit:
                max_width, max_height = calculate_resolution(pixel_limit, (width, height))
                params["width"] = max_width
                params["height"] = max_height
            if steps > 28:
                params["steps"] = 28

        if variety:
            params["skip_cfg_above_sigma"] = calculate_skip_cfg_above_sigma(params["width"], params["height"])

        if sampler == "ddim" and model in ("nai-diffusion-furry-3", "nai-diffusion-3", "nai-diffusion-4-curated-preview", "nai-diffusion-4-full"):
            params["sampler"] = "ddim_v3"

        if action == "infill" and model != "nai-diffusion-2":
            model = f"{model}-inpainting"

        image = blank_image()
        try:
            zipped_bytes = generate_image(self.access_token, positive, model, action, params, timeout, retry)
            zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
            image_bytes = zipped.read(zipped.infolist()[0]) # only support one n_samples

            ## save original png to comfy output dir
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
            file = f"{filename}_{counter:05}_.png"
            d = Path(full_output_folder)
            d.mkdir(exist_ok=True)
            (d / file).write_bytes(image_bytes)

            image = bytes_to_image(image_bytes, keep_alpha)
        except Exception as e:
            if "ignore_errors" in option and option["ignore_errors"]:
                print("ignore error:", e)
            else:
                raise e

        return (image,)


def base_augment(access_token, output_dir, limit_opus_free, ignore_errors, req_type, image, options=None):
    image = image.movedim(-1, 1)
    w, h = (image.shape[3], image.shape[2])
    image = image.movedim(1, -1)

    if limit_opus_free:
        pixel_limit = 1024 * 1024
        if w * h > pixel_limit:
            w, h = calculate_resolution(pixel_limit, (w, h))
    base64_image = image_to_base64(resize_image(image, (w, h)))
    result_image = blank_image()
    try:
        # Build request based on NAI v4 API spec
        request = {
            "image": base64_image,
            "req_type": req_type,
            "width": w,
            "height": h
        }

        # Add optional parameters if provided
        if options:
            if "defry" in options:
                request["defry"] = options["defry"]
            if "prompt" in options:
                request["prompt"] = options["prompt"]

        zipped_bytes = augment_image(access_token, req_type, w, h, base64_image, options=options)
        zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
        image_bytes = zipped.read(zipped.infolist()[0]) # only support one n_samples

        ## save original png to comfy output dir
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("NAI_autosave", output_dir)
        file = f"{filename}_{counter:05}_.png"
        d = Path(full_output_folder)
        d.mkdir(exist_ok=True)
        (d / file).write_bytes(image_bytes)

        result_image = bytes_to_image(image_bytes)
    except Exception as e:
        if ignore_errors:
            print("ignore error:", e)
        else:
            raise e

    return (result_image,)

class RemoveBGAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "bg-removal", image)

class LineArtAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "lineart", image)

class SketchAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "sketch", image)

class ColorizeAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
                "defry": ("INT", { "default": 0, "min": 0, "max": 5, "step": 1, "display": "number" }),
                "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, defry, prompt):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "colorize", image, options={ "defry": defry, "prompt": prompt })

class EmotionAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    strength_list = ["normal", "slightly_weak", "weak", "even_weaker", "very_weak", "weakest"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
                "mood": (["neutral", "happy", "sad", "angry", "scared",
                     "surprised", "tired", "excited", "nervous", "thinking",
                     "confused", "shy", "disgusted", "smug", "bored",
                     "laughing", "irritated", "aroused", "embarrassed", "worried",
                     "love", "determined", "hurt", "playful"], { "default": "neutral" }),
                "strength": (s.strength_list, { "default": "normal" }),
                "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, mood, strength, prompt):
        prompt = f"{mood};;{prompt}"
        defry = EmotionAugment.strength_list.index(strength)
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "emotion", image, options={ "defry": defry, "prompt": prompt })

class DeclutterAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }),
                "ignore_errors": ("BOOLEAN", { "default": False }),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "declutter", image)

class V4PromptConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_coords": ("BOOLEAN", { "default": False }),
                "use_order": ("BOOLEAN", { "default": False }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI/v4"
    def set_option(self, use_coords, use_order, option=None):
        option = copy.deepcopy(option) if option else {}
        if "v4_prompt" not in option:
            option["v4_prompt"] = {}
        option["v4_prompt"]["use_coords"] = use_coords
        option["v4_prompt"]["use_order"] = use_order
        return (option,)

class CharacterNAI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "x": (["A", "B", "C", "D", "E"], {"default": "C"}),
                "y": (["1", "2", "3", "4", "5"], {"default": "3"}),
            }
        }

    RETURN_TYPES = ("CHARACTER_NAI",)
    FUNCTION = "create"
    CATEGORY = "NovelAI/v4"

    def create(self, positive_prompt, x, y, negative_prompt):
        # Convert x ('A'-'E') and y (1-5) to normalized float (0.0~1.0)
        x_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        x_norm = x_map.get(x, 2) / 4.0
        y_norm = (int(y) - 1) / 4.0
        return ({
            "char_caption": positive_prompt,
            "centers": [{"x": x_norm, "y": y_norm}],
            "negative_caption": negative_prompt,
        },)

class CharacterConcatenateNAI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character1": ("CHARACTER_NAI", {"forceInput": True}),
            },
            "optional": {
                "character2": ("CHARACTER_NAI", {"forceInput": False}),
                "character3": ("CHARACTER_NAI", {"forceInput": False}),
                "character4": ("CHARACTER_NAI", {"forceInput": False}),
                "character5": ("CHARACTER_NAI", {"forceInput": False}),
                "character6": ("CHARACTER_NAI", {"forceInput": False}),
            }
        }

    RETURN_TYPES = ("CHARACTER_LIST_NAI",)
    FUNCTION = "concat"
    CATEGORY = "NovelAI/v4"

    def concat(self, character1, character2=None, character3=None, character4=None, character5=None, character6=None):
        # Convert inputs to list and exclude "None"
        characters = [character1, character2, character3, character4, character5, character6]
        characters = [c for c in characters if c is not None]
        return (characters,)

# GenerateNAID node extension: character prompt slots added
# Add 'characters' to INPUT_TYPES of existing GenerateNAID and internally assemble char_captions in a metadata.yaml structure.
# Replace existing GenerateNAID.INPUT_TYPES

old_generate_naid_input_types = GenerateNAID.INPUT_TYPES

def new_generate_naid_input_types(s):
    types = old_generate_naid_input_types()
    types["required"]["characters"] = ("CHARACTER_LIST_NAI", {"default": [], "forceInput": False, "multiline": True})
    return types
GenerateNAID.INPUT_TYPES = classmethod(new_generate_naid_input_types)

# Add 'characters' to INPUT_TYPES of existing GenerateNAID and internally assemble char_captions in a metadata.yaml structure.

old_generate_naid_generate = GenerateNAID.generate

def new_generate_naid_generate(self, *args, **kwargs):
    # characters is always the last argument
    import inspect
    sig = inspect.signature(old_generate_naid_generate)
    params = list(sig.parameters.keys())
    # Existing arguments + characters + option
    option = kwargs.get('option', None)
    characters = kwargs.get('characters', None)
    if characters is None and len(args) >= len(params):
        characters = args[len(params)-1]
    # Build v4_prompt
    if option is None:
        option = {}
    if characters:
        # Same structure as metadata.yaml
        char_captions = []
        for idx, c in enumerate(characters):
            centers = c.get("centers", [])
            # Warn if centers is empty or None
            if not centers or centers is None or not isinstance(centers, list) or not centers:
                print(f"[WARN] Character index {idx} centers is empty or invalid: {centers}")
            else:
                # If centers is not in [{"x":..., "y":...}] format, fix it
                if isinstance(centers, dict):
                    centers = [centers]
                elif isinstance(centers, list):
                    # If it's a list but not a list of dicts, fix it
                    if centers and not isinstance(centers[0], dict):
                        print(f"[WARN] Unexpected format for centers: {centers}")
                        centers = [{"x": centers[0], "y": centers[1]}] if len(centers) == 2 else []
            char_captions.append({
                "char_caption": c.get("char_caption", ""),
                "centers": centers
            })
        if char_captions:
            if "v4_prompt" not in option:
                option["v4_prompt"] = {"caption": {}}
            if "caption" not in option["v4_prompt"]:
                option["v4_prompt"]["caption"] = {}
            option["v4_prompt"]["caption"]["char_captions"] = char_captions
    # Print option contents before sending to API
    print("[DEBUG] API送信前 option:", option)
    kwargs['option'] = option
    # Remove characters from kwargs
    if 'characters' in kwargs:
        del kwargs['characters']
    return old_generate_naid_generate(self, *args[:len(params)-1], **kwargs)

GenerateNAID.generate = new_generate_naid_generate

NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "VibeTransferOptionNAID": VibeTransferOption,
    "NetworkOptionNAID": NetworkOption,
    "MaskImageToNAID": ImageToNAIMask,
    "PromptToNAID": PromptToNAID,
    "RemoveBGNAID": RemoveBGAugment,
    "LineArtNAID": LineArtAugment,
    "SketchNAID": SketchAugment,
    "ColorizeNAID": ColorizeAugment,
    "EmotionNAID": EmotionAugment,
    "DeclutterNAID": DeclutterAugment,
    "V4PromptConfig": V4PromptConfig,
    "CharacterNAI": CharacterNAI,
    "CharacterConcatenateNAI": CharacterConcatenateNAI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate NAI",
    "ModelOptionNAID": "ModelOption NAI",
    "Img2ImgOptionNAID": "Img2ImgOption NAI",
    "InpaintingOptionNAID": "InpaintingOption NAI",
    "VibeTransferOptionNAID": "VibeTransferOption NAI",
    "NetworkOptionNAID": "NetworkOption NAI",
    "MaskImageToNAID": "Convert Mask Image NAI",
    "PromptToNAID": "Convert Prompt NAI",
    "RemoveBGNAID": "Remove BG NAI",
    "LineArtNAID": "LineArt NAI",
    "SketchNAID": "Sketch NAI",
    "ColorizeNAID": "Colorize NAI",
    "EmotionNAID": "Emotion NAI",
    "DeclutterNAID": "Declutter NAI",
    "V4PromptConfig": "V4 Prompt Config NAI",
    "CharacterNAI": "Character NAI",
    "CharacterConcatenateNAI": "CharacterConcatenate NAI",
}
