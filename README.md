This project is a fork of [bedovyy/ComfyUI_NAIDGenerator](https://github.com/bedovyy/ComfyUI_NAIDGenerator), licensed under GPL-3.0.

# ComfyUI_NAIDGenerator

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) extension for generating image via NovelAI API.

## Installation

- `git clone https://github.com/micode64/ComfyUI_NAIDGenerator` into the `custom_nodes` directory.
- or 'Install via Git URL' from [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Setting up NAI account

Before using the nodes, you should set NAI_ACCESS_TOKEN on `ComfyUI/.env` file.

```
NAI_ACCESS_TOKEN=<ACCESS_TOKEN>
```

You can get persistent API token by **User Settings > Account > Get Persistent API Token** on NovelAI webpage.

Otherwise, you can get access token which is valid for 30 days using [novelai-api](https://github.com/Aedial/novelai-api).

## Usage

The nodes are located at `NovelAI` category.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8ab1ecc0-2ba8-4e38-8810-727e50a20923)

### Txt2img

Simply connect `GenerateNAID` node and `SaveImage` node.

![generate](https://github.com/micode64/ComfyUI_NAIDGenerator/assets/GenerateNAID.png)

Note that all generated images via `GeneratedNAID` node are saved as `output/NAI_autosave_12345_.png` for keeping original metadata.

### Img2img

Connect `Img2ImgOptionNAID` node to `GenerateNAID` node and put original image.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/15ff8961-4f6b-4f23-86bf-34b86ace45c0)

Note that width and height of the source image will be resized to generation size.

### Inpainting

Connect `InpaintingOptionNAID` node to `GenerateNAID` node and put original image and mask image.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/5ed1ad77-b90e-46be-8c37-9a5ee0935a3d)

Note that both source image and mask will be resized fit to generation size.

(You don't need `MaskImageToNAID` node to convert mask image to NAID mask image.)

### Vibe Transfer

Connect `VibeTransferOptionNAID` node to `GenerateNAID` node and put reference image.

![Comfy_workflow](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/8c6c1c2e-f29d-42a1-b615-439155cb3164)

You can also relay Img2ImgOption on it.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/acf0496c-8c7c-48f4-9530-18e6a23669d5)

Note that width and height of the source images will be resized to generation size. **This will change aspect ratio of source images.**

#### Multiple Vibe Transfer

Just connect multiple `VibeTransferOptionNAID` nodes to `GenerateNAID` node.

![preview_vibe_2](https://github.com/user-attachments/assets/2d56c0f7-bcd5-48ff-b436-012ea43604fe)

### ModelOption

The default model of `GenerateNAID` node is `nai-diffusion-3`(NAI Diffusion Anime V3).

If you want to change model, put `ModelOptionNAID` node to `GenerateNAID` node.

![ModelOption](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/0b484edb-bcb5-428a-b2af-1372a9d7a34f)

### NetworkOption

You can set timeout or retry option from `NetworkOption` node.
Moreover, you can ignore error by `ignore_errors`. In that case, the result will be 1x1 size grayscale image.
Without this node, the request never retry and wait response forever, and stop the queue when error occurs

![preview_network](https://github.com/user-attachments/assets/d82b0ff2-c57c-4870-9024-8d78261a8fea)

**Note that if you set timeout too short, you may not get image but spend Anlas.**

### PromptToNAID

ComfyUI use `()` or `(word:weight)` for emphasis, but NovelAI use `{}` and `[]`. This node convert ComfyUI's prompt to NovelAI's.

Optionally, you can choose weight per brace. If you set `weight_per_brace` to 0.10, `(word:1.1)` will convert to `{word}` instead of `{{word}}`.

![image](https://github.com/bedovyy/ComfyUI_NAIDGenerator/assets/137917911/25c48350-7268-4d6f-81fe-9eb080fc6e5a)

### Director Tools

![image](https://github.com/user-attachments/assets/e205a51e-59dc-4d5a-94c8-29715ed98739)

You can find director tools like `LineArtNAID` or `EmotionNAID` on NovelAI > director_tools.

![augment_example](https://github.com/user-attachments/assets/5833e9fb-f92e-4d53-9069-58ca8503a3e7)

#### New Model Option

NAI Diffusion V4 Full is now available in the ModelOptionNAID node:

```python
model = "nai-diffusion-4-full"
```

#### CharacterNAI Node
- **Purpose**: Defines a single character prompt with explicit position and negative prompt.
- **Inputs:**
  - `positive_prompt` (string, multiline): Character's positive prompt.
  - `negative_prompt` (string, multiline, optional): Character's negative prompt.
  - `x` (A–E): Column selection on a 5x5 grid (default: C).
  - `y` (1–5): Row selection on a 5x5 grid (default: 3).
- **How it works:**
  - The (x, y) selection is mapped to normalized float coordinates (0.0–1.0) per NovelAI API spec.
  - The node outputs a dictionary with `char_caption`, `negative_caption`, and `centers` (list of {x, y} dicts).
- **UI:**
  - The grid UI allows intuitive placement. The selected cell is highlighted for clarity.
  ![image](https://github.com/micode64/ComfyUI_NAIDGenerator/assets/CharacterNAI.png)

#### CharacterConcatenateNAI Node
- **Purpose**: Combines up to 6 CharacterNAI nodes into a single character list.
- **Inputs:**
  - `character1` (required), `character2`–`character6` (optional): CharacterNAI nodes.
- **How it works:**
  - All provided characters are merged into a list, omitting any empty slots.
  - Output is a `CHARACTER_LIST_NAI` type, suitable for direct connection to the `characters` input of GenerateNAID.
  ![image](https://github.com/micode64/ComfyUI_NAIDGenerator/assets/CharacterConcatenateNAI.png)

#### Connecting to GenerateNAID
- The `characters` slot of the `GenerateNAID` node accepts a single CharacterNAI or a CharacterConcatenateNAI node.
- Internally, all character prompts are assembled into the `v4_prompt.caption.char_captions` structure as required by NovelAI V4 API.
- Each character entry includes its prompt, negative prompt, and position.

#### Technical Details
- Character positions are always normalized floats (0.0–1.0) for both x and y.
- The prompt merging logic ensures that character prompts remain separate from the base prompt, following the metadata.yaml structure.
- The UI grid is implemented in the web extension for intuitive character positioning.

#### Example Workflow
1. Create several CharacterNAI nodes with desired prompts and positions.
2. Combine them using CharacterConcatenateNAI if needed.
3. Connect the resulting character list to the `characters` input of GenerateNAID.

See the [examples](#) for more details.

**Note:**
- Up to 6 characters can be merged.
- If no character is provided, the workflow behaves as a standard single-prompt generation.
- For advanced users: you may inspect the internal structure passed to the API for debugging or extension.


Note: Basic img2img functionality works with V4 preview. For inpainting, the node will automatically use V3 model but can still work on V4-generated images. Vibe transfer will be supported once V4 fully releases.
