<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_segment_anything_3</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_segment_anything_3">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_segment_anything_3">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_segment_anything_3/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_segment_anything_3.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference for the Segment Anything Model 3 (SAM3). SAM3 predicts instance masks that indicate the desired object given geometric prompts (points and/or boxes) or **text prompts**. With its promptable segmentation capability, SAM3 delivers unmatched versatility for various image analysis tasks.

![SAM3 segmentation example](https://raw.githubusercontent.com/Ikomia-hub/infer_segment_anything_3/main/images/output.jpg)

## :key: Authentication Setup

**Important**: SAM 3 requires authentication to download model checkpoints from Hugging Face. Before using SAM 3, you must:

### 1. Request Access
Request access to the SAM 3 checkpoints on the Hugging Face repository:
- Visit: https://huggingface.co/facebook/sam3
- Click "Request access" and wait for approval

### 2. Generate Access Token
Once your access is approved, generate a Hugging Face access token:
- Go to: https://huggingface.co/settings/tokens
- Create a new token with **read** permissions (or **write** if you plan to upload models)
- Copy the token

### 3. Authenticate
Authenticate using one of the following methods:

**Option A: Command Line (Recommended)**
```bash
pip install --upgrade huggingface_hub
hf auth login
```
When prompted, paste your access token.

**Option B: Environment Variable**
```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"

# Linux/Mac
export HF_TOKEN="your_token_here"
```

**Option C: Python Script**
```python
from huggingface_hub import login
login(token="your_token_here")
```

For more details, see the [Hugging Face authentication guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication).

> **Note**: The model weights will be automatically downloaded on first use. If authentication fails, you'll receive clear error messages with instructions.

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow
```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_segment_anything_3", auto_connect=True)

# Set point prompt
algo.set_parameters({
    "input_point": "[[520, 375]]",
    "input_point_label": "[1]"
})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/truck.jpg")

# Inspect your result
display(algo.get_image_with_mask())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

### General parameters
- `cuda` `[bool]`: If True, CUDA-based inference (GPU). If False, run on CPU.
- `input_size_percent` `[int]`: Percentage size of the input image. Can be reduced to save memory usage.
- `confidence_threshold` `[float]`: Confidence threshold for predictions (0.0 to 1.0). Default: 0.5.

### Text predictor parameters
- `input_text` `[str]`: Text prompt describing the object to segment (e.g., "shoe", "car", "person"). This enables text-based segmentation without needing geometric prompts.

### Geometric prompt predictor parameters
- `input_box` `[list]`: A Nx4 array of given box prompts to the model, in [[XYXY]] or [[XYXY], [XYXY]] format.
- `input_point` `[list]`: A Nx2 array of point prompts to the model. Each point is in [[X,Y]] or [[XY], [XY]] in pixels.
- `input_point_label` `[list]`: A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
- `multimask_output` `[bool]`: If true, the model will return three masks. For ambiguous prompts such as a single point, it is recommended to use `multimask_output=True`. The best single mask can be chosen by picking the one with the highest score.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_segment_anything_3", auto_connect=True)

# Example: Setting parameters for prompt prediction
algo.set_parameters({
    "input_text": "shoe",
    "confidence_threshold": "0.3"
})

# Run directly on your image
wf.run_on(url="https://github.com/facebookresearch/sam3/blob/main/assets/images/test_image.jpg?raw=true")

# Inspect your result
img_output = algo.get_output(0)
mask_output = algo.get_output(1)
display(img_output.get_image_with_mask(mask_output), title="Best Mask")
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_segment_anything_3", auto_connect=True)

# Set prompts
algo.set_parameters({
    "input_point": "[[520, 375]]",
    "input_point_label": "[1]"
})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/truck.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

### 1. Text prompt (NEW in SAM3)
SAM3 introduces text-based prompting - simply describe what you want to segment!

```python
algo.set_parameters({
    "input_text": "shoe",
    "confidence_threshold": "0.5"
})
```

This will segment all objects matching the text description in the image.

### 2. Single point prompt
SAM3 with `"multimask_output":"True"` generates three output masks given a single point (3 best scores). This is useful for ambiguous prompts where the intended object is unclear.

```python
algo.set_parameters({
    "input_point": "[[520, 375]]",
    "input_point_label": "[1]",
    "multimask_output": "True"
})
```

### 3. Multiple points
A single point can be ambiguous. Using multiple points can improve the quality of the expected mask. Use label 1 for foreground points and 0 for background points.

```python
algo.set_parameters({
    "input_point": "[[500, 375], [1125, 625]]",
    "input_point_label": "[1, 0]",  # First point: foreground, Second point: background
    "multimask_output": "False"
})
```

### 4. Box prompt
Drawing a box over the desired object usually outputs a mask closer to expectation compared to point(s).

```python
algo.set_parameters({
    "input_box": "[425, 600, 700, 875]",  # xyxy format
    "multimask_output": "False"
})
```

SAM3 can also take multiple box prompts for batch inference:
```python
algo.set_parameters({
    "input_box": "[[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800]]",
    "multimask_output": "False"
})
```

### 5. Point and box combined
Point and box can be combined by including both types of prompts. This can be used to refine the selection, for example selecting just the truck's tire instead of the entire wheel.

```python
algo.set_parameters({
    "input_box": "[425, 600, 700, 875]",
    "input_point": "[[575, 750]]",
    "input_point_label": "[0]",  # Background point to exclude
    "multimask_output": "False"
})
```

### Using Ikomia STUDIO
- **Text prompt**: Enter a text description in the 'Text prompt' field (e.g., "shoe", "car")
- **Point**: Select the point tool and click on the object
- **Box**: Select the Square/Rectangle tool and draw around the object
- **Coordinate prompts**: Use the parameter fields in the settings panel
    - Text: 'Text prompt' e.g., "shoe"
    - Point: 'Point coord. [[xy]]' e.g., [[520, 375]]
    - Point label: 'Point label [i]' e.g., [1] for foreground, [0] for background
    - Box: 'Box coord. [[xyxy]]' e.g., [[425, 600, 700, 875]]
