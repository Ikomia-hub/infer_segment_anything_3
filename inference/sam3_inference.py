"""
SAM3 inference functions for text and geometric prompts.
"""
import json
import numpy as np
from PIL import Image

from ikomia import core


def _process_multimask_output(masks, scores, multimask_output):
    """
    Process masks based on multimask_output setting.

    Args:
        masks: Predicted masks (numpy array)
        scores: Mask scores (numpy array)
        multimask_output: Boolean indicating whether to return multiple masks

    Returns:
        List of processed masks
    """
    # Sort by score
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    # Handle multimask_output
    masks = np.squeeze(masks)
    if masks.ndim == 3:
        # Multiple masks per prompt
        if multimask_output:
            # Return all masks as separate outputs
            masks = [masks[i] for i in range(masks.shape[0])]
        else:
            # Return the best mask
            masks = [masks[0]]
    elif masks.ndim == 2:
        # Single mask
        masks = [masks]
    else:
        masks = [masks[0]]

    return masks


def infer_text_predictor(processor, src_image, param):
    """
    Run SAM3 prediction using text prompt.

    Args:
        processor: Sam3Processor instance
        src_image: Source image (numpy array or PIL Image)
        param: Algorithm parameters

    Returns:
        List of predicted masks
    """
    # Convert numpy array to PIL Image for SAM3
    if isinstance(src_image, np.ndarray):
        pil_image = Image.fromarray(src_image)
    else:
        pil_image = src_image

    # Update confidence threshold if needed
    processor.confidence_threshold = param.confidence_threshold

    # Set the image and get inference state
    state = processor.set_image(pil_image)

    # Reset any previous prompts and set text prompt
    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=param.input_text)

    # Get masks from state dictionary
    # The text predictor stores results as state["masks"], state["scores"], state["boxes"]
    if "masks" in state and state["masks"] is not None and len(state["masks"]) > 0:
        # Boolean tensor of shape (N, 1, H, W)
        masks_tensor = state["masks"]
        scores = state["scores"]  # Tensor of scores
        num_objects = len(scores)
        print(
            f"Found {num_objects} object(s) for text prompt: '{param.input_text}'")

        # Convert to numpy
        masks_np = masks_tensor.squeeze(1).cpu().numpy()  # (N, H, W)

        if masks_np.ndim == 3 and masks_np.shape[0] > 0:
            # Combine multiple masks into single labeled mask
            mask_output = np.zeros(masks_np.shape[1:], dtype=np.uint8)
            for i, mask in enumerate(masks_np):
                mask_output[mask] = i + 1
            masks = [mask_output]
        elif masks_np.ndim == 2:
            masks = [masks_np.astype(np.uint8)]
        else:
            masks = [
                np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.uint8)]
    else:
        print(f"No objects found for text prompt: '{param.input_text}'")
        masks = [
            np.zeros((pil_image.size[1], pil_image.size[0]), dtype=np.uint8)]

    return masks


def infer_geometric_predictor(model, processor, graph_input, src_image, resizing, param):
    """
    Run SAM3 prediction using point and/or box prompts.

    Args:
        model: SAM3 model instance
        processor: Sam3Processor instance
        graph_input: Graphics input from Ikomia Studio
        src_image: Source image (numpy array or PIL Image)
        resizing: Resize ratio applied to the image
        param: Algorithm parameters

    Returns:
        List of predicted masks
    """
    input_box = None
    input_label = None
    input_point = None

    # Parse prompts from parameters or graphics input
    if param.input_box or param.input_point:
        # Get input from coordinate prompt param - STUDIO/API
        if param.input_box:
            box_list = json.loads(param.input_box)
            input_box = np.array(box_list)
            input_box = input_box * resizing

        if param.input_point:
            point = json.loads(param.input_point)
            input_point = np.array(point)
            input_point = input_point * resizing

            if param.input_point_label:
                label_id = json.loads(param.input_point_label)
                input_label = np.array(label_id)
            else:
                # Default to foreground points
                input_label = np.ones(len(input_point), dtype=np.int32)
    else:
        # Get input from drawn graphics - STUDIO
        graphics = graph_input.get_items()
        box = []
        point = []
        labels = []

        # First pass: collect all boxes and points
        for i, graphic in enumerate(graphics):
            bboxes = graphics[i].get_bounding_rect()
            if graphic.get_type() == core.GraphicsItem.RECTANGLE:
                x1 = bboxes[0] * resizing
                y1 = bboxes[1] * resizing
                x2 = (bboxes[2] + bboxes[0]) * resizing
                y2 = (bboxes[3] + bboxes[1]) * resizing
                box.append([x1, y1, x2, y2])

            if graphic.get_type() == core.GraphicsItem.POINT:
                x1 = bboxes[0] * resizing
                y1 = bboxes[1] * resizing
                point.append([x1, y1])

        # Set labels: if there's a box, points are background (0), otherwise foreground (1)
        if point:
            # If both box and point exist, points are background (0), otherwise foreground (1)
            point_label = 0 if box else 1
            labels = [point_label] * len(point)

        if box:
            input_box = np.array(box)
        if point:
            input_point = np.array(point)
            input_label = np.array(labels)

    # Convert numpy array to PIL Image for SAM3
    if isinstance(src_image, np.ndarray):
        pil_image = Image.fromarray(src_image)
    else:
        pil_image = src_image

    # Set the image and get inference state
    inference_state = processor.set_image(pil_image)

    # Run prediction based on available prompts
    if input_box is not None and len(input_box) > 1:
        # Multiple boxes - batch inference
        if input_point is not None:
            print('Point input(s) not used with multiple boxes')

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=param.multimask_output,
        )

        # Process masks from multiple boxes
        masks = np.squeeze(masks)
        if masks.ndim == 3:
            # Single box with multiple masks
            if param.multimask_output:
                # Return all masks as separate outputs
                masks = [masks[i] for i in range(masks.shape[0])]
            else:
                # Return only the best mask
                masks = [masks[0]]
        elif masks.ndim == 4:
            # Multiple boxes, shape: [num_multimasks, num_boxes, H, W]
            mask_outputs = []
            for j in range(masks.shape[1]):  # For each box
                if param.multimask_output:
                    # Return all multimasks for this box as separate outputs
                    for i in range(masks.shape[0]):
                        mask_outputs.append(masks[i, j, :, :])
                else:
                    # Return only the best mask for this box
                    mask_outputs.append(masks[0, j, :, :])
            masks = mask_outputs
        else:
            raise ValueError("Unexpected mask dimensions")

    elif input_point is not None and input_box is None:
        # Points only
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=param.multimask_output,
        )

        masks = _process_multimask_output(
            masks, scores, param.multimask_output)

    elif input_point is None and input_box is not None and len(input_box) == 1:
        # Single box
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=param.multimask_output,
        )

        masks = _process_multimask_output(
            masks, scores, param.multimask_output)

    elif input_point is not None and input_box is not None and len(input_box) == 1:
        # Single box with point(s) - use point as background to refine
        # In SAM3, combining box and points is supported
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[0] if input_box.ndim == 2 else input_box,
            multimask_output=param.multimask_output,
        )

        masks = _process_multimask_output(
            masks, scores, param.multimask_output)

    else:
        masks = [np.zeros(src_image.shape[:2], dtype=np.uint8)]
        print("Please select a point and/or a box as prompt")

    return masks
