"""
Module that implements the core logic of algorithm execution.
Implements SAM3 (Segment Anything Model 3) for interactive instance segmentation.
"""
import copy
import os

import torch
import numpy as np
import cv2

from ikomia import core, dataprocess, utils

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from infer_segment_anything_3.utils.utils_ik import check_float16_and_bfloat16_support, resize_mask, get_checkpoint_path, fix_cuda_caches_and_buffers
from infer_segment_anything_3.inference import infer_text_predictor, infer_geometric_predictor


class InferSegmentAnything3Param(core.CWorkflowTaskParam):
    """
    Class to handle the algorithm parameters.
    Inherits PyCore.CWorkflowTaskParam from Ikomia API.
    """

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.input_size_percent = 100
        self.input_point = ''
        self.input_box = ''
        self.input_point_label = ''
        self.input_text = ''  # Text prompt for text-based segmentation
        self.confidence_threshold = 0.5  # Confidence threshold for predictions
        self.cuda = torch.cuda.is_available()
        self.multimask_output = False  # SAM3 recommends True for ambiguous prompts
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.input_size_percent = int(param_map["input_size_percent"])
        self.input_point = param_map['input_point']
        self.input_point_label = param_map['input_point_label']
        self.input_box = param_map['input_box']
        self.input_text = param_map['input_text']
        self.confidence_threshold = float(param_map["confidence_threshold"])
        self.multimask_output = utils.strtobool(param_map["multimask_output"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "input_size_percent": str(self.input_size_percent),
            "input_point": str(self.input_point),
            "input_point_label": str(self.input_point_label),
            "input_box": str(self.input_box),
            "input_text": str(self.input_text),
            "confidence_threshold": str(self.confidence_threshold),
            "multimask_output": str(self.multimask_output),
            "cuda": str(self.cuda)
        }
        return param_map


class InferSegmentAnything3ParamFactory(dataprocess.CTaskParamFactory):
    """Factory class to create parameters object."""

    def __init__(self):
        dataprocess.CTaskParamFactory.__init__(self)
        self.name = "infer_segment_anything_3"

    def create(self):
        """Instantiate parameters object."""
        return InferSegmentAnything3Param()


class InferSegmentAnything3(dataprocess.CSemanticSegmentationTask):
    """
    Class that implements SAM3 (Segment Anything Model 3) for interactive instance segmentation.
    Inherits PyCore.CWorkflowTask or derived from Ikomia API.
    """

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)

        # Create parameters object
        if param is None:
            self.set_param_object(InferSegmentAnything3Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Initialize to CPU, will be set correctly in _load_model
        self.device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.inference_state = None
        self.input_point = None
        self.input_label = np.array([1])  # foreground point
        self.input_box = None
        # Initialize to float32, will be set correctly in _load_model
        self.dtype = torch.float32
        self.base_dir = os.path.dirname(os.path.realpath(__file__))

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def get_progress_steps(self):
        """
        Ikomia Studio only.
        Function returning the number of progress steps for this algorithm.
        This is handled by the main progress bar of Ikomia Studio.
        """
        return 1

    def _load_model(self):
        """Load SAM3 model and create processor."""
        param = self.get_param_object()

        # Determine device FIRST based on param.cuda setting
        # Only check CUDA availability if param.cuda is True to avoid triggering CUDA init
        if param.cuda:
            try:
                # Only check CUDA if explicitly requested
                cuda_available = torch.cuda.is_available()
                self.device = torch.device("cuda") if cuda_available else torch.device("cpu")
            except RuntimeError:
                # CUDA not available or initialization failed
                self.device = torch.device("cpu")
                print("CUDA requested but not available - using CPU")
        else:
            # CUDA explicitly disabled, use CPU
            self.device = torch.device("cpu")

        # Check for float16 and bfloat16 support
        float16_support, bfloat16_support = check_float16_and_bfloat16_support(
            param.cuda)

        # Determine dtype based on GPU support
        self.dtype = torch.bfloat16 if bfloat16_support else torch.float16 if float16_support else torch.float32

        if self.device.type == "cuda":
            # Use bfloat16 for CUDA
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            # Print message when CUDA is disabled
            print("No GPU found - using CPU")

        # Load SAM3 model
        bpe_path = os.path.join(
            self.base_dir, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )

        # Get checkpoint path from weights directory
        checkpoint_path = get_checkpoint_path(self.base_dir)

        # Build model with local checkpoint path
        # Always build on CPU first to avoid device mismatch, then move to target device
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device="cpu",  # Always build on CPU first to ensure clean state
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            load_from_HF=False  # Don't download from HF, use local path
        )

        # Explicitly ensure model and all submodules are on the correct device
        # This is critical to avoid device mismatch errors
        # Use eval() to ensure model is in eval mode
        self.model.eval()

        # Clear any coordinate caches before moving to device (they'll be recreated on correct device)
        fix_cuda_caches_and_buffers(
            self.model, torch.device("cpu"), clear_cache=False)

        # Now move to target device
        self.model = self.model.to(self.device)

        # Force clear any cached device property to ensure it reflects the new device
        if hasattr(self.model, '_device'):
            self.model._device = None

        # Recursively clear CUDA caches and ensure all buffers/parameters are on correct device
        # This also clears coordinate caches that might have been created during model building
        fix_cuda_caches_and_buffers(self.model, self.device)

        # Create processor with confidence threshold and correct device
        self.processor = Sam3Processor(
            self.model, device=device_str, confidence_threshold=param.confidence_threshold)

        param.update = False

    def run(self):
        """Main function and entry point for algorithm execution."""
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Get input (numpy array)
        src_image = self.get_input(0).get_image()

        # Check the number of channels
        if src_image.shape[-1] == 4:  # RGBA?
            src_image = src_image[:, :, :3]  # Keep only RGB channels

        # Store original dimensions
        h_orig, w_orig = src_image.shape[0], src_image.shape[1]

        # Resize image if needed
        ratio = param.input_size_percent / 100
        if param.input_size_percent < 100:
            width = int(src_image.shape[1] * ratio)
            height = int(src_image.shape[0] * ratio)
            dim = (width, height)
            src_image = cv2.resize(
                src_image, dim, interpolation=cv2.INTER_LINEAR)

        # Load model if needed
        if self.model is None or param.update:
            self._load_model()

        # Check graphic input prompt
        graph_input = self.get_input(1)

        # Determine which inference mode to use
        has_geometric_prompt = (graph_input.is_data_available() or
                                param.input_box or param.input_point)
        has_text_prompt = bool(param.input_text)

        if has_text_prompt:
            # Run inference with text prompt
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    masks = infer_text_predictor(
                        processor=self.processor,
                        src_image=src_image,
                        param=param
                    )
            else:
                masks = infer_text_predictor(
                    processor=self.processor,
                    src_image=src_image,
                    param=param
                )
        elif has_geometric_prompt:
            # Run inference with geometric prompts (points/boxes)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    masks = infer_geometric_predictor(
                        model=self.model,
                        processor=self.processor,
                        graph_input=graph_input,
                        src_image=src_image,
                        resizing=ratio,
                        param=param
                    )
            else:
                masks = infer_geometric_predictor(
                    model=self.model,
                    processor=self.processor,
                    graph_input=graph_input,
                    src_image=src_image,
                    resizing=ratio,
                    param=param
                )
        else:
            # No prompts provided - return empty mask with message
            print(
                "SAM3 requires prompts. Please provide input_text, input_point, or input_box.")
            masks = [np.zeros((h_orig, w_orig), dtype=np.uint8)]

        # Clear extra outputs
        for i in range(2, 3):
            self.remove_output(i)

        # Set image output
        if len(masks) > 1:
            for i, mask in enumerate(masks):
                self.add_output(dataprocess.CSemanticSegmentationIO())
                mask = mask.astype("uint8")

                if param.input_size_percent < 100:
                    mask = resize_mask(mask, h_orig, w_orig)

                output = self.get_output(i + 1)
                output.set_mask(mask)
        else:
            mask = masks[0].astype("uint8")
            if param.input_size_percent < 100:
                mask = resize_mask(mask, h_orig, w_orig)

            # Set output mask (Semantic Seg)
            self.get_output(0)
            self.set_mask(mask)

        # Step progress bar (Ikomia Studio)
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


class InferSegmentAnything3Factory(dataprocess.CTaskFactory):
    """
    Factory class to create process object.
    Inherits PyDataProcess.CTaskFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_segment_anything_3"
        self.info.short_description = "Inference for Segment Anything Model 3 (SAM3) - Interactive instance segmentation."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Meta AI Research"
        self.info.article = "SAM 3: Segment Anything in Images and Videos"
        self.info.journal = "ArXiv"
        self.info.year = 2025
        self.info.license = "SAM License"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.16.0"

        # Python compatibility
        self.info.min_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://github.com/facebookresearch/sam3"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_segment_anything_3"
        self.info.original_repository = "https://github.com/facebookresearch/sam3"

        # Keywords used for search
        self.info.keywords = "SAM, SAM3, Segment Anything, ViT, Zero-Shot, Interactive Segmentation, Meta"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Algorithms tasks
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        """Instantiate algorithm object."""
        return InferSegmentAnything3(self.info.name, param)
