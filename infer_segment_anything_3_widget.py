"""
Module that implements the UI widget of the algorithm.
"""
from torch.cuda import is_available

from PyQt6.QtWidgets import *

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_segment_anything_3.infer_segment_anything_3_process import InferSegmentAnything3Param


class InferSegmentAnything3Widget(core.CWorkflowTaskWidget):
    """
    Class that implements UI widget to adjust algorithm parameters.
    Inherits PyCore.CWorkflowTaskWidget from Ikomia API.
    """

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferSegmentAnything3Param()
        else:
            self.parameters = param

        # Create layout : QVBoxLayout by default
        self.grid_layout = QVBoxLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Section 1: General Settings
        self.section1 = QGroupBox("General Settings")
        self.section1_layout = QGridLayout()
        self.section1.setLayout(self.section1_layout)
        self.grid_layout.addWidget(self.section1)

        # CUDA checkbox
        self.check_cuda = pyqtutils.append_check(
            self.section1_layout, "Cuda", self.parameters.cuda and is_available()
        )
        self.check_cuda.setEnabled(is_available())

        # Image size percentage
        self.spin_input_size_percent = pyqtutils.append_spin(
            self.section1_layout,
            "Image size (%)",
            self.parameters.input_size_percent,
            min=1,
            max=100
        )

        # Confidence threshold
        self.spin_confidence_threshold = pyqtutils.append_double_spin(
            self.section1_layout,
            "Confidence threshold",
            self.parameters.confidence_threshold,
            min=0.0,
            max=1.0,
            step=0.05
        )

        # Multimask output checkbox
        self.check_multimask_output = pyqtutils.append_check(
            self.section1_layout, "Multimask output", self.parameters.multimask_output
        )

        # Section 2: TEXT PREDICTOR
        self.toggle_text_button = QPushButton("▶ TEXT PREDICTOR")
        self.toggle_text_button.setCheckable(True)
        self.toggle_text_button.setChecked(False)
        self.toggle_text_button.toggled.connect(self.toggle_text_group)
        self.toggle_text_button.setStyleSheet(
            "text-align: left; padding: 5px;")
        self.grid_layout.addWidget(self.toggle_text_button)

        self.text_group = QGroupBox()
        self.text_layout = QGridLayout()
        self.text_group.setLayout(self.text_layout)
        self.text_group.setVisible(False)
        self.grid_layout.addWidget(self.text_group)

        # Text prompt input
        self.edit_text_input = pyqtutils.append_edit(
            self.text_layout,
            "Text prompt",
            self.parameters.input_text
        )

        # Section 3: PROMPT PREDICTOR
        self.toggle_prompt_button = QPushButton("▶ PROMPT PREDICTOR")
        self.toggle_prompt_button.setCheckable(True)
        self.toggle_prompt_button.setChecked(False)
        self.toggle_prompt_button.toggled.connect(self.toggle_prompt_group)
        self.toggle_prompt_button.setStyleSheet(
            "text-align: left; padding: 5px;")
        self.grid_layout.addWidget(self.toggle_prompt_button)

        self.prompt_group = QGroupBox()
        self.prompt_layout = QGridLayout()
        self.prompt_group.setLayout(self.prompt_layout)
        self.prompt_group.setVisible(False)
        self.grid_layout.addWidget(self.prompt_group)

        # Box input
        self.edit_box_input = pyqtutils.append_edit(
            self.prompt_layout,
            "Box coord. [[xyxy]]",
            self.parameters.input_box
        )

        # Point input
        self.edit_point_input = pyqtutils.append_edit(
            self.prompt_layout,
            "Point coord. [[xy]]",
            self.parameters.input_point
        )

        # Point label input
        self.edit_point_label = pyqtutils.append_edit(
            self.prompt_layout,
            "Point label [i]",
            self.parameters.input_point_label
        )

        # Set widget layout
        self.set_layout(layout_ptr)

    def toggle_text_group(self, checked):
        """Toggle visibility of text predictor section."""
        if checked:
            self.toggle_text_button.setText("▼ TEXT PREDICTOR")
        else:
            self.toggle_text_button.setText("▶ TEXT PREDICTOR")
        self.text_group.setVisible(checked)

    def toggle_prompt_group(self, checked):
        """Toggle visibility of prompt predictor section."""
        if checked:
            self.toggle_prompt_button.setText("▼ PROMPT PREDICTOR")
        else:
            self.toggle_prompt_button.setText("▶ PROMPT PREDICTOR")
        self.prompt_group.setVisible(checked)

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size_percent = self.spin_input_size_percent.value()
        self.parameters.confidence_threshold = self.spin_confidence_threshold.value()
        self.parameters.input_text = self.edit_text_input.text()
        self.parameters.input_box = self.edit_box_input.text()
        self.parameters.input_point = self.edit_point_input.text()
        self.parameters.input_point_label = self.edit_point_label.text()
        self.parameters.multimask_output = self.check_multimask_output.isChecked()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


class InferSegmentAnything3WidgetFactory(dataprocess.CWidgetFactory):
    """
    Factory class to create algorithm widget object.
    Inherits PyDataProcess.CWidgetFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_segment_anything_3"

    def create(self, param):
        """Instantiate widget object."""
        return InferSegmentAnything3Widget(param, None)
