"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_segment_anything_3.infer_segment_anything_3_process import InferSegmentAnything3Factory
from infer_segment_anything_3.infer_segment_anything_3_process import InferSegmentAnything3ParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferSegmentAnything3Factory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_segment_anything_3.infer_segment_anything_3_widget import InferSegmentAnything3WidgetFactory
        return InferSegmentAnything3WidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferSegmentAnything3ParamFactory()
