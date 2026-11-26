"""VERILENS - Member 3: entity extraction and rule-based mutation detection.

Public entry point::

    from shivansh_work import EntityMutationPipeline
"""

from .pipeline_interface import EntityMutationPipeline, PipelineConfig, to_json

__all__ = ["EntityMutationPipeline", "PipelineConfig", "to_json"]
