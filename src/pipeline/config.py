"""Pipeline configuration for M-041 (surgical_phase_recognition)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-041 pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]  # Max samples (None = all)
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="endoscapes_surgical_scene")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw M-041 data",
    )
    s3_prefix: str = Field(
        default="M-108_Endoscapes/raw/",
        description="S3 key prefix for the dataset raw data",
    )
    fps: int = Field(
        default=3,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default="This laparoscopic cholecystectomy frame (Endoscapes). Segment/identify safety-critical anatomy (Critical View of Safety) and surgical instruments.",
        description="The task instruction shown to the reasoning model.",
    )
