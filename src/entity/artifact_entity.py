from dataclasses import dataclass , field
from typing import Optional

@dataclass
class DataIngestionArtifact:
    feature_store_path:str


@dataclass
class DataValidationArtifact:
    valid_data_file_path: str  # Required field (must be before default fields)
    drift_report_file_path: Optional[str] = field(default=None)
    validation_status: Optional[bool] = field(default=None)

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str