from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path:str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_data_file_path: str
    drift_report_file_path: str