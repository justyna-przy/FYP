from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "bird_data"
    raw_dir: str = "raw"
    clips_dir: str = "clips"
    manifests_dir: str = "manifests"


@dataclass(frozen=True)
class XenoCantoConfig:
    base_url: str = "https://xeno-canto.org/api/3/recordings"
    quality: List[str] = field(default_factory=lambda: ["A", "B"])
    min_len: int = 2
    max_len: int = 60
    countries: List[str] = field(
        default_factory=lambda: [
            "United Kingdom", "Ireland", "France", "Germany", "Spain", "Portugal",
            "Italy", "Netherlands", "Belgium", "Switzerland", "Austria", "Denmark",
            "Sweden", "Norway", "Finland", "Poland", 
        ]
    )
    
    
@dataclass(frozen=True)
class PreprocessingConfig:
    # Audio
    sample_rate_model: int = 16000
    sample_rate_teacher: int = 48000  # BirdNET expects 48k input audio
    clip_len_s: float = 3.0
    stride_s: float = 3.0
    skip_first_s: float = 1.0 # skip initial silence

    # Recording cap
    recording_cap_s: float = 30.0
    step_cap_s: float = 5.0

    # RMS gating
    rms_keep_percentile: float = 30.0
    rms_abs_min_db: float = -40.0
    eps: float = 1e-10

    # Teacher thresholds
    bird_conf_thr: float = 0.1 
    species_conf_thr: float = 0.3

    max_species_clips_per_rec: int = 3
    nonbird_clips_per_rec: int = 2
    seed: int = 22 # Lucky number


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    xeno_canto: XenoCantoConfig = field(default_factory=XenoCantoConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


CONFIG = Config()
