from dataclasses import dataclass, field
from typing import List, Tuple


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
class SpectrogramConfig:
    version: str = "specs_v1"

    sample_rate: int = 16000

    # STFT / Mel
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    window: str = "hann"
    n_mels: int = 128
    fmin: float = 120.0
    fmax: float = 8000.0
    power: float = 2.0
    center: bool = False

    # dB + clamp (fixed)
    db_ref: str = "max"     # power_to_db(ref=np.max)
    db_min: float = -80.0
    db_max: float = 0.0

    # Output behavior
    flip_freq_axis: bool = True  # make low freqs bottom (like papers)

    # Save both:
    save_full_res_png: bool = True
    save_train_png: bool = True

    # Final train image size (H, W) = (mel, time)
    train_out_shape_hw: Tuple[int, int] = (64, 128)



@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    xeno_canto: XenoCantoConfig = field(default_factory=XenoCantoConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)



CONFIG = Config()
