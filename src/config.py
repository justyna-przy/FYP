from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PathsConfig:
    root: str = "bird_data"
    raw_dir: str = "raw"
    clips_dir: str = "clips"


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
            "Sweden", "Norway", "Finland", "Poland", "Czech Republic", "Slovakia",
            "Hungary", "Greece", "Romania", "Bulgaria", "Croatia", "Slovenia",
            "Estonia", "Latvia", "Lithuania", "Luxembourg", "Liechtenstein",
            "Iceland", "Malta"
        ]
    )


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    xeno_canto: XenoCantoConfig = field(default_factory=XenoCantoConfig)


CONFIG = Config()
