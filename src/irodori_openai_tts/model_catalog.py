"""AlSlimeフォークのモデルカタログ。

フォーク追加分: AlSlime の画面から選べるチェックポイントの一覧。容量は
Hugging Face のファイル一覧に表示される合計（GB）を手で写した固定値で、
未ダウンロードのモデルにも「おおよその容量」を出すために使う。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    checkpoint: str
    name: str
    approx_size_gb: float


CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry("Aratako/Irodori-TTS-v4-Small", "Irodori-TTS v4 Small", 3.1),
    CatalogEntry("Aratako/Irodori-TTS-v4.1-Small", "Irodori-TTS v4.1 Small", 3.1),
    CatalogEntry("Aratako/Irodori-TTS-v4-Large", "Irodori-TTS v4 Large", 13.2),
)

# 音声コーデック（Semantic-DACVAE）の weights.pth。初回のモデル取得時に一緒に落とす。
CODEC_APPROX_SIZE_GB = 0.43


def catalog_entry(checkpoint: str) -> CatalogEntry | None:
    value = str(checkpoint).strip()
    for entry in CATALOG:
        if entry.checkpoint == value:
            return entry
    return None
