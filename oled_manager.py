#!/usr/bin/python3
# -*- coding: utf-8 -*-
# oled_manager.py
###########################################################################
# Filename      :oled_manager.py
# Description   :Prodtrac OLED manager
# Author        :Akihiko Fujita
# Update        :2025/12/25
# Version       :1.7.10
############################################################################

import heapq
import importlib.util
import threading
import time
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import json

from prodtrac_utils import parse_float

# 単位表記を正規化して小文字の短縮形へ揃える
def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    """単位文字列を正規化し、使用できない値は ``None`` として扱う。

    引数:
        unit: メタデータから取得した単位文字列。

    戻り値:
        正規化された単位文字列、または ``None``。"""
    if not isinstance(unit, str):
        return None
    # 前後の空白を取り除き小文字化して比較しやすくする
    return unit.strip().lower()


# フレーム時間を数値化し範囲外は既定値にフォールバックする
def _coerce_duration(value: Any, scale: float, default: float) -> tuple[float, bool]:
    """フレーム時間を数値変換し、異常値は既定値で補正したうえで妥当性を返す。

    引数:
        value: メタデータから取得した時間値。
        scale: 単位に応じた倍率。
        default: 異常時に利用する既定の時間。

    戻り値:
        (正常化されたフレーム時間（秒）, 妥当性フラグ)。
        妥当性フラグは、入力が正の数値として解釈できた場合に ``True``。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default, False

    number *= scale
    if number <= 0:
        return default, False
    return number, True


def _duration_scale(unit: Optional[str]) -> float:
    """単位文字列に対応する倍率を返す。未知の単位は秒として扱う。"""

    unit_norm = _normalize_unit(unit)
    if unit_norm in {"ms", "millisecond", "milliseconds"}:
        return 0.001
    if unit_norm in {"us", "microsecond", "microseconds"}:
        return 0.000001
    if unit_norm in {"s", "sec", "second", "seconds", None}:
        return 1.0

    logging.debug("Unknown frame time unit '%s'; assuming seconds", unit_norm)
    return 1.0


def _load_metadata_file(meta_path: str) -> Any:
    if not os.path.isfile(meta_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        logging.warning(f"Failed to read frame timings '{meta_path}': {exc}")
        return None


def _normalize_sequence_length(sequence: List[Any], target_size: int) -> List[Any]:
    """フレーム数に合わせてシーケンス長を調整する。"""

    if len(sequence) < target_size:
        sequence = sequence + [None] * (target_size - len(sequence))
    elif len(sequence) > target_size:
        sequence = sequence[:target_size]
    return sequence


@dataclass(order=True)
class DisplayRequest:
    priority: int
    sequence: int
    action: Callable[[], None] = field(compare=False)
    label: str = field(default="", compare=False)
    wait_time: float = field(default=0.0, compare=False)


class DisplayRequestQueue:
    """表示要求を直列化する優先度付きキュー。"""

    PRIORITY_QR = 0
    PRIORITY_OVERLAY = 10
    PRIORITY_ANIMATION = 20
    PRIORITY_BACKGROUND = 30

    def __init__(self, name: str = "display-request-queue") -> None:
        self.name = name
        self._queue: list[DisplayRequest] = []
        self._condition = threading.Condition()
        self._stopped = False
        self._seq = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(
        self,
        action: Callable[[], None],
        *,
        priority: int = PRIORITY_OVERLAY,
        wait: Optional[float] = None,
        label: str = "",
    ) -> bool:
        """アクションを優先度付きで登録する。

        Args:
            action: 実行する呼び出し可能オブジェクト。
            priority: 小さいほど優先度が高い。
            wait: 実行後に次の要求まで待機する秒数。
            label: ログ用のラベル。
        """

        if not callable(action):
            return False

        wait_time = parse_float(wait, default=0.0) if wait is not None else 0.0
        if wait_time is None:
            wait_time = 0.0
        wait_time = max(0.0, float(wait_time))

        with self._condition:
            if self._stopped:
                return False
            self._seq += 1
            heapq.heappush(
                self._queue,
                DisplayRequest(
                    priority,
                    self._seq,
                    action,
                    label,
                    wait_time,
                ),
            )
            self._condition.notify()
        return True

    def stop(self, *, drain: bool = True) -> None:
        """ワーカースレッドを停止し、必要に応じてキューを破棄する。"""

        with self._condition:
            self._stopped = True
            if not drain:
                self._queue.clear()
            self._condition.notify_all()

        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopped:
                    self._condition.wait()

                if self._stopped and not self._queue:
                    break

                request = heapq.heappop(self._queue)

            try:
                request.action()
            except Exception:
                logging.exception(
                    "DisplayRequestQueue '%s' failed to execute request '%s'", self.name, request.label
                )

            if request.wait_time > 0:
                try:
                    time.sleep(request.wait_time)
                except Exception:
                    logging.debug(
                        "DisplayRequestQueue '%s' wait interrupted for '%s'", self.name, request.label
                    )


class DisplayConfigLoader:
    """フレームタイミングやメタデータ読み込みを一元化するローダー。"""
    def __init__(
        self,
        *,
        default_frame_time: float = 0.10,
        min_frame_time: float = 0.01,
        max_frame_time: Optional[float] = None,
    ):
        self.default_frame_time = default_frame_time
        self.min_frame_time = min_frame_time
        self.max_frame_time = max_frame_time

    def load_metadata(self, meta_path: str) -> Any:
        return _load_metadata_file(meta_path)

    def _normalize_duration(self, raw: Any, scale: float) -> tuple[float, bool]:
        coerced, is_valid = _coerce_duration(
            raw, scale, default=self.default_frame_time
        )

        if coerced < self.min_frame_time:
            return self.default_frame_time, False

        if self.max_frame_time is not None and coerced > self.max_frame_time:
            return self.default_frame_time, False

        return coerced, is_valid

    def _collect_sequence(self, metadata: Any, png_files: List[str]) -> tuple[Optional[str], List[Any]]:
        unit = None
        raw_sequence: List[Any] = []

        if isinstance(metadata, list):
            raw_sequence = list(metadata)
        elif isinstance(metadata, dict):
            unit = metadata.get("unit") or metadata.get("units")

            list_candidates = ("frames", "durations", "timings", "values", "frame_times")
            for key in list_candidates:
                candidate = metadata.get(key)
                if isinstance(candidate, list):
                    raw_sequence = list(candidate)
                    break

            if not raw_sequence:
                mapping_keys = ("frames", "durations", "timings", "frame_map", "values")
                mapping = None
                for key in mapping_keys:
                    candidate = metadata.get(key)
                    if isinstance(candidate, dict):
                        mapping = candidate
                        break

                if isinstance(mapping, dict):
                    for idx, path in enumerate(png_files):
                        base = os.path.basename(path)
                        stem, _ = os.path.splitext(base)
                        for key in (base, stem, str(idx)):
                            if key in mapping:
                                raw_sequence.append(mapping.get(key))
                                break
                        else:
                            raw_sequence.append(mapping.get("default"))

            if not raw_sequence and "default" in metadata:
                raw_sequence = [metadata.get("default")] * len(png_files)

        return unit, _normalize_sequence_length(raw_sequence, len(png_files))

    def _normalize_sequence(self, raw_sequence: List[Any], *, scale: float, frame_count: int) -> List[float]:
        normalized: List[float] = []
        valid_count = 0

        for raw in _normalize_sequence_length(raw_sequence, frame_count):
            coerced, is_valid = self._normalize_duration(raw, scale)
            normalized.append(coerced)
            if is_valid:
                valid_count += 1

        if not normalized:
            normalized = [self.default_frame_time] * frame_count
        elif valid_count == 0:
            normalized = [self.default_frame_time] * len(normalized)

        return normalized

    def frame_timings(self, meta_path: str, png_files: List[str]) -> List[float]:
        metadata: Any = self.load_metadata(meta_path)
        unit, raw_sequence = self._collect_sequence(metadata, png_files)

        scale = _duration_scale(unit)
        normalized = self._normalize_sequence(
            raw_sequence, scale=scale, frame_count=len(png_files)
        )

        logging.debug(
            "Frame timings loaded from %s -> %s",
            meta_path,
            normalized[: min(5, len(normalized))],
        )

        return normalized


def _load_frame_timings(
    meta_path: str, png_files: List[str], default_frame_time: float
) -> List[float]:
    """DisplayConfigLoader を利用してフレーム時間を正規化する。"""

    loader = DisplayConfigLoader(default_frame_time=default_frame_time)
    return loader.frame_timings(meta_path, png_files)


def _resolve_frame_timings(
    frames_dir: str,
    *,
    meta_json: str = "frame_times.json",
    default_frame_time: float = 0.10,
    min_frame_time: float = 0.01,
    max_frame_time: Optional[float] = None,
) -> Tuple[List[str], List[float], Optional[str]]:
    """フレームPNGとタイミングを共通のルートで解決する。"""

    try:
        entries = os.listdir(frames_dir)
    except Exception as exc:
        logging.error(f"Failed to list frames_dir '{frames_dir}': {exc}")
        return [], [], None

    png_files = sorted(
        os.path.join(frames_dir, f) for f in entries if f.lower().endswith(".png")
    )
    if not png_files:
        logging.warning(f"No PNG files found in {frames_dir}")
        return [], [], None

    meta_path = (
        meta_json if os.path.isabs(meta_json) else os.path.join(frames_dir, meta_json)
    )

    loader = DisplayConfigLoader(
        default_frame_time=default_frame_time,
        min_frame_time=min_frame_time,
        max_frame_time=max_frame_time,
    )
    frame_times = loader.frame_timings(meta_path, png_files)

    return png_files, frame_times, meta_path

try:
    from PIL import Image, ImageFont, ImageDraw
except Exception:
    Image = None
    ImageFont = None
    ImageDraw = None

# 1bit PNGの高速取り扱い用キャッシュ（Palette -> 0/1マップ）
_PALETTE_BIT_CACHE: Dict[bytes, bytes] = {}

# 1bitパレット画像を高速にモノクロへ変換する
def _repack_palette_1bit(image: "Image.Image") -> Optional["Image.Image"]:
    """palette形式(1bit)のPNGを高速にmode="1"へ再パックする。

    引数:
        image: PILで読み込んだパレット形式の画像。 ``None`` は許容しない。

    戻り値:
        再パックした ``Image`` オブジェクト。条件を満たさない場合は ``None``。"""

    if Image is None:
        return None

    if getattr(image, "mode", None) != "P":
        return None

    if image.info.get("bits") != 1:
        return None

    palette = image.getpalette()
    if not palette:
        return None

    palette_bytes = bytes(palette)
    lut = _PALETTE_BIT_CACHE.get(palette_bytes)

    if lut is None:
        try:
            used_colors = image.getcolors(maxcolors=256)
        except Exception:
            used_colors = None

        if not used_colors:
            return None

        # 輝度の低い色をoff、高い色をonとして判定する
        def _luminance(index: int) -> int:
            base = index * 3
            if base + 2 >= len(palette):
                return 0
            r, g, b = palette[base : base + 3]
            return r * 299 + g * 587 + b * 114

        sorted_by_luma = sorted(used_colors, key=lambda item: _luminance(item[1]))
        if not sorted_by_luma:
            return None

        darkest_index = sorted_by_luma[0][1]
        brightest_index = sorted_by_luma[-1][1]

        table = bytearray(256)
        table[brightest_index] = 1 if brightest_index != darkest_index else 0
        lut = bytes(table)
        _PALETTE_BIT_CACHE[palette_bytes] = lut

    try:
        width, height = image.size
        stride = (width + 7) // 8
        packed = bytearray(stride * height)
        src = memoryview(image.tobytes())
        lut_view = memoryview(lut)
        src_index = 0

        for y in range(height):
            dest_index = y * stride
            byte_val = 0
            bit = 7
            for x in range(width):
                # ルックアップテーブルを参照し輝度に応じてビットを立てる
                if lut_view[src[src_index]]:
                    byte_val |= 1 << bit
                bit -= 1
                src_index += 1
                if bit < 0:
                    packed[dest_index] = byte_val
                    dest_index += 1
                    byte_val = 0
                    bit = 7

            if bit != 7:
                packed[dest_index] = byte_val

        return Image.frombytes("1", (width, height), bytes(packed))
    except Exception:
        return None

# =============== デフォルト値設定 =======================================

# このディレクトリ上に存在するアニメーションは、ディレクトリ指定がある場合は参照されない画像
DEFAULT_PAIR_ANIM_DIR = "/home/pi/Prodtrac/png_def"

# luma.oled に合わせたデフォルト
DEFAULT_DISPLAY_CONFIG = {
    "contrast": 255,                              # 0-255: ssd1309 で有効
    "font_size": 12,                              # truetype使用時のサイズ
    "font_path": None,                            # Noneなら Pillow デフォルトフォント
    "screensaver_enabled": True,
    "screensaver_timeout_sec": 120*60,            # 必要なら消灯時コントラスト値
    "screensaver_contrast": 0,                    # 0で実質ブラックアウト（任意）
    "default_font_rel": None,                     # 内蔵フォントパスの相対位置をデフォルト化したい場合 例: "assets/fonts/MyFont.ttf"（未指定なら None）
    "pair_animation_dir": DEFAULT_PAIR_ANIM_DIR,  # ペア成立時アニメーションの既定配置場所
    "suppress_initial_screen": True,              # 起動時の初期画面更新を抑制する。
}

# 参考情報用（display_animation は endswith('.png') 固定なのでこの定数は使わなくてもOK）
PAIR_ANIM_FRAME_GLOB = "frame_apngframe*.png"

# =============== OLED設定パラメータ =======================================
# デバッグモードフラグ
def _env_flag(name: str, default: bool = False) -> bool:
    """環境変数から真偽値を読み取る。"""

    raw = os.environ.get(name)
    if raw is None:
        return default

    if isinstance(raw, str):
        raw_normalized = raw.strip().lower()
    else:
        raw_normalized = str(raw).strip().lower()

    if not raw_normalized:
        return default

    return raw_normalized in {"1", "true", "yes", "on", "enable", "enabled"}


# ここを明示的にTrue/Falseにすれば環境変数に関係なく固定、コメントアウト時や空白時は環境変数を参照 (通常はFalse)
DEBUG_MODE_DEFAULT = False
DEBUG_MODE = _env_flag("OLED_DEBUG", DEBUG_MODE_DEFAULT)


# 環境変数読み込み関数
def get_env_int(
    name: str,
    default: int,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> int:
    """
    環境変数から整数値を安全に取得し、指定された範囲内にあるか検証する
    範囲外の場合は制限値に修正し、数値に変換できない場合はデフォルト値を使用
    """
    try:
        value = int(os.environ.get(name, default))
        if min_val is not None and value < min_val:
            logging.warning(
                f"{name} value {value} below minimum {min_val}, using {min_val}"
            )
            return min_val
        if max_val is not None and value > max_val:
            logging.warning(
                f"{name} value {value} above maximum {max_val}, using {max_val}"
            )
            return max_val
        return value
    except ValueError:
        logging.warning(f"Invalid {name} value, using default {default}")
        return int(default)


# 接続方式: 'i2c' または 'spi' を指定
OLED_CONNECTION_TYPE = os.environ.get("OLED_CONNECTION_TYPE", "i2c")

# I2C設定パラメータ
I2C_PORT = get_env_int("I2C_PORT", 1, 0, 10)  # 通常はRaspberry Piではポート1
I2C_ADDRESS = int(
    os.environ.get("I2C_ADDRESS", "0x3C"), 0
)  # デフォルト0x3C、モデルにより0x3Dもあり sudo i2cdetect -y 1 で確認

# SPI設定パラメータ
SPI_PORT = get_env_int("SPI_PORT", 0, 0, 10)  # 通常は0
SPI_DEVICE = get_env_int("SPI_DEVICE", 0, 0, 10)  # CE0=0, CE1=1
SPI_BUS_SPEED = get_env_int("SPI_BUS_SPEED", 8000000, 1000000, 32000000)  # 8MHz

# ディスプレイパラメータ
OLED_WIDTH = get_env_int("OLED_WIDTH", 128, 1, 256)  # ディスプレイ横ピクセル数
OLED_HEIGHT = get_env_int("OLED_HEIGHT", 64, 1, 128)  # ディスプレイ縦ピクセル数
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FONT_FILENAME = "JF-Dot-MPlusH12.ttf"
DEFAULT_FONT_PATH = BASE_DIR / DEFAULT_FONT_FILENAME
STATUS_FONT_FILENAME = "JF-Dot-MPlusH10.ttf"
STATUS_FONT_PATH = BASE_DIR / STATUS_FONT_FILENAME

OLED_FONT_PATH = os.environ.get("OLED_FONT_PATH", str(DEFAULT_FONT_PATH))
"""フォントファイルパス。環境変数が未指定の場合は同梱フォントを指す。"""
OLED_FONT_SIZE = get_env_int("OLED_FONT_SIZE", 12, 8, 24)  # フォントサイズ(pt)
# ==========================================================================

try:
    from luma.core.interface.serial import i2c, spi
    from luma.oled.device import ssd1309

    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False


GPIO_AVAILABLE = False
GPIO = None
try:
    gpio_spec = importlib.util.find_spec("RPi.GPIO")
except ModuleNotFoundError:
    gpio_spec = None

if gpio_spec is not None:
    import RPi.GPIO as GPIO  # type: ignore[import-untyped]

    GPIO_AVAILABLE = True
    GPIO.setwarnings(False)


class OledDisplayManager:
    """
    I2CまたはSPI接続の有機ELディスプレイ制御を担う。
    接続方法は環境変数またはグローバル設定で切り替え可能。
    状態情報や任意メッセージをスレッド安全に表示、スレッド駆動。
    OLEDが無い/初期化失敗時はエラー/再初期化、自動復帰にも対応。
    """

    def _resolve_font_path(self) -> Optional[str]:
        """設定からフォントパスを解決する。"""

        candidates: List[Optional[str]] = []

        if hasattr(self, "display_cfg"):
            candidates.append(self.display_cfg.get("font_path"))
            candidates.append(self.display_cfg.get("default_font_rel"))

        candidates.append(OLED_FONT_PATH)
        candidates.append(str(DEFAULT_FONT_PATH))

        for candidate in candidates:
            if not isinstance(candidate, str):
                continue

            normalized = candidate.strip()
            if not normalized:
                continue

            expanded = os.path.expanduser(normalized)
            if not os.path.isabs(expanded):
                expanded = os.path.join(str(BASE_DIR), expanded)

            expanded = os.path.abspath(expanded)
            if os.path.exists(expanded):
                return expanded

        return None

    # フォントを安全にロードし、失敗時はデフォルトにフォールバックする
    def _load_font_object(self) -> Optional[Any]:

        if ImageFont is None:
            return None

        font_path = self._resolve_font_path()

        if not hasattr(self, "_font_warning_paths"):
            self._font_warning_paths = set()

        def _warn_font_once(message: str, exc: Exception):
            if font_path in self._font_warning_paths:
                logging.debug(message, font_path, exc)
            else:
                logging.warning(message, font_path, exc)
                self._font_warning_paths.add(font_path)

        try:
            font_size = int(self.display_cfg.get("font_size", OLED_FONT_SIZE))
        except Exception:
            font_size = OLED_FONT_SIZE

        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception as fe:
                _warn_font_once("Failed to load truetype font '%s': %s", fe)

        elif font_path:
            logging.warning("Font file not found: %s", font_path)

        try:
            return ImageFont.load_default()

        except Exception as fe3:
            logging.warning(
                "Failed to load default PIL font: %s. Using no font; text rendering disabled.",
                fe3,
            )
            return None

    def _load_message_font(
        self,
        font_path: Optional[str],
        font_size: Optional[int],
        fallback_font: Optional[Any],
    ) -> Optional[Any]:
        """一時メッセージ表示用のフォントを解決する。

        Args:
            font_path: 呼び出し元から指定されたフォントパス。
            font_size: 呼び出し元から指定されたフォントサイズ。
            fallback_font: ロードに失敗した場合のフォントオブジェクト。
        """

        if ImageFont is None:
            return fallback_font

        resolved_path: Optional[str] = None

        if isinstance(font_path, str) and font_path.strip():
            expanded = os.path.expanduser(font_path.strip())
            if not os.path.isabs(expanded):
                expanded = os.path.join(str(BASE_DIR), expanded)

            expanded = os.path.abspath(expanded)
            if os.path.exists(expanded):
                resolved_path = expanded
            else:
                logging.warning("Message font path not found: %s", expanded)

        target_size = None
        if font_size is not None:
            try:
                target_size = int(font_size)
            except (TypeError, ValueError):
                logging.warning("Invalid message font size: %s", font_size)

        if resolved_path:
            try:
                return ImageFont.truetype(
                    resolved_path, target_size if target_size else OLED_FONT_SIZE
                )
            except Exception as exc:
                logging.warning("Failed to load message font '%s': %s", resolved_path, exc)

        return fallback_font

    
    def _safe_text_height(
        self,
        draw: "ImageDraw.ImageDraw",
        text: Any,
        font: Optional[Any] = None,
        default_height: int = 12,
    ) -> int:
        """textbbox/getsizeの高さ取得時にエンコード例外を握りつぶし、既定値で返す。"""

        normalized = "" if text is None else str(text)
        font_to_use = font or self.font

        if font_to_use is None and ImageFont is not None:
            try:
                font_to_use = ImageFont.load_default()
                self.font = font_to_use
            except Exception as exc:
                logging.debug(
                    "Failed to obtain default font for text height calc: %s", exc
                )
                font_to_use = None

        height_fallback = getattr(font_to_use, "size", default_height) if font_to_use else default_height

        if font_to_use is None:
            return height_fallback

        try:
            bbox = draw.textbbox((0, 0), normalized, font=font_to_use)
            return bbox[3] - bbox[1]
        except Exception as exc:
            is_encoding_error = isinstance(exc, UnicodeEncodeError) or (
                "codec can't encode" in str(exc)
            )

            if is_encoding_error:
                logging.debug(
                    "Text height skipped due to encoding issue for '%s': %s", normalized, exc
                )
            try:
                return font_to_use.getsize(normalized)[1]
            except Exception as exc2:
                is_encoding_error_2 = isinstance(exc2, UnicodeEncodeError) or (
                    "codec can't encode" in str(exc2)
                )

                if is_encoding_error_2:
                    logging.debug(
                        "Fallback text height skipped due to encoding issue for '%s': %s",
                        normalized,
                        exc2,
                    )
                else:
                    logging.debug(
                        "Fallback text height failed for '%s': %s", normalized, exc2
                    )
                return height_fallback

    def _safe_draw_text(
        self,
        draw: "ImageDraw.ImageDraw",
        position: Sequence[Union[int, float]],
        text: Any,
        *,
        fill: int,
        font: Optional[Any] = None,
    ) -> None:
        """テキスト描画時の例外を抑止し、フォントが壊れても描画を続行する。"""

        if ImageDraw is None:
            return

        normalized = "" if text is None else str(text)
        font_to_use = font or self.font

        if font_to_use is None and ImageFont is not None:
            try:
                font_to_use = ImageFont.load_default()
                self.font = font_to_use
            except Exception as exc:
                logging.warning(
                    "Failed to obtain default font for text rendering: %s", exc
                )
                font_to_use = None

        if font_to_use is None:
            logging.warning("Skipping text rendering because no font is available")
            return

        try:
            draw.text(position, normalized, font=font_to_use, fill=fill)
        except Exception as exc:
            is_encoding_error = isinstance(exc, UnicodeEncodeError) or (
                "codec can't encode" in str(exc)
            )

            if is_encoding_error:
                logging.debug(
                    "Text render skipped due to encoding issue for '%s': %s",
                    normalized,
                    exc,
                )
            else:
                logging.warning(
                    "Text render failed for '%s': %s; retrying with default font",
                    normalized,
                    exc,
                )
            try:
                fallback = ImageFont.load_default()
                draw.text(position, normalized, font=fallback, fill=fill)
                self.font = fallback
            except Exception as exc2:
                is_encoding_error_2 = isinstance(exc2, UnicodeEncodeError) or (
                    "codec can't encode" in str(exc2)
                )

                if is_encoding_error_2:
                    logging.debug(
                        "Fallback text render skipped due to encoding issue for '%s': %s",
                        normalized,
                        exc2,
                    )
                else:
                    logging.error("Fallback text render failed: %s", exc2)


                if is_encoding_error_2:
                    logging.debug(
                        "Fallback text render skipped due to encoding issue for '%s': %s",
                        normalized,
                        exc2,
                    )
                else:
                    logging.error("Fallback text render failed: %s", exc2)

    # OLEDマネージャの初期化とスレッド起動、設定反映を行う
    def __init__(
        self,
        connection_type: Optional[str] = None,
        display_cfg: Optional[Mapping[str, Any]] = None,
        request_queue: Optional["DisplayRequestQueue"] = None,
    ):

        self.lock = threading.Lock()                     # 描画データ・状態フラグなどの排他制御用ロック
        self.display_data: Dict[str, Any] = {}           # 画面に表示する最新データ（状態・タイマー・作業者名など）を保持する共有ディクショナリ
        self._stop = (threading.Event())                 # display_loop スレッドに停止を指示するためのイベントフラグ
        self.oled_ok = False                             # 現在 OLED デバイスが使用可能かどうか（初期化成功かつ利用中）を示すフラグ
        self.error_msg = (None)                          # 最新のエラーメッセージ（ログ/表示用）。None の場合はエラーなし
        self.need_reinit = False                         # 自動復帰方式のための再初期化要求フラグ（True のとき次サイクルで再初期化を試行）
        self.device = None                               # 実際のデバイスインスタンス（luma のデバイス or Dummy）。未初期化時は None
        self.font = None                                 # 描画に使用するフォントオブジェクト（PIL の ImageFont）。未ロード時は None
        self._last_activity_time = (time.time())         # 最終ユーザー操作（または状態更新）時刻。スクリーンセーバー/減光の判定に使用
        self._screensaver = False                        # スクリーンセーバーが現在有効かどうかの状態フラグ
        self._pre_screensaver_blink = None               # スクリーンセーバー突入前のblink状態を退避するフィールド
        self._pending_screensaver_restore = False        # セーバー復帰時に前回画面を即時再描画するためのフラグ
        self._screensaver_snapshot: Dict[str, Any] = {}  # セーバー突入前の表示データ退避
        self._pending_redraw = False                     # アニメーションなどで通常画面に戻す必要がある場合の再描画要求フラグ

        # OLED初期化強制ダミーフラグ（再初期化ループ抑止用）
        self._force_dummy = False

        self.request_queue = request_queue or DisplayRequestQueue("oled-display")
        self._owns_request_queue = request_queue is None

        self.display_cfg = dict(DEFAULT_DISPLAY_CONFIG)
        if display_cfg:
            self.display_cfg.update(display_cfg)

        # 接続タイプ確定
        self.connection_type = connection_type or OLED_CONNECTION_TYPE
        if self.connection_type not in ("i2c", "spi"):
            self.connection_type = "i2c"
            logging.warning("Invalid connection type specified, defaulting to I2C")


        # デバイス初期化後にコントラストを適用（フォントは _init_oled 側で設定済み）
        if OLED_AVAILABLE:
            self._init_oled()
            try:
                if self.device and hasattr(self.device, "contrast"):
                    contrast = int(self.display_cfg.get("contrast", DEFAULT_DISPLAY_CONFIG["contrast"]))
                    self.device.contrast(max(0, min(255, contrast)))
            except Exception as e:
                logging.warning(f"Failed to apply contrast: {e}")
        else:
            self.error_msg = "OLED library not found"

        # スレッドは初期化結果に関係なく起動し、必要に応じて再初期化を試みる
        self.thread = threading.Thread(target=self.display_loop, daemon=True)
        self.thread.start()

        if DEBUG_MODE:
            print(f"[DEBUG] OledDisplayManager initialized: {self.debug_info()}")
            logging.debug(f"OledDisplayManager initialized: {self.debug_info()}")


    def __enter__(self):
        """
        コンテキストマネージャーのエントリーポイント
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        コンテキストマネージャーの終了時に呼び出され、リソースをクリーンアップ
        """
        self.stop()


    def submit_display_request(
        self,
        action: Callable[[], None],
        *,
        priority: Optional[int] = None,
        wait: Optional[float] = None,
        label: Optional[str] = None,
    ) -> bool:
        """表示要求をキュー経由で実行するヘルパー。"""

        resolved_priority = (
            priority if priority is not None else DisplayRequestQueue.PRIORITY_OVERLAY
        )
        wait_time = parse_float(wait, default=0.0) if wait is not None else 0.0
        if wait_time is None:
            wait_time = 0.0

        queue = getattr(self, "request_queue", None)
        if isinstance(queue, DisplayRequestQueue):
            return queue.submit(
                action, priority=resolved_priority, wait=wait_time, label=label or ""
            )

        try:
            action()
            if wait_time:
                time.sleep(max(0.0, float(wait_time)))
            return True
        except Exception:
            logging.exception("Display request execution failed (no queue).")
            return False


    def _init_oled(self):
        """
        OLEDハード・ライブラリの初期化。
        接続方式に応じて I2C または SPI 経由でデバイスを初期化。
        成功すれば oled_ok=True、失敗なら error_msg をセット。
        - luma.core 2.4.2 / luma.oled 3.14.0 に合わせて SPI の引数名を bus_speed_hz に修正。
        - PIL 不在時は Dummy デバイスに切り替え、後段描画はスキップ/ログのみで継続可能にする。
        """

        # すでにDummy強制モードなら再初期化しない
        if getattr(self, "_force_dummy", False):
            return

        # 簡易ダミーデバイス（PIL 不在時やヘッドレス運用向け）
        class DummyOledDevice:
            def __init__(self, width=128, height=64):
                self.width = width
                self.height = height

            def contrast(self, value):
                logging.debug(f"[DummyOLED] contrast({value})")

            def display(self, image=None):
                logging.debug("[DummyOLED] display() called")

            def clear(self):
                logging.debug("[DummyOLED] clear() called")

        try:
            # PIL 不在ならダミーデバイスに切り替え
            if Image is None or ImageFont is None or ImageDraw is None:
                logging.warning(
                    "PIL not available: switching to Dummy OLED device (no image/text rendering)."
                )
                self.device = DummyOledDevice(width=OLED_WIDTH, height=OLED_HEIGHT)
                self.font = None
                self.oled_ok = True
                self.error_msg = None
                if DEBUG_MODE:
                    print(f"[DEBUG] OLED initialized in dummy mode: {self.debug_info() if hasattr(self, 'debug_info') else ''}")
                    logging.debug(
                        f"OLED initialized in dummy mode: {self.debug_info() if hasattr(self, 'debug_info') else ''}"
                    )
                return

            # 通信インターフェース初期化
            if self.connection_type == "i2c":
                logging.info(
                    f"Initializing OLED display via I2C (port={I2C_PORT}, address=0x{I2C_ADDRESS:X})"
                )
                serial_if = i2c(port=I2C_PORT, address=I2C_ADDRESS)
            else:  # 'spi'
                logging.info(
                    f"Initializing OLED display via SPI (port={SPI_PORT}, device={SPI_DEVICE}, speed={SPI_BUS_SPEED}Hz)"
                )
                # luma.core 2.4.2 では bus_speed_hz が正しい引数名
                serial_if = spi(
                    port=SPI_PORT,
                    device=SPI_DEVICE,
                    bus_speed_hz=SPI_BUS_SPEED,
                    # 必要に応じて gpio=... を追加（RST/DC 制御が必要な場合）
                )

            # デバイス生成（最大3回リトライ）
            MAX_RETRY = 3
            RETRY_DELAY_SEC = 1.5
            for attempt in range(1, MAX_RETRY + 1):
                try:
                    logging.info(f"OLED init attempt {attempt}/{MAX_RETRY}")
                    self.device = ssd1309(serial_if, width=OLED_WIDTH, height=OLED_HEIGHT)
                    self.oled_ok = True
                    self.error_msg = None
                    self.need_reinit = False
                    break
                except Exception as e:
                    logging.error(f"OLED init failed (try {attempt}/{MAX_RETRY}): {e}")
                    self.oled_ok = False
                    self.error_msg = f"OLED init failed: {e}"
                    if attempt < MAX_RETRY:
                        time.sleep(RETRY_DELAY_SEC)
            else:
                # 3回失敗した場合はDummyに切り替え
                logging.warning(
                    f"OLED initialization failed after {MAX_RETRY} attempts. Switching to Dummy device."
                )
                class DummyOledDevice:
                    def __init__(self, width=128, height=64):
                        self.width = width
                        self.height = height
                    def contrast(self, value): logging.debug(f"[DummyOLED] contrast({value})")
                    def display(self, image=None): logging.debug("[DummyOLED] display() called")
                    def clear(self): logging.debug("[DummyOLED] clear() called")
                self.device = DummyOledDevice(width=OLED_WIDTH, height=OLED_HEIGHT)
                self.oled_ok = True
                self.need_reinit = False
                self._force_dummy = True   # ★ 以後の再初期化を封止
                self.error_msg = f"OLED not detected; switched to Dummy"
                return

            # フォントのロード（PIL が使える前提で到達）
            self.font = self._load_font_object()
            if self.font is None:
                logging.warning(
                    "No font could be loaded; text rendering will be skipped until a font becomes available."
                )

            # 初期化成功
            self.oled_ok = True
            self.error_msg = None

            if DEBUG_MODE and hasattr(self, "debug_info"):
                print(f"[DEBUG] OLED initialized successfully: {self.debug_info()}")
                logging.debug(f"OLED initialized successfully: {self.debug_info()}")

        except TypeError as te:
            # 典型: 引数名の不一致など
            self.oled_ok = False
            self.error_msg = f"OLED init failed (TypeError): {te}"
            logging.error(self.error_msg)

        except Exception as e:
            self.oled_ok = False
            self.error_msg = f"OLED init failed: {e}"
            logging.error(self.error_msg)

    # スクリーンセーバー解除直後に前回画面を再描画するためのフラグ設定
    def _mark_screensaver_restore_needed(self) -> None:
        self._pending_screensaver_restore = True

    # 外部イベント後に通常画面へ復帰させるための再描画要求をセットする
    def _mark_redraw_needed(self) -> None:
        with self.lock:
            self._pending_redraw = True
            self._last_activity_time = time.time()


    def update(self, **kwargs):
        """
        表示内容データを受け取り、逐次スレッド安全に更新する
        引数: kwargs 任意の状態パラメータ(process_lcd, worker_lcd, 等)
        戻り値: なし
        """
        with self.lock:
            special_active = False
            message_fields = {}

            # エラーなど一時メッセージ表示中に通常更新が上書きしないよう保持
            if hasattr(self, "special_display_until") and self.special_display_until:
                try:
                    special_active = time.time() < float(self.special_display_until)
                except Exception:
                    special_active = False

            if special_active and isinstance(self.display_data, dict):
                for key in (
                    "message",
                    "_message_font_path",
                    "_message_font_size",
                    "_message_invert",
                ):
                    if key in self.display_data:
                        message_fields[key] = self.display_data[key]

            self.display_data.update(kwargs)

            if special_active and message_fields:
                self.display_data.update(message_fields)
            self._last_activity_time = time.time()
            # 入力が来たら直ちにスクリーンセーバー解除
            if self._screensaver:
                self._screensaver = False
                self._exit_screensaver()
                # ★ スクリーンセーバー解除後にblink状態を復元
                if self._pre_screensaver_blink:
                    if isinstance(self.display_data, dict):
                        self.display_data["show_blink"] = True
                self._pre_screensaver_blink = None
                self._mark_screensaver_restore_needed()

    def force_screensaver_on(self) -> None:
        """QRコードなど外部契機で即時スクリーンセーバーに移行する。"""

        with self.lock:
            if self._screensaver:
                return

            if isinstance(self.display_data, dict):
                self._pre_screensaver_blink = self.display_data.get("show_blink", False)
                self._screensaver_snapshot = self._capture_display_snapshot()

            self._screensaver = True
            self._last_activity_time = time.time()

        self._enter_screensaver()
        if DEBUG_MODE:
            logging.info("[force_screensaver_on] OLED screensaver ON (QR trigger)")

    def force_screensaver_off(self) -> None:
        """外部契機でスクリーンセーバーから復帰する。"""

        with self.lock:
            if not self._screensaver:
                return

            self._screensaver = False
            self._last_activity_time = time.time()

            if self._pre_screensaver_blink and isinstance(self.display_data, dict):
                self.display_data["show_blink"] = True

            self._pre_screensaver_blink = None
            self._mark_screensaver_restore_needed()

        self._exit_screensaver()
        if DEBUG_MODE:
            logging.info("[force_screensaver_off] OLED screensaver OFF (QR trigger)")


# エラー内容をOLEDに表示する
    def show_error(self, lines, duration=None, *, invert: bool = True):
        """
        エラー内容をOLEDに表示する。
        :param lines: 1～2行の文字列リスト
        :param duration: 表示継続秒数 (Noneなら固定表示)
        """
        if isinstance(lines, (list, tuple)):
            normalized_lines = [str(line) for line in lines if line is not None]
        else:
            normalized_lines = [str(lines)]

        if not normalized_lines:
            normalized_lines = [""]
        try:
            # PIL 不在時はコンソール出力にフォールバック
            if Image is None or ImageDraw is None:
                print("[OLED ERROR]", " / ".join(normalized_lines))
                return

            if self.device is None:
                print("[OLED ERROR]", " / ".join(normalized_lines))
                return

            width = getattr(self.device, "width", OLED_WIDTH)
            height = getattr(self.device, "height", OLED_HEIGHT)

            background_fill = 1 if invert else 0
            text_fill = 0 if invert else 1

            img = Image.new("1", (width, height), background_fill)  # 最後の数字が0=黒、1=白
            draw = ImageDraw.Draw(img)

            # フォントフォールバック
            font = self.font or self._load_font_object()
            if font is not None:
                self.font = font

            if font is None:
                logging.warning("No font available for show_error; falling back to console output.")
                print("[OLED ERROR]", " / ".join(lines))
                return

            # 指定に応じた配色で描画する（invert=True で白背景/黒文字）
            y = 0
            for line in normalized_lines:
                # Pillow 将来互換: textbbox で高さ算出（getsizeでも可）
                try:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_h = bbox[3] - bbox[1]
                except Exception:
                    line_h = font.getsize(line)[1]
                self._safe_draw_text(draw, (0, y), line, font=font, fill=text_fill)  # 最後の数字が0=黒、1=白
                y += line_h

            self.device.display(img)

            if duration:
                time.sleep(duration)
                # 画面クリアは self.clear() ではなく device.clear() にフォールバック
                if hasattr(self.device, "clear"):
                    try:
                        self.device.clear()
                    except Exception as ce:
                        logging.debug(f"device.clear() failed after show_error: {ce}")

        except Exception as e:
            # OLEDが物理的に死んでいる場合など
            print(f"[OLED ERROR fallback] {lines} ({e})")

    def stop(self):
        """
        メインディスプレイスレッドの停止と解放を行い、デバイスをクリーンアップする
        戻り値: なし
        """
        self._stop.set()
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.device:
            try:
                if hasattr(self.device, "cleanup"):
                    self.device.cleanup()
                elif hasattr(self.device, "clear"):
                    # 任意: clear だけしておく（必要なら）
                    self.device.clear()
                if DEBUG_MODE:
                    logging.debug("Device cleanup completed successfully")
            except Exception as e:
                logging.warning(f"Error during device cleanup: {e}")

        # ★ デバイスを完全に無効化して display_loop から参照されないようにする
        self.device = None
        self.oled_ok = False

        if getattr(self, "_owns_request_queue", False):
            try:
                if isinstance(self.request_queue, DisplayRequestQueue):
                    self.request_queue.stop()
            except Exception:
                logging.debug("Failed to stop request queue during OLED stop", exc_info=True)

    # 現在の表示データを辞書として退避する
    def _capture_display_snapshot(self) -> Dict[str, Any]:
        return self.display_data.copy() if isinstance(self.display_data, dict) else {}

    # 退避した表示データへ復帰する
    def _restore_display_snapshot(
        self, snapshot: Mapping[str, Any], latest_state: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Args:
            snapshot: 退避済みの表示データ。
            latest_state: 復帰時点での表示データを反映したい場合に指定する差分。
        """

        base = snapshot.copy() if isinstance(snapshot, Mapping) else {}
        if latest_state:
            try:
                base.update(latest_state)
            except Exception as exc:
                logging.exception("[OLED] Failed to merge latest display state into snapshot.")
                print(
                    f"[OLED][ERROR] Failed to merge latest display state into snapshot: {exc}"
                )
        return base

    # OLED全画面でメッセージを一時的に表示し、指定秒後に内容を復元する
    def display_message(
        self,
        message: Union[str, Sequence[str]],
        duration: float = 0.5,
        *,
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
        invert: bool = False,
    ):
        """
        OLED全画面でメッセージを一時的に表示し、指定秒後に内容を復元する
        非同期で動作し、呼び出し元をブロックしない
        """
        if isinstance(message, (list, tuple)):
            normalized_message: Union[str, List[str]] = [
                str(line) for line in message if line is not None
            ]
        else:
            normalized_message = str(message)

        with self.lock:
            backup = self._capture_display_snapshot()

            message_meta: Dict[str, Any] = {}
            if font_path:
                message_meta["_message_font_path"] = font_path
            if font_size is not None:
                message_meta["_message_font_size"] = font_size
            if invert:
                message_meta["_message_invert"] = True

            self.display_data = {"message": normalized_message, **message_meta}
            self.special_display_until = time.time() + duration
            self._last_activity_time = time.time()

            # メッセージ表示はユーザー操作扱いでセーバー解除
            if self._screensaver:
                self._screensaver = False
                self._exit_screensaver()
                # ★ スクリーンセーバー解除後にblink状態を復元
                if self._pre_screensaver_blink:
                    if isinstance(self.display_data, dict):
                        self.display_data["show_blink"] = True
                self._pre_screensaver_blink = None

        def restore_after_timeout():
            time.sleep(duration)
            with self.lock:
                latest_state: Dict[str, Any] = {}
                if isinstance(self.display_data, dict):
                        latest_state = {
                            key: value
                            for key, value in self.display_data.items()
                            if key
                            not in {
                                "message",
                                "_message_font_path",
                                "_message_font_size",
                                "_message_invert",
                            }
                        }

                if latest_state:
                    self.display_data = self._restore_display_snapshot(
                        backup, latest_state
                    )
                else:
                    self.display_data = self._restore_display_snapshot(backup)
                if hasattr(self, "special_display_until"):
                    del self.special_display_until

        # 非同期実行
        threading.Thread(target=restore_after_timeout, daemon=True).start()


    # バックグラウンドで永久ループし、現在の状態/内容に応じてOLED物理画面を更新し続ける
    def display_loop(self):
        """
        バックグラウンドで永久ループし、現在の状態/内容に応じてOLED物理画面を更新し続ける。

        - エラー時は自動で初期化再試行
        - 通常は画面設計に従い内容分岐描画
        - 'special_display_until' があれば優先的にメッセージを表示
        - show_blink指定時は点滅も表現
        - 無操作で減光、一定時間でスクリーンセーバーに移行。復帰時はコントラストを戻す
        - 周期補正を入れて「1秒ごとにtick」するよう調整
        - DEBUG_MODE 時に処理時間の100回平均をログ出力
        """

        # ===== 調整可能パラメータ =====
        screensaver_enabled_value = self.display_cfg.get("screensaver_enabled")
        if screensaver_enabled_value is None:
            SCREENSAVER_ENABLED = True
        elif isinstance(screensaver_enabled_value, bool):
            SCREENSAVER_ENABLED = screensaver_enabled_value
        else:
            SCREENSAVER_ENABLED = str(screensaver_enabled_value).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
                "enable",
                "enabled",
            }

        try:
            SCREEN_SAVER_TIMEOUT = int(
                self.display_cfg.get("screensaver_timeout_sec", 120 * 60)
            )
        except Exception:
            # 設定値が壊れている場合でも既定の 120 分にフォールバックする
            SCREEN_SAVER_TIMEOUT = 120 * 60

        # 極端に短いスクリーンセーバー設定は誤設定とみなし、最低 600 秒に丸める
        if SCREEN_SAVER_TIMEOUT < 60:
            SCREEN_SAVER_TIMEOUT = 600

        DIMMING_TIMEOUT = 5 * 60           # 5分無操作で減光
        try:
            DEFAULT_CONTRAST = int(self.display_cfg.get("contrast", 255))
        except Exception:
            DEFAULT_CONTRAST = 255
        DIMMING_CONTRAST = 0               # 減光時コントラスト（0で実質OFF）
        INTERVAL = 1.0                     # 更新周期 [秒]
        # =============================

        blink = True
        error_count = 0
        prev_display_data = None
        prev_error_msg = None
        last_blink_toggle = time.time()

        dimmed = False
        current_contrast = DEFAULT_CONTRAST

        next_tick = time.time()

        while not self._stop.is_set():
            loop_start = time.time()
            try:
                # 自動復帰方式
                if not getattr(self, "need_reinit", False) and not self.oled_ok:
                    self.need_reinit = True

                if getattr(self, "need_reinit", False):
                    try:
                        self._init_oled()
                        if not self.oled_ok:
                            error_count += 1
                            continue
                        error_count = 0
                        self.need_reinit = False
                        # デバイス復帰時はコントラスト初期化
                        if self.device and hasattr(self.device, "contrast"):
                            self.device.contrast(DEFAULT_CONTRAST)
                            current_contrast = DEFAULT_CONTRAST
                    except Exception as e:
                        logging.error(f"OLED loop error: {e}")
                        self.oled_ok = False
                        self.need_reinit = True
                        error_count += 1

                # デバイスが解放されたらループ終了
                if not self.device:
                    if self._stop.is_set():
                        break
                    time.sleep(INTERVAL)
                    continue

                # デバイス異常なら描画スキップ
                if not self.oled_ok:
                    continue

                now = time.time()

                # 共有データのスナップショット取得
                with self.lock:
                    d = self.display_data.copy() if isinstance(self.display_data, dict) else {}
                    special_time = getattr(self, "special_display_until", 0)
                    special_active = special_time and now < special_time
                    error_msg = self.error_msg
                    last_activity = self._last_activity_time
                    pending_restore = self._pending_screensaver_restore
                    pending_redraw = self._pending_redraw
                    if pending_redraw:
                        self._pending_redraw = False
                    screensaver_snapshot = (
                        dict(self._screensaver_snapshot)
                        if isinstance(self._screensaver_snapshot, dict)
                        else {}
                    )

                if special_active:
                    # 特別表示中はスクリーンセーバーのカウントをリセットする
                    with self.lock:
                        self._last_activity_time = now
                    last_activity = now

                # ===== スクリーンセーバー・減光 =====
                if (
                    SCREENSAVER_ENABLED
                    and SCREEN_SAVER_TIMEOUT > 0
                    and not self._screensaver
                ):
                    if last_activity and (now - last_activity > SCREEN_SAVER_TIMEOUT):
                        # ★ 現在の blink 状態を退避
                        with self.lock:
                            if isinstance(self.display_data, dict):
                                self._pre_screensaver_blink = self.display_data.get("show_blink", False)
                                self._screensaver_snapshot = self._capture_display_snapshot()
                        self._enter_screensaver()
                        self._screensaver = True
                        dimmed = False
                        if DEBUG_MODE:
                            print(f"[DEBUG] OLED screensaver ON")
                            logging.info("OLED screensaver ON")
                        continue

                if self._screensaver:
                    # スクリーンセーバー中は描画せず待機
                    continue

                if pending_restore:
                    if not d and screensaver_snapshot:
                        d = dict(screensaver_snapshot)
                    with self.lock:
                        self._pending_screensaver_restore = False

                if self.device and hasattr(self.device, "contrast") and DIMMING_TIMEOUT > 0:
                    if last_activity and (now - last_activity > DIMMING_TIMEOUT):
                        if not dimmed and current_contrast != DIMMING_CONTRAST:
                            self.device.contrast(DIMMING_CONTRAST)
                            current_contrast = DIMMING_CONTRAST
                            if DEBUG_MODE:
                                print(f"[DEBUG] OLED dim ON (contrast={DIMMING_CONTRAST})")
                                logging.info(f"OLED dim ON (contrast={DIMMING_CONTRAST})")
                        dimmed = True
                    else:
                        if dimmed and current_contrast != DEFAULT_CONTRAST:
                            self.device.contrast(DEFAULT_CONTRAST)
                            current_contrast = DEFAULT_CONTRAST
                            if DEBUG_MODE:
                                print(f"[DEBUG] OLED dim OFF (contrast restored)")
                                logging.info("OLED dim OFF (contrast restored)")
                        dimmed = False

                # ===== 点滅制御 =====
                show_blink_active = bool(d.get("show_blink", False))
                blink_state_changed = False

                if show_blink_active:
                    if now - last_blink_toggle >= 1.0:
                        blink = not blink
                        last_blink_toggle = now
                        blink_state_changed = True
                else:
                    if blink is not True:
                        blink = True
                        blink_state_changed = True
                    last_blink_toggle = now

                # ===== 内容変化判定 =====
                display_changed = (
                    pending_restore
                    or pending_redraw
                    or d != prev_display_data
                    or error_msg != prev_error_msg
                    or special_active
                    or blink_state_changed
                )
                if not display_changed:
                    # 内容変化なし → 描画スキップ
                    continue

                else:
                    # ===== 描画 =====
                    if Image is None or ImageDraw is None:
                        logging.debug("PIL not available; skipping render cycle")
                    else:
                        img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
                        draw = ImageDraw.Draw(img)

                        font_for_draw = self.font or self._load_font_object()
                        if font_for_draw is not None:
                            self.font = font_for_draw

                        if special_active and "message" in d:
                            invert_message = bool(d.get("_message_invert", False))
                            background_fill = 1 if invert_message else 0
                            text_fill = 0 if invert_message else 1
                            draw.rectangle(
                                (0, 0, OLED_WIDTH - 1, OLED_HEIGHT - 1), fill=background_fill #最後の数字で色変更
                            )

                            message_font = font_for_draw
                            if "_message_font_path" in d or "_message_font_size" in d:
                                message_font = self._load_message_font(
                                    d.get("_message_font_path"),
                                    d.get("_message_font_size"),
                                    font_for_draw,
                                )

                            message_obj = d.get("message", "")
                            if isinstance(message_obj, (list, tuple)):
                                message_lines = [
                                    str(line) for line in message_obj if line is not None
                                ]
                            else:
                                text_value = str(message_obj or "")
                                split_lines = text_value.splitlines()
                                message_lines = split_lines or [text_value]

                            if not message_lines:
                                message_lines = [""]

                            y = 12
                            for line in message_lines:
                                self._safe_draw_text(
                                    draw,
                                    (10, y),
                                    line,
                                    font=message_font,
                                    fill=text_fill, #最後の数字で色変更
                                )
                                try:
                                    bbox = draw.textbbox((10, y), line, font=message_font)
                                    line_height = bbox[3] - bbox[1]
                                except Exception:
                                    if message_font is not None:
                                        line_height = message_font.getsize(line)[1]
                                    else:
                                        line_height = 12
                                y += max(line_height + 2, 14)

                        elif error_msg:
                            background_fill = 1
                            text_fill = 0
                            draw.rectangle(
                                (0, 0, OLED_WIDTH - 1, OLED_HEIGHT - 1), fill=background_fill
                            )
                            self._safe_draw_text(
                                draw, (3, 22), "ERROR:", font=font_for_draw, fill=text_fill
                            )
                            self._safe_draw_text(
                                draw,
                                (3, 32),
                                str(error_msg)[:15],
                                font=font_for_draw,
                                fill=text_fill,
                            )
                            self._safe_draw_text(
                                draw,
                                (3, 42),
                                f"({self.connection_type})",
                                font=font_for_draw,
                                fill=text_fill,
                            )

                        else:
                            s = d.get("status_lcd") or d.get("status", "作業時間")
                            timer = d.get("timer", "00:00")

                            if d.get("show_rework", False):
                                s = "* 手直し"
                                blink_char = "□" if d.get("show_blink", False) and blink else "　"
                            else:
                                blink_char = "■" if d.get("show_blink", False) and blink else "　"

                            self._safe_draw_text(
                                draw, (0, 0), d.get("process_lcd", ""), font=font_for_draw, fill=1
                            )
                            self._safe_draw_text(
                                draw,
                                (0, 20),
                                f"{(d.get('check_no_lcd') or '      ')} |{d.get('worker_lcd', '')}",
                                font=font_for_draw,
                                fill=1,
                            )
                            self._safe_draw_text(
                                draw,
                                (0, 42),
                                f"{s} {blink_char} {timer}",
                                font=font_for_draw,
                                fill=1,
                            )

                        # 画像が生成された場合のみログ・表示を実行
                        if DEBUG_MODE and "img" in locals():
                            # print(f"[DEBUG] device={self.device}, img={type(img)}")
                            logging.debug(f"device={self.device}, img={type(img)}")

                        if self.device and "img" in locals():
                            self.device.display(img)

                # 内容に関わらず、ここで prev_* を更新（try の中でOK）
                prev_display_data = d.copy()
                prev_error_msg = error_msg

                if DEBUG_MODE and error_count % 5 == 0:
                   # print(f"[DEBUG] OLED status: {self.debug_info()}")
                    logging.debug(f"OLED status: {self.debug_info()}")

            except Exception as e:
                logging.exception(f"OLED loop error: {e}")
                self.oled_ok = False
                self.need_reinit = True
                error_count += 1

            # ===== 周期補正付きsleep =====（try-except の外側に配置）
            next_tick += INTERVAL
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 遅延が積み重なった場合は即次ループ
                next_tick = time.time()


    # アニメーション処理
    def display_animation(
        self,
        frames_dir,
        default_frame_time=0.10,
        meta_json="frame_times.json",
        *,
        block: bool = False,
        min_frame_time: Optional[float] = None,
        max_frame_time: Optional[float] = None,
    ):
        """
        SSD1309で指定ディレクトリの連番PNGアニメーションを表示
        読み出し時に全フレームをメモリへ事前ロードしてから再生する。
        """
        if Image is None:
            logging.error("PIL is not available; cannot play animations")
            return

        base_frame_time = parse_float(default_frame_time, default=0.10)
        if base_frame_time is None:
            base_frame_time = 0.10

        min_frame = parse_float(min_frame_time, default=0.01)
        if min_frame is None:
            min_frame = 0.01

        max_frame = parse_float(max_frame_time, default=None)

        png_files, frame_times, _ = _resolve_frame_timings(
            frames_dir,
            meta_json=meta_json,
            default_frame_time=base_frame_time,
            min_frame_time=min_frame,
            max_frame_time=max_frame,
        )
        if not png_files:
            return

        wait_time = sum(frame_times) if frame_times else 0.0

        def _play_animation() -> None:
            try:
                # 3) 全フレーム事前ロード（必要なら1bitへ変換）
                frames = []
                convert_kwargs = {}
                if Image is not None:
                    # ディザ無効化で余計な計算を避け、読み込み性能を確保する
                    try:
                        convert_kwargs["dither"] = Image.NONE
                    except AttributeError as exc:
                        # Pillowが古い場合に備えて警告を残し、処理自体は継続する
                        logging.debug("Pillow does not support Image.NONE: %s", exc)
                        pass

                target_mode = "1"
                if Image is not None:
                    device_for_mode = getattr(self, "device", None)
                    mode_from_device = getattr(device_for_mode, "mode", None)
                    if mode_from_device:
                        target_mode = mode_from_device

                for idx, png_path in enumerate(png_files):
                    try:
                        with Image.open(png_path) as im:
                            # Pillowの遅延読み込みを明示的に完了させ、I/O待ちを再生前に済ませる
                            im.load()
                            img = None

                            if Image is not None:
                                if target_mode == "1":
                                    if im.mode == "1":
                                        img = im.copy()
                                    else:
                                        repacked = _repack_palette_1bit(im)
                                        if repacked is not None:
                                            img = repacked
                                elif im.mode == target_mode:
                                    img = im.copy()

                            if img is None:
                                if Image is not None and target_mode and im.mode != target_mode:
                                    try:
                                        img = im.convert(target_mode, **convert_kwargs)
                                    except Exception:
                                        img = im.copy()
                                else:
                                    img = im.copy()
                            frames.append(img)
                    except Exception as e:
                        logging.error(f"Failed to load frame {png_path}: {e}")
                        frames.append(None)

                # 読み込み成功が1枚もなければ終了
                if not any(f is not None for f in frames):
                    logging.error("No valid frames could be loaded.")
                    return

                # 4) 再生（I/Oなし）
                device = self.device
                if device is None:
                    logging.error("No OLED device available for animation playback.")
                    return

                display_fn = getattr(device, "display", None)
                if not callable(display_fn):
                    logging.error("OLED device does not provide a callable 'display' method.")
                    return

                frame_times_local = list(frame_times)
                # frame_times は常にフレーム数と同じ長さになるとは限らないため調整する
                if len(frame_times_local) < len(frames):
                    frame_times_local = frame_times_local + [base_frame_time] * (
                        len(frames) - len(frame_times_local)
                    )
                elif len(frame_times_local) > len(frames):
                    frame_times_local = frame_times_local[: len(frames)]

                perf_counter = time.perf_counter
                sleeper = time.sleep

                next_due = perf_counter()
                for idx, (img, duration) in enumerate(zip(frames, frame_times_local)):
                    if img is None:
                        next_due = perf_counter() + duration
                        continue

                    # 目標時刻まで待機（余分な待機やディスプレイ処理時間を補正）
                    now = perf_counter()
                    sleep_time = next_due - now
                    if sleep_time > 0:
                        sleeper(sleep_time)

                    try:
                        display_fn(img)
                    except Exception as e:
                        logging.error(
                            f"Failed to display preloaded frame index={idx}: {e}"
                        )
                    finally:
                        # display() 呼び出しで遅延が発生しても次の表示間隔を維持
                        next_due = max(next_due + duration, perf_counter())

                # 5) 必要なら既定画面復帰
            finally:
                self._mark_redraw_needed()

        if block:
            _play_animation()
            return

        if self.submit_display_request(
            _play_animation,
            priority=DisplayRequestQueue.PRIORITY_ANIMATION,
            wait=wait_time,
            label=f"animation:{frames_dir}",
        ):
            return

        _play_animation()


    # ペア成立時のアニメーション表示
    def play_pair_animation(
        self,
        frames_dir: Optional[str] = None,
        frame_time: float = 0.08,
        meta_json: str = "frame_times.json",
    ):
        """
        ペア成立時のアニメーション表示
        - frames_dir: ペア成立アニメPNGディレクトリ。未指定時は設定/既定パスから解決。
        """
        # frames_dir 解決: 引数 > self.display_cfg['pair_animation_dir'] > 既定

        cfg_dir_raw = self.display_cfg.get("pair_animation_dir") if hasattr(self, "display_cfg") else None
        if isinstance(cfg_dir_raw, str):
            cfg_dir = cfg_dir_raw.strip() or None
        elif cfg_dir_raw is not None:
            cfg_dir = str(cfg_dir_raw).strip() or None
        else:
            cfg_dir = None

        candidate = frames_dir or cfg_dir or DEFAULT_PAIR_ANIM_DIR
        anim_dir = os.path.abspath(candidate)

        # 存在確認
        if not os.path.isdir(anim_dir):
            logging.warning(
                f"pair_animation: directory not found: {anim_dir} (specify frames_dir explicitly)"
            )
            return

        # フレーム存在の簡易確認（display_animationでも検証するが、ここでも案内用にチェック可）
        try:
            entries = os.listdir(anim_dir)
        except Exception as e:
            logging.error(f"pair_animation: failed to list directory '{anim_dir}': {e}")
            return
        if not any(f.lower().endswith(".png") for f in entries):
            logging.warning(f"pair_animation: no PNG frames in {anim_dir}")
            return

        # フレームタイムメタデータの存在を確認し、起動時アニメと同様に利用する
        meta_path: Optional[str]
        if meta_json:
            resolved_meta = meta_json if os.path.isabs(meta_json) else os.path.join(anim_dir, meta_json)
            if os.path.isfile(resolved_meta):
                meta_path = resolved_meta
            else:
                logging.warning(
                    "pair_animation: frame timing metadata not found: %s",
                    resolved_meta,
                )
                meta_path = None
        else:
            meta_path = None

        # 既存の display_animation を利用（元の挙動を完全踏襲）
        if meta_path:
            self.display_animation(
                anim_dir, default_frame_time=frame_time, meta_json=meta_path
            )
        else:
            self.display_animation(
                anim_dir, default_frame_time=frame_time
            )

    # スクリーンセーバー（消灯/低コントラスト）状態へ移行する
    def _enter_screensaver(self):
        if not self.device:
            return
        try:
            cs = int(self.display_cfg.get("screensaver_contrast", 0))
            if hasattr(self.device, "contrast"):
                self.device.contrast(max(0, min(255, cs)))

            # 真っ黒化（PILがある場合は画像で、ない場合はclear()があれば呼ぶ）
            if Image is not None:
                try:
                    img = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
                    self.device.display(img)
                except Exception as ie:
                    logging.debug(f"Screensaver image draw failed, fallback to clear(): {ie}")
                    if hasattr(self.device, "clear"):
                        self.device.clear()
            else:
                if hasattr(self.device, "clear"):
                    self.device.clear()

            if DEBUG_MODE:
                logging.info("OLED screensaver ON")
        except Exception as e:
            logging.warning(f"Failed to enter screensaver: {e}")

    # スクリーンセーバー状態から復帰し、コントラストを既定値へ戻す
    def _exit_screensaver(self):
        if not self.device:
            return
        try:
            contrast = int(self.display_cfg.get("contrast", 255))
            if hasattr(self.device, "contrast"):
                self.device.contrast(max(0, min(255, contrast)))
            if DEBUG_MODE:
                print(f"[DEBUG] OLED screensaver OFF")
                logging.info("OLED screensaver OFF")
        except Exception as e:
            logging.warning(f"Failed to exit screensaver: {e}")

    def debug_info(self):
        """
        デバッグ情報を辞書形式で返す
        """
        info = {
            "connection_type": self.connection_type,
            "status": "connected" if self.oled_ok else "disconnected",
            "error": self.error_msg,
            "display_data_keys": list(self.display_data.keys())
            if self.display_data
            else None,
            "thread_alive": self.thread.is_alive() if hasattr(self, "thread") else None,
            "device_type": type(self.device).__name__ if self.device else None,
            "font_path": OLED_FONT_PATH,
            "font_exists": os.path.exists(OLED_FONT_PATH),
        }
        return info


# OLEDが使えない場合のダミークラス（テスト・デバッグ用）
class DummyLCD:
    """
    OLED未接続もしくはライブラリ未導入環境用のダミーLCD。
    画面の代わりにprintで動作内容を通知。
    OledDisplayManagerと同じインターフェースを提供し、互換性を確保。
    """

    # ダミーLCDの初期化
    def __init__(
        self,
        connection_type: Optional[str] = None,
        display_cfg: Optional[Mapping[str, Any]] = None,
        led_pin: Optional[int] = None,
        request_queue: Optional["DisplayRequestQueue"] = None,
    ):
        self.connection_type = connection_type or OLED_CONNECTION_TYPE
        self.display_data = {}
        self.error_msg = None
        self.stopped = False
        self._screensaver = False
        self._last_activity_time = time.time()
        self.display_cfg = dict(DEFAULT_DISPLAY_CONFIG)
        if display_cfg:
            self.display_cfg.update(display_cfg)
        self._pending_screensaver_restore = False
        self._led_pin = self._resolve_led_pin(led_pin)
        self._led_ready = False
        self._led_state = False

        self.request_queue = request_queue or DisplayRequestQueue("dummy-display")
        self._owns_request_queue = request_queue is None

        if self._led_pin is not None:
            self._setup_led_pin()

        logging.info(f"[DummyLCD] Initialized with {self.connection_type} mode")
        if DEBUG_MODE:
            print(
                f"[DummyLCD] DEBUG MODE ON - initialized with {self.connection_type} mode"
            )

    def _exit_screensaver(self) -> None:
        """スクリーンセーバー終了時のダミー処理。"""
        contrast = int(self.display_cfg.get("contrast", DEFAULT_DISPLAY_CONFIG["contrast"]))
        logging.debug(
            "[DummyLCD] screensaver exit (restore contrast=%s)",
            max(0, min(255, contrast)),
        )

    # コンテキストマネージャーのサポート
    def __enter__(self):
        return self

    # コンテキストマネージャー終了時のクリーンアップ
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def submit_display_request(
        self,
        action: Callable[[], None],
        *,
        priority: Optional[int] = None,
        wait: Optional[float] = None,
        label: Optional[str] = None,
    ) -> bool:
        resolved_priority = (
            priority if priority is not None else DisplayRequestQueue.PRIORITY_OVERLAY
        )
        wait_time = parse_float(wait, default=0.0) if wait is not None else 0.0
        if wait_time is None:
            wait_time = 0.0

        queue = getattr(self, "request_queue", None)
        if isinstance(queue, DisplayRequestQueue):
            return queue.submit(
                action, priority=resolved_priority, wait=wait_time, label=label or ""
            )

        try:
            action()
            if wait_time:
                time.sleep(max(0.0, float(wait_time)))
            return True
        except Exception:
            logging.debug("Display request execution failed (no queue).", exc_info=True)
            return False


    def _resolve_led_pin(self, led_pin: Optional[int]) -> Optional[int]:
        if not GPIO_AVAILABLE:
            return None

        if led_pin is not None:
            try:
                return int(led_pin)
            except (TypeError, ValueError):
                logging.warning(f"[DummyLCD] invalid led_pin specified: {led_pin}")
                return None

        env_pin = os.environ.get("PRODTRAC_LED_PIN")
        if env_pin:
            try:
                return int(env_pin)
            except ValueError:
                logging.warning(f"[DummyLCD] invalid PRODTRAC_LED_PIN value: {env_pin}")
        return None

    def _setup_led_pin(self) -> None:
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._led_pin, GPIO.OUT)
            GPIO.output(self._led_pin, GPIO.LOW)
            self._led_ready = True
            logging.info(f"[DummyLCD] LED indicator enabled on pin {self._led_pin}")
        except Exception as exc:
            logging.warning(f"[DummyLCD] Failed to initialize LED pin {self._led_pin}: {exc}")
            self._led_pin = None
            self._led_ready = False

    def led_on(self) -> None:
        if not self._led_ready:
            return
        try:
            GPIO.output(self._led_pin, GPIO.HIGH)
            self._led_state = True
        except Exception as exc:
            logging.debug(f"[DummyLCD] led_on failed: {exc}")

    def led_off(self) -> None:
        if not self._led_ready:
            return
        try:
            GPIO.output(self._led_pin, GPIO.LOW)
            self._led_state = False
        except Exception as exc:
            logging.debug(f"[DummyLCD] led_off failed: {exc}")

    def update_work_indicator(self, is_working: bool) -> None:
        if not self._led_ready:
            return
        if is_working:
            if not self._led_state:
                self.led_on()
        else:
            if self._led_state:
                self.led_off()

    def _sync_work_indicator(self, payload: Mapping[str, Any]) -> None:
        if not self._led_ready:
            return
        if "show_blink" not in payload:
            return
        try:
            self.update_work_indicator(bool(payload.get("show_blink")))
        except Exception:
            logging.debug("[DummyLCD] failed to sync work indicator", exc_info=True)

    # 表示内容データを受け取り、画面の代わりにログに出力
    def update(self, **kwargs):
        self.display_data.update(kwargs)
        self._last_activity_time = time.time()
        if self._screensaver:
            self._screensaver = False
            self._exit_screensaver()
            self._pending_screensaver_restore = True
        self._sync_work_indicator(kwargs)
        if DEBUG_MODE:
            print(f"[LCD] Update: {kwargs}")
        else:
            logging.debug(f"[LCD] Update: {kwargs}")

    def force_screensaver_on(self) -> None:
        """QRコードなどでの即時スクリーンセーバー指示に対応するダミー処理。"""

        self._screensaver = True
        self._last_activity_time = time.time()
        if DEBUG_MODE:
            logging.info("[DummyLCD] screensaver ON (QR trigger)")

    def force_screensaver_off(self) -> None:
        """ダミー環境でのスクリーンセーバー解除。"""

        if not self._screensaver:
            return

        self._screensaver = False
        self._last_activity_time = time.time()
        self._pending_screensaver_restore = True
        if DEBUG_MODE:
            logging.info("[DummyLCD] screensaver OFF (QR trigger)")

    # Dummy環境向けのアニメーション表示。
    def display_animation(
        self,
        frames_dir,
        default_frame_time=0.10,
        meta_json="frame_times.json",
        *,
        block: bool = False,
        min_frame_time: Optional[float] = None,
        max_frame_time: Optional[float] = None,
    ):
        """Dummy環境向けのアニメーション表示。"""
        base_frame_time = parse_float(default_frame_time, default=0.10)
        if base_frame_time is None:
            base_frame_time = 0.10

        min_frame = parse_float(min_frame_time, default=0.01)
        if min_frame is None:
            min_frame = 0.01

        max_frame = parse_float(max_frame_time, default=None)

        png_files, frame_times, meta_path = _resolve_frame_timings(
            frames_dir,
            meta_json=meta_json,
            default_frame_time=base_frame_time,
            min_frame_time=min_frame,
            max_frame_time=max_frame,
        )
        if not png_files:
            return

        wait_time = sum(frame_times) if frame_times else 0.0

        def _play_dummy_animation() -> None:
            logging.info(
                f"[DummyLCD] Playing animation from {frames_dir} using timings {meta_path or meta_json}"
            )
            frame_times_local = list(frame_times)
            if len(frame_times_local) < len(png_files):
                frame_times_local = frame_times_local + [base_frame_time] * (
                    len(png_files) - len(frame_times_local)
                )
            elif len(frame_times_local) > len(png_files):
                frame_times_local = frame_times_local[: len(png_files)]

            for idx, png_path in enumerate(png_files):
                delay = (
                    frame_times_local[idx]
                    if idx < len(frame_times_local)
                    else base_frame_time
                )
                logging.debug(
                    f"[DummyLCD] Display frame {idx + 1}/{len(png_files)}: {png_path}"
                )
                time.sleep(max(0.0, float(delay)))

        if block:
            _play_dummy_animation()
            return

        if self.submit_display_request(
            _play_dummy_animation,
            priority=DisplayRequestQueue.PRIORITY_ANIMATION,
            wait=wait_time,
            label=f"dummy-animation:{frames_dir}",
        ):
            return

        _play_dummy_animation()


    # エラーメッセージをログに出力
    def show_error(self, msg: Union[str, Sequence[str]], duration: Optional[float] = None, *, invert: bool = True):
        lines: list[str]
        if isinstance(msg, (list, tuple)):
            lines = [str(line) for line in msg if line is not None]
        else:
            lines = [str(msg)]

        rendered = " / ".join(lines)
        self.error_msg = rendered
        logging.error(f"[LCD][ERR] {rendered}")
        if DEBUG_MODE:
            print(f"[LCD][ERR] {rendered}")

    # 停止処理（リソース解放はないがインターフェース互換性のため）
    def stop(self):
        self.stopped = True
        try:
            self.led_off()
        except Exception:
            logging.debug("[DummyLCD] failed to turn off LED on stop", exc_info=True)
        if DEBUG_MODE:
            print("[LCD] Stopped")
        logging.debug("[LCD] Stopped")

        if getattr(self, "_owns_request_queue", False):
            try:
                if isinstance(self.request_queue, DisplayRequestQueue):
                    self.request_queue.stop()
            except Exception:
                logging.debug("Failed to stop request queue during Dummy stop", exc_info=True)

