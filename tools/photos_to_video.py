#!/usr/bin/env python3
"""Local photo-to-video renderer.

Bu araç API kullanmadan, render işlemini tamamen bilgisayarda yapar.
Bir klasördeki fotoğrafları sıralar ve her fotoğrafı belirlenen süre kadar gösteren
MP4 video üretir.

Örnek:
  python tools/photos_to_video.py --input photos --output outputs/video.mp4 --duration 3 --size 1080x1920
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps

try:
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
except Exception:  # pragma: no cover - sadece kullanıcıya net hata vermek için
    AudioFileClip = None
    ImageClip = None
    concatenate_videoclips = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg"}


def natural_sort_key(path: Path) -> list[object]:
    """1.jpg, 2.jpg, 10.jpg sıralamasını doğru yapar."""
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Fotoğraf klasörü bulunamadı: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Bu yol klasör değil: {input_dir}")

    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=natural_sort_key)
    if not images:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"Klasörde fotoğraf bulunamadı. Desteklenen uzantılar: {allowed}")
    return images


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("Boyut formatı 1080x1920 gibi olmalı.")
    width, height = int(match.group(1)), int(match.group(2))
    if width < 240 or height < 240:
        raise argparse.ArgumentTypeError("Genişlik/yükseklik en az 240 olmalı.")
    return width, height


def normalize_image(src: Path, dst: Path, size: tuple[int, int], fit: str) -> None:
    """Fotoğrafı hedef video oranına göre crop/pad ederek JPEG olarak kaydeder."""
    width, height = size
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if fit == "contain":
            # Fotoğraf kırpılmaz; boş alanlar siyah dolgu olur.
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            x = (width - image.width) // 2
            y = (height - image.height) // 2
            canvas.paste(image, (x, y))
            canvas.save(dst, quality=94, optimize=True)
            return

        # Varsayılan: cover. Video oranına doldurur, kenarlardan kırpabilir.
        fitted = ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        fitted.save(dst, quality=94, optimize=True)


def create_video(
    images: Sequence[Path],
    output: Path,
    duration: float,
    fps: int,
    size: tuple[int, int],
    fit: str,
    audio: Path | None,
) -> None:
    if ImageClip is None or concatenate_videoclips is None:
        raise RuntimeError(
            "moviepy yüklü değil. Önce proje klasöründe `uv sync --frozen` veya "
            "`pip install -r requirements.txt` çalıştır."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mpt_photo_video_") as temp_dir:
        temp_path = Path(temp_dir)
        normalized_images: list[Path] = []
        for index, image_path in enumerate(images, start=1):
            dst = temp_path / f"frame_{index:05d}.jpg"
            normalize_image(image_path, dst, size=size, fit=fit)
            normalized_images.append(dst)

        clips = [ImageClip(str(img)).with_duration(duration) for img in normalized_images]
        final_clip = concatenate_videoclips(clips, method="compose")

        if audio:
            if AudioFileClip is None:
                raise RuntimeError("moviepy AudioFileClip yüklenemedi.")
            if not audio.exists():
                raise FileNotFoundError(f"Müzik dosyası bulunamadı: {audio}")
            if audio.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError("Desteklenmeyen müzik dosyası. mp3, wav, m4a, aac veya ogg kullan.")
            music = AudioFileClip(str(audio))
            # MoviePy v2 için güvenli süre kısaltma.
            if getattr(music, "duration", 0) and music.duration > final_clip.duration:
                music = music.subclipped(0, final_clip.duration)
            final_clip = final_clip.with_audio(music)

        final_clip.write_videofile(
            str(output),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=max(1, (os.cpu_count() or 2) - 1),
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

        final_clip.close()
        for clip in clips:
            clip.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fotoğrafları lokal MP4 videoya dönüştürür.")
    parser.add_argument("--input", "-i", default="photos", help="Fotoğraf klasörü. Varsayılan: photos")
    parser.add_argument("--output", "-o", default="outputs/photo-video.mp4", help="Çıkış MP4 dosyası.")
    parser.add_argument("--duration", "-d", type=float, default=3.0, help="Her fotoğraf kaç saniye görünsün. Örn: 3 veya 5")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS. Varsayılan: 30")
    parser.add_argument("--size", type=parse_size, default=(1080, 1920), help="Video boyutu. Shorts için 1080x1920")
    parser.add_argument("--fit", choices=["cover", "contain"], default="cover", help="cover: kırpar/doldurur, contain: kırpmaz siyah dolgu yapar")
    parser.add_argument("--audio", help="İsteğe bağlı arka plan müziği: mp3/wav/m4a/aac/ogg")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration 0'dan büyük olmalı.")
    if args.fps < 10 or args.fps > 60:
        parser.error("--fps 10 ile 60 arasında olmalı.")

    input_dir = Path(args.input).resolve()
    output = Path(args.output).resolve()
    audio = Path(args.audio).resolve() if args.audio else None

    images = find_images(input_dir)
    total_duration = len(images) * args.duration
    print(f"Fotoğraf sayısı: {len(images)}")
    print(f"Her fotoğraf: {args.duration:.2f} sn")
    print(f"Toplam video süresi: {total_duration:.2f} sn")
    print(f"Boyut: {args.size[0]}x{args.size[1]} | FPS: {args.fps} | Fit: {args.fit}")
    print(f"Çıkış: {output}")

    create_video(
        images=images,
        output=output,
        duration=args.duration,
        fps=args.fps,
        size=args.size,
        fit=args.fit,
        audio=audio,
    )
    print("Bitti ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
