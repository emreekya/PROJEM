import glob
import itertools
import io
import os
import random
import gc
import shutil
import subprocess
from contextlib import redirect_stdout
from typing import List
from loguru import logger
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageFont

from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.config import config
from app.services.utils import video_effects
from app.utils import file_security, utils

class SubClippedVideoClip:
    def __init__(self, file_path, start_time=None, end_time=None, width=None, height=None, duration=None):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
# Docker 里的 ffmpeg/AAC 组合在默认配置下更容易出现音频质量波动，
# 这里显式抬高音频码率，避免成片阶段因为默认值过低而引入明显失真。
audio_bitrate = "192k"
video_codec = "libx264"
fps = 30
_BGM_EXTENSIONS = (".mp3",)


def get_ffmpeg_binary():
    # 优先复用用户在 config.toml / 环境变量里显式指定的 ffmpeg，可避免
    # Windows 便携包、Docker、自定义安装目录等场景下 PATH 不一致。
    configured_ffmpeg = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured_ffmpeg:
        return configured_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled_ffmpeg:
            return bundled_ffmpeg
    except Exception as exc:
        logger.warning(f"failed to resolve bundled ffmpeg binary: {str(exc)}")

    return "ffmpeg"


def _escape_ffmpeg_concat_path(file_path: str) -> str:
    # concat demuxer 使用单引号包裹路径，路径中的单引号需要先转义。
    return file_path.replace("'", "'\\''")


def concat_video_clips_with_ffmpeg(
    clip_files: List[str], output_file: str, threads: int, output_dir: str
):
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as fp:
        for clip_file in clip_files:
            absolute_path = os.path.abspath(clip_file)
            fp.write(f"file '{_escape_ffmpeg_concat_path(absolute_path)}'\n")

    command = [
        get_ffmpeg_binary(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_file,
        "-c:v",
        video_codec,
        "-threads",
        str(threads or 2),
        "-pix_fmt",
        "yuv420p",
        output_file,
    ]

    try:
        # 使用 ffmpeg 只做一次串联与编码，避免 MoviePy 逐段合并时反复重编码，
        # 从而降低画质劣化与颜色偏移风险。
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error_message or "ffmpeg concat failed")
    finally:
        delete_files(concat_list_file)


def _sanitize_image_file(image_path: str) -> str:
    # 某些本地图片虽然能被 Pillow 打开，但会因为损坏的 EXIF/eXIf 元数据导致
    # ImageClip 在解析阶段直接抛异常。这里重新导出一份“干净图片”，把坏元数据剥离掉。
    image_root, _ = os.path.splitext(image_path)
    sanitized_path = f"{image_root}.sanitized.png"

    with Image.open(image_path) as image:
        image.load()
        # 统一导出为 PNG，避免 JPEG/PNG 不同元数据路径继续把坏块带过去。
        cleaned_image = Image.new(image.mode, image.size)
        cleaned_image.putdata(list(image.getdata()))
        cleaned_image.save(sanitized_path)

    return sanitized_path


def _open_image_clip_with_fallback(image_path: str):
    # 优先直接打开原始图片；如果因为损坏元数据失败，再尝试生成无元数据副本。
    try:
        return ImageClip(image_path), image_path
    except Exception as exc:
        logger.warning(
            f"failed to open image directly, trying sanitized copy: {image_path}, error: {str(exc)}"
        )
        sanitized_path = _sanitize_image_file(image_path)
        return ImageClip(sanitized_path), sanitized_path


def _open_video_clip_quietly(video_path: str, audio: bool = False) -> VideoFileClip:
    """
    安静地打开视频文件，避免 MoviePy 2.1.x 把 ffmpeg 探测信息直接打印到 stdout。

    背景：
    当前依赖版本的 `FFMPEG_VideoReader` 内部存在 `print(self.infos)` 和
    `print(ffmpeg command)`，读取无音轨的中间视频时会输出
    `audio_found: False`。这只是输入素材 metadata，不代表最终成片没有音频，
    但会误导 WebUI/终端用户以为生成失败。

    实现：
    1. 只在打开 VideoFileClip 的短窗口内重定向 stdout；
    2. 默认 `audio=False`，因为项目视频素材阶段不需要保留素材原声，
       最终音频会在 `generate_video()` 阶段统一挂载；
    3. 如果依赖库确实输出了内容，降级为 debug 日志，便于必要时排查。
    """
    captured_stdout = io.StringIO()
    with redirect_stdout(captured_stdout):
        clip = VideoFileClip(video_path, audio=audio)

    moviepy_stdout = captured_stdout.getvalue().strip()
    if moviepy_stdout:
        logger.debug(
            "suppressed MoviePy video reader stdout for "
            f"{video_path}, chars: {len(moviepy_stdout)}"
        )

    return clip


def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            logger.debug(f"failed to delete file {file}: {str(e)}")


def _resolve_bgm_file_path(song_dir: str, bgm_file: str) -> str:
    # 背景音乐只允许读取 resource/songs 目录内的文件，避免用户输入任意路径后
    # 被 MoviePy 打开。这里兼容两种常见输入：
    # 1. output000.mp3：来自 BGM 列表或用户只填写文件名
    # 2. ./resource/songs/output000.mp3：用户按项目目录结构填写的相对路径
    # 两种写法最终都会再次通过 resource/songs 白名单校验，不能绕过目录限制。
    try:
        return file_security.resolve_path_within_directory(song_dir, bgm_file)
    except ValueError as song_dir_exc:
        if os.path.isabs(bgm_file):
            raise song_dir_exc

        project_relative_file = os.path.join(utils.root_dir(), bgm_file)
        try:
            return file_security.resolve_path_within_directory(
                song_dir, project_relative_file
            )
        except ValueError as root_dir_exc:
            raise ValueError(str(root_dir_exc)) from song_dir_exc


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file:
        song_dir = utils.song_dir()
        try:
            resolved_bgm_file = _resolve_bgm_file_path(song_dir, bgm_file)
        except ValueError as exc:
            # API 请求里的 bgm_file 来自用户输入，不能直接把任意绝对路径交给
            # MoviePy 打开。这里强制限制到 resource/songs 目录，阻止读取
            # /etc/passwd、配置文件、密钥等非背景音乐文件。
            logger.warning(
                f"reject unsafe bgm file: {bgm_file}, song_dir: {song_dir}, error: {str(exc)}"
            )
            return ""

        if not resolved_bgm_file.lower().endswith(_BGM_EXTENSIONS):
            logger.warning(f"reject unsupported bgm file extension: {resolved_bgm_file}")
            return ""

        return resolved_bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        # 当背景音乐目录为空时，直接回退为“不使用 BGM”，避免 random.choice([]) 抛异常。
        if not files:
            logger.warning(f"no bgm files found in song directory: {song_dir}")
            return ""
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    try:
        # 这里只需要读取旁白音频时长来决定素材视频拼接长度；后续不会再使用
        # audio_clip。读取完成后立即关闭，避免早退或异常路径泄漏文件句柄。
        audio_duration = audio_clip.duration
    finally:
        close_clip(audio_clip)
    logger.info(f"audio duration: {audio_duration} seconds")
    logger.info(f"maximum clip duration: {max_clip_duration} seconds")

    # 兼容 API 直接调用时未传转场模式的情况，避免后续访问 .value 时崩溃。
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = _open_video_clip_quietly(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)
        
        start_time = 0

        while start_time < clip_duration:
            end_time = min(start_time + max_clip_duration, clip_duration)

            # 保留所有有效分段。
            # 这样既不会丢掉“整段视频本身就短于 max_clip_duration”的素材，
            # 也不会吞掉长视频最后剩下的一小段尾部内容。
            if end_time > start_time:
                subclipped_items.append(
                    SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=clip_w,
                        height=clip_h,
                    )
                )

            start_time = end_time
            if video_concat_mode.value == VideoConcatMode.sequential.value:
                break

    # random subclipped_items order
    if video_concat_mode.value == VideoConcatMode.random.value:
        random.shuffle(subclipped_items)
        
    logger.debug(f"total subclipped items: {len(subclipped_items)}")
    
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration > audio_duration:
            break
        
        logger.debug(f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, current duration: {video_duration:.2f}s, remaining: {audio_duration - video_duration:.2f}s")
        
        try:
            clip = _open_video_clip_quietly(subclipped_item.file_path).subclipped(
                subclipped_item.start_time, subclipped_item.end_time
            )
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {video_width}x{video_height}, ratio: {video_ratio:.2f}")
                
                if clip_ratio == video_ratio:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    # Çerçeveyi tamamen DOLDUR (kırp), siyah bant bırakma:
                    # hedefi kapsayacak şekilde ölçekle, sonra ortadan kırp.
                    if clip_ratio > video_ratio:
                        scale_factor = video_height / clip_h
                    else:
                        scale_factor = video_width / clip_w

                    new_width = int(clip_w * scale_factor)
                    new_height = int(clip_h * scale_factor)

                    clip_resized = clip.resized(new_size=(new_width, new_height))
                    clip = clip_resized.cropped(
                        x_center=new_width / 2,
                        y_center=new_height / 2,
                        width=video_width,
                        height=video_height,
                    )
                    
            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if transition_value in (None, VideoTransitionMode.none.value):
                clip = clip
            elif transition_value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif transition_value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif transition_value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif transition_value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)
                
            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            clip.write_videofile(clip_file, logger=None, fps=fps, codec=video_codec)

            # Store clip duration before closing
            clip_duration_saved = clip.duration
            close_clip(clip)

            processed_clips.append(SubClippedVideoClip(file_path=clip_file, duration=clip_duration_saved, width=clip_w, height=clip_h))
            video_duration += clip_duration_saved
            
        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")
    
    # loop processed clips until the video duration matches or exceeds the audio duration.
    if video_duration < audio_duration:
        logger.warning(f"video duration ({video_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s), looping clips to match audio length.")
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= audio_duration:
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, looped {len(processed_clips)-len(base_clips)} clips")
     
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path
    
    # if there is only one clip, use it directly
    if len(processed_clips) == 1:
        logger.info("using single clip directly")
        shutil.copy(processed_clips[0].file_path, combined_video_path)
        delete_files([processed_clips[0].file_path])
        logger.info("video combining completed")
        return combined_video_path

    clip_files = [clip.file_path for clip in processed_clips]
    logger.info(f"concatenating {len(clip_files)} clips with ffmpeg")
    concat_video_clips_with_ffmpeg(
        clip_files=clip_files,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
    )
    
    # clean temp files
    delete_files(clip_files)
            
    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # 字幕换行必须在真正创建 TextClip 前完成，否则 MoviePy 只会按原始文本
    # 计算渲染区域。这里用 PIL 按当前字体和字号测量宽度，确保每一行都尽量
    # 控制在视频可用宽度内，避免大字号或中文长句直接溢出画面。
    font = ImageFont.truetype(font, fontsize)
    max_width = int(max_width)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        if not inner_text:
            return 0, fontsize
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    def split_long_token(token):
        # 当一个 token 本身就超宽时（常见于中文无空格长句，或英文超长单词），
        # 退化为字符级拆分。关键点是：检测到 candidate 超宽时，先提交上一个
        # 仍然合法的 current，再把当前字符放入下一行，不能把超宽字符塞回上一行。
        lines = []
        current = ""
        for char in token:
            candidate = f"{current}{char}"
            candidate_width, _ = get_text_size(candidate)
            if candidate_width <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = char
        if current:
            lines.append(current)
        return lines

    lines = []
    current = ""
    words = text.split(" ")
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        candidate_width, _ = get_text_size(candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        word_width, _ = get_text_size(word)
        if word_width <= max_width:
            current = word
        else:
            lines.extend(split_long_token(word))
            current = ""

    if current:
        lines.append(current)

    result = "\n".join(line.strip() for line in lines if line.strip()).strip()
    height = len(lines) * height
    return result, height


def _hex_to_ass_color(hex_color, default="&H00FFFFFF"):
    """#RRGGBB -> ASS &H00BBGGRR (alfa opak)."""
    try:
        c = str(hex_color).lstrip("#")
        if len(c) != 6:
            return default
        r, g, b = c[0:2], c[2:4], c[4:6]
        return f"&H00{b}{g}{r}".upper()
    except Exception:
        return default


def _font_family_name(font_path):
    """Font dosyasının iç aile adını döndürür (libass FontName için)."""
    try:
        from fontTools.ttLib import TTCollection, TTFont
        if str(font_path).lower().endswith(".ttc"):
            font = TTCollection(font_path).fonts[0]
        else:
            font = TTFont(font_path)
        for rec in font["name"].names:
            if rec.nameID == 1 and rec.platformID == 3:
                return rec.toUnicode()
        for rec in font["name"].names:
            if rec.nameID == 1:
                return rec.toUnicode()
    except Exception as e:
        logger.warning(f"font aile adı okunamadı: {str(e)}")
    return ""


def _srt_ts_to_ass(ts):
    """'HH:MM:SS,mmm' -> ASS 'H:MM:SS.cc'."""
    ts = ts.strip().replace(".", ",")
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{int(ms) // 10:02d}"


def _escape_ass_filter_path(p):
    """ffmpeg filtre argümanı için Windows yolunu kaçışla (ters bölü->düz, ':' kaçışlı)."""
    return p.replace("\\", "/").replace(":", "\\:")


# Proje kök dizini (app/services/video.py -> 3 üst).
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ass_filter_vf(ass_path):
    """`ass`/`fontsdir` için GÖRELİ yol kullanan filtre dizesi döndürür.

    Windows'ta mutlak yolların 'C:' kaçışı ffmpeg filtre ayrıştırıcısını bozuyor.
    Göreli yolda iki nokta olmadığı için sorun çıkmaz — ffmpeg kök dizinden (cwd) çalıştırılır.
    """
    rel_ass = os.path.relpath(ass_path, _ROOT_DIR).replace("\\", "/")
    rel_fonts = os.path.relpath(utils.font_dir(), _ROOT_DIR).replace("\\", "/")
    return f"ass={rel_ass}:fontsdir={rel_fonts}"


def _build_ass_from_srt(
    srt_path, ass_path, params, video_width, video_height, font_family
) -> bool:
    """SRT'yi, MoviePy altyazı stiline (font/boyut/renk/kutu/konum) yakın bir ASS'e çevirir.

    PlayRes video çözünürlüğüne sabitlenir; böylece font boyutu 1:1 piksel olur.
    """
    import re as _re

    fontsize = int(params.font_size)
    primary = _hex_to_ass_color(params.text_fore_color, "&H00FFFFFF")
    outline_col = _hex_to_ass_color(params.stroke_color, "&H00000000")
    bg = params.text_background_color
    # MODERN SHORTS STİLİ (varsayılan): opak kutu yerine kalın siyah outline + gölge.
    # Kutu YALNIZCA kullanıcı açıkça bir renk dizesi verirse kullanılır; varsayılan
    # True/None durumunda temiz, dinamik "TikTok" görünümü için outline stiline geç.
    if isinstance(bg, str) and bg.strip():
        border_style = 3  # opak kutu (kullanıcı açıkça renk verdi)
        back_col = _hex_to_ass_color(bg, "&H00000000")
        outline = max(4, int(params.font_size * 0.12))  # kutu dolgusu
        shadow = 0
    else:
        border_style = 1  # sadece çerçeve (modern)
        back_col = "&H00000000"
        # Kalın, okunaklı siyah çerçeve + belirgin gölge → arka plan ne olursa okunur.
        outline = max(3, int(params.font_size * 0.10))
        shadow = max(2, int(params.font_size * 0.045))
    pos = params.subtitle_position
    if pos == "top":
        alignment, margin_v = 8, int(video_height * 0.05)
    elif pos == "center":
        alignment, margin_v = 5, 0
    elif pos == "custom":
        try:
            margin_v = int(video_height * (float(params.custom_position) / 100.0))
        except (TypeError, ValueError):
            margin_v = int(video_height * 0.70)
        alignment = 8
    else:  # bottom (TikTok güvenli bölgesi: alt %21 UI ile kaplı; altyazı onun üstünde kalsın)
        alignment, margin_v = 2, int(video_height * 0.23)
    margin_lr = int(video_width * 0.05)
    style = (
        f"Style: Default,{font_family or 'Arial'},{fontsize},{primary},&H000000FF,"
        f"{outline_col},{back_col},-1,0,0,0,100,100,0,0,{border_style},{outline},{shadow},"
        f"{alignment},{margin_lr},{margin_lr},{margin_v},1"
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {video_width}\nPlayResY: {video_height}\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"{style}\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    try:
        with open(srt_path, encoding="utf-8") as f:
            srt = f.read()
    except Exception:
        return False

    def _ts_sec(ts: str) -> float:
        ts = ts.strip().replace(",", ".")
        try:
            hh, mm, ss = ts.split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        except Exception:
            return 0.0

    # 1) SRT bloklarını (start, end, text) olarak ayrıştır.
    raw = []
    for blk in _re.split(r"\n\s*\n", srt.strip()):
        rows = blk.strip().split("\n")
        if len(rows) >= 2 and "-->" in rows[1]:
            a, b = rows[1].split("-->")
            txt = " ".join(r.strip() for r in rows[2:]).strip()
            if txt:
                raw.append((a.strip(), b.strip(), txt))

    # 2) OKUNABİLİRLİK İÇİN BİRLEŞTİR: edge-tts kelime/virgül bazında çok parçalı
    # cue üretir ("Birincisi", "kahve", "Bu" gibi 0.2sn'lik tek kelimeler). Komşu
    # cue'ları karakter/süre/duraklama sınırlarına göre tek okunabilir satırda
    # topla. (Kaynak SRT ve sahne zamanlaması DEĞİŞMEZ; bu yalnızca ekran gösterimi.)
    MAX_CHARS = 42
    MAX_DUR = 3.6
    MAX_GAP = 0.6
    merged = []  # (start_str, end_str, text)
    for a, b, txt in raw:
        if not merged:
            merged.append([a, b, txt])
            continue
        pa, pb, ptxt = merged[-1]
        gap = _ts_sec(a) - _ts_sec(pb)
        combined = f"{ptxt} {txt}"
        dur = _ts_sec(b) - _ts_sec(pa)
        # Kısa bağlaç parçaları ("İkincisi", "Öncelikle" gibi) tek başına yanıp
        # sönmesin: önceki parça çok kısaysa, daha büyük duraklamaya rağmen birleştir.
        short_connector = len(ptxt) <= 14
        gap_ok = gap <= (1.3 if short_connector else MAX_GAP)
        dur_cap = MAX_DUR + (1.0 if short_connector else 0.0)
        if gap_ok and len(combined) <= MAX_CHARS and dur <= dur_cap:
            merged[-1] = [pa, b, combined]
        else:
            merged.append([a, b, txt])

    # 3) Uzun satırları iki dengeli satıra böl (tek satır taşmasın).
    def _wrap(text: str) -> str:
        if len(text) <= 24:
            return text
        words = text.split()
        best, target = None, len(text) / 2
        acc = 0
        for i in range(1, len(words)):
            acc = len(" ".join(words[:i]))
            if best is None or abs(acc - target) < abs(best[1] - target):
                best = (i, acc)
        i = best[0] if best else len(words)
        return " ".join(words[:i]) + "\\N" + " ".join(words[i:])

    lines = []
    for a, b, txt in merged:
        lines.append(
            f"Dialogue: 0,{_srt_ts_to_ass(a)},{_srt_ts_to_ass(b)},Default,,0,0,0,,{_wrap(txt)}"
        )
    if not lines:
        return False
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")
    return True


def _try_ffmpeg_final_render(
    video_path, voice_audio_path, subtitle_path, output_file, params,
    font_path, video_width, video_height, output_dir,
) -> bool:
    """Final videoyu ffmpeg ile üretir: altyazıyı gömer + sesi mux'lar.

    MoviePy kare-kare compositing'inden ~10x hızlıdır; görünüm/kalite korunur.
    Ses (voice + BGM) MoviePy ile birebir aynı mantıkla hazırlanıp geçici dosyaya
    yazılır, sonra ffmpeg muxlar. Herhangi bir adım başarısızsa False döner (MoviePy'ye düşülür).
    """
    temp_audio = os.path.join(output_dir, "_final_audio.m4a")
    ass_file = os.path.join(output_dir, "_subtitle.ass")
    cleanup = []
    try:
        # 1) Video süresi (BGM loop için) — render etmeden sadece süre okunur.
        probe = _open_video_clip_quietly(video_path)
        total_duration = probe.duration
        close_clip(probe)

        # 2) Ses: voice (+ varsa BGM) -> geçici dosya.
        bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)

        # 2a) DUCKING (BGM varsa): müzik, konuşma varken otomatik kısılır
        # (sidechaincompress), konuşma olmayan anlarda normale döner. Yayın
        # hissi veren profesyonel denge. Başarısız olursa MoviePy mix'e düşülür.
        duck_done = False
        if bgm_file:
            try:
                vv = float(params.voice_volume or 1.0)
                bv = float(params.bgm_volume if params.bgm_volume is not None else 0.2)
                fade_st = max(0.0, float(total_duration) - 3.0)
                fc = (
                    f"[0:a]volume={vv},aformat=channel_layouts=stereo,asplit=2[v][vk];"
                    f"[1:a]volume={bv},aformat=channel_layouts=stereo[m];"
                    f"[m][vk]sidechaincompress=threshold=0.03:ratio=10:attack=5:release=300[md];"
                    f"[md]afade=t=out:st={fade_st:.3f}:d=3[mdf];"
                    f"[v][mdf]amix=inputs=2:duration=first:normalize=0[a]"
                )
                cmd_a = [
                    get_ffmpeg_binary(), "-y", "-loglevel", "error",
                    "-i", voice_audio_path, "-stream_loop", "-1", "-i", bgm_file,
                    "-filter_complex", fc, "-map", "[a]",
                    "-t", f"{total_duration}",
                    "-c:a", audio_codec, "-b:a", audio_bitrate, temp_audio,
                ]
                r_a = subprocess.run(cmd_a, capture_output=True, text=True, check=False)
                duck_done = (
                    r_a.returncode == 0
                    and os.path.exists(temp_audio)
                    and os.path.getsize(temp_audio) > 0
                )
                if duck_done:
                    cleanup.append(temp_audio)
                    logger.success("ses: BGM ducking uygulandı (konuşmada müzik kısılır)")
                else:
                    logger.warning(
                        f"ducking başarısız, MoviePy mix'e düşülüyor: {(r_a.stderr or '')[:150]}"
                    )
            except Exception as e:
                logger.warning(f"ducking hatası, MoviePy mix'e düşülüyor: {str(e)}")

        # 2b) MoviePy yolu (ducking yoksa/başarısızsa): voice + basit BGM mix.
        if not duck_done:
            audio_clip = AudioFileClip(voice_audio_path).with_effects(
                [afx.MultiplyVolume(params.voice_volume)]
            )
            if bgm_file:
                try:
                    bgm_clip = AudioFileClip(bgm_file).with_effects(
                        [
                            afx.MultiplyVolume(params.bgm_volume),
                            afx.AudioFadeOut(3),
                            afx.AudioLoop(duration=total_duration),
                        ]
                    )
                    audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
                except Exception as e:
                    logger.error(f"failed to add bgm: {str(e)}")
            output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
            audio_clip.write_audiofile(
                temp_audio, codec=audio_codec, fps=output_audio_fps,
                bitrate=audio_bitrate, logger=None,
            )
            close_clip(audio_clip)
            cleanup.append(temp_audio)

        # 3) Altyazı ASS (varsa). Font aile adı okunamazsa hızlı yoldan vazgeç.
        vf = None
        if subtitle_path and font_path:
            family = _font_family_name(font_path)
            if not family:
                return False
            if not _build_ass_from_srt(
                subtitle_path, ass_file, params, video_width, video_height, family
            ):
                return False
            cleanup.append(ass_file)
            vf = _ass_filter_vf(ass_file)

        # 4) ffmpeg: altyazı gömme (varsa) + ses mux.
        base = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-i", video_path, "-i", temp_audio,
        ]
        tail = [
            "-map", "0:v:0", "-map", "1:a:0",
            # Yayın-standardı ses normalizasyonu (~-14 LUFS): her videoda tek tip,
            # tutarlı ses seviyesi; çok kısık/çok yüksek sesi engeller (EBU R128).
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", audio_codec, "-b:a", audio_bitrate, "-shortest", output_file,
        ]

        def _run(extra_video_args):
            cmd = base + extra_video_args + tail
            # ass filtresinin göreli yolları için kök dizinden çalıştır.
            r = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=_ROOT_DIR
            )
            ok = (
                r.returncode == 0
                and os.path.exists(output_file)
                and os.path.getsize(output_file) > 0
            )
            return ok, (r.stderr or r.stdout or "")

        if not vf:
            # Altyazı yoksa videoyu kopyala (kayıpsız, çok hızlı).
            ok, err = _run(["-c:v", "copy"])
            if ok:
                return True
            logger.warning(f"ffmpeg copy-mux başarısız: {err[:150]}")
            return False

        # Altyazı gömülecek -> video yeniden encode olmalı. Önce DONANIM hızlandırma
        # (Intel QuickSync, ~10x hızlı), başarısız olursa x264 (yazılım) yedeği.
        # crf/global_quality değerleri görsel olarak (neredeyse) kayıpsız tutulur.
        encoders = [
            ("h264_qsv (donanım)", [
                "-vf", vf, "-c:v", "h264_qsv", "-global_quality", "20",
                "-preset", "veryfast", "-pix_fmt", "nv12",
            ]),
            ("libx264 (yazılım)", [
                "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p",
            ]),
        ]
        for name, args in encoders:
            ok, err = _run(args)
            if ok:
                logger.success(f"final render encoder: {name}")
                return True
            logger.warning(f"final render {name} başarısız: {err[:150]}")
    except Exception as e:
        logger.warning(f"ffmpeg final render hatası (MoviePy'ye düşülüyor): {str(e)}")
    finally:
        for f in cleanup:
            try:
                if os.path.exists(f):
                    delete_files(f)
            except Exception:
                pass
    return False


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
):
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    # HIZLI YOL: ffmpeg ile altyazı gömme + ses mux (MoviePy compositing'inden ~10x hızlı,
    # görünüm/kalite birebir korunur). Başarısız olursa aşağıdaki MoviePy yoluna düşülür.
    _has_sub = bool(
        subtitle_path and os.path.exists(subtitle_path) and params.subtitle_enabled
    )
    if _try_ffmpeg_final_render(
        video_path,
        audio_path,
        subtitle_path if _has_sub else "",
        output_file,
        params,
        font_path,
        video_width,
        video_height,
        output_dir,
    ):
        logger.success(f"final video (hızlı ffmpeg yolu): {output_file}")
        return
    logger.info("ffmpeg hızlı yolu kullanılamadı, MoviePy ile devam ediliyor")

    def resolve_subtitle_background_color():
        # 兼容历史参数：API 里 `text_background_color` 既可能是布尔值，
        # 也可能是实际颜色字符串。统一在这里归一化，避免把 True/False
        # 直接传给 TextClip 后出现不可预期的渲染结果。
        if isinstance(params.text_background_color, bool):
            return "#000000" if params.text_background_color else None
        return params.text_background_color

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        wrapped_txt, txt_height = wrap_text(
            phrase, max_width=max_width, font=font_path, fontsize=params.font_size
        )
        interline = int(params.font_size * 0.25)
        line_count = wrapped_txt.count("\n") + 1
        vertical_padding = int(params.font_size * 0.35)
        # MoviePy 在 `method=label` 下会自动收缩文本框高度，遇到多行字幕、
        # 描边或背景色时，容易把最后一行的下半部分裁掉。这里显式传入
        # 一个更保守的高度，把行间距和额外上下留白一并算进去，保证字幕
        # 背景框与文字本身都能完整渲染出来。
        size = (
            int(max_width),
            int(txt_height + vertical_padding + (interline * line_count)),
        )

        _clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=params.font_size,
            color=params.text_fore_color,
            bg_color=resolve_subtitle_background_color(),
            stroke_color=params.stroke_color,
            stroke_width=params.stroke_width,
            interline=interline,
            size=size,
            text_align="center",
        )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        if params.subtitle_position == "bottom":
            # TikTok güvenli bölgesi: altyazı, kullanıcı adı/açıklama alanının üstünde kalsın.
            _clip = _clip.with_position(("center", video_height * 0.77 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    video_clip = _open_video_clip_quietly(video_path)
    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=params.font_size,
        )

    if subtitle_path and os.path.exists(subtitle_path):
        sub = SubtitlesClip(
            subtitles=subtitle_path, encoding="utf-8", make_textclip=make_textclip
        )
        text_clips = []
        for item in sub.subtitles:
            clip = create_text_clip(subtitle_item=item)
            text_clips.append(clip)
        video_clip = CompositeVideoClip([video_clip, *text_clips])

    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
        try:
            bgm_clip = AudioFileClip(bgm_file).with_effects(
                [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                    afx.AudioLoop(duration=video_clip.duration),
                ]
            )
            audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
        except Exception as e:
            logger.error(f"failed to add bgm: {str(e)}")

    video_clip = video_clip.with_audio(audio_clip)
    # 显式沿用输入音频的采样率；如果取不到，再回退到 MoviePy 默认的 44100Hz。
    # 这样可以减少不同运行环境，尤其是 Docker 环境中再次重采样带来的音质波动。
    output_audio_fps = int(getattr(audio_clip, "fps", 0) or 44100)
    video_clip.write_videofile(
        output_file,
        audio_codec=audio_codec,
        audio_fps=output_audio_fps,
        audio_bitrate=audio_bitrate,
        temp_audiofile_path=output_dir,
        threads=params.n_threads or 2,
        logger=None,
        fps=fps,
    )
    video_clip.close()
    del video_clip


def _fit_clip_to_frame(clip, video_width: int, video_height: int):
    """Bir klibi hedef çerçeveyi TAM dolduracak şekilde ölçekler ve ortadan kırpar.

    Aspect uymadığında siyah bant bırakmak yerine kapsayacak şekilde büyütüp ortadan kırpar.
    """
    clip_w, clip_h = clip.size
    if clip_w == video_width and clip_h == video_height:
        return clip
    clip_ratio = clip_w / clip_h
    video_ratio = video_width / video_height
    if clip_ratio == video_ratio:
        return clip.resized(new_size=(video_width, video_height))
    if clip_ratio > video_ratio:
        scale_factor = video_height / clip_h
    else:
        scale_factor = video_width / clip_w
    new_width = int(clip_w * scale_factor)
    new_height = int(clip_h * scale_factor)
    resized = clip.resized(new_size=(new_width, new_height))
    return resized.cropped(
        x_center=new_width / 2,
        y_center=new_height / 2,
        width=video_width,
        height=video_height,
    )


def _render_image_clip_ffmpeg(
    image_path: str, out_path: str, duration: float, video_width: int, video_height: int
) -> bool:
    """Bir görseli ffmpeg `zoompan` ile zoom efektli klibe çevirir.

    MoviePy'nin kare-kare Python render'ından ~25x hızlıdır. Süre, çözünürlük ve
    zoom hızı MoviePy yoluyla BİREBİR aynıdır (zamanlama/senkron korunur).
    Başarılıysa True döner; ffmpeg yoksa/başarısızsa False döner (MoviePy'ye düşülür).
    """
    try:
        fps = 30
        frames = max(1, int(round(duration * fps)))
        big_w, big_h = video_width * 2, video_height * 2
        denom = max(1, frames - 1)

        # SİNEMATİK HAREKET: pan açıksa, kamera sahne üzerinde süzülür (zoom + yön
        # değiştiren kaydırma). Yön klibe göre değişir (dosya adıyla seed'lenir) ki
        # her sahnede farklı, daha canlı görünsün. Statik zoom'dan çok daha dinamik.
        pan_enabled = bool(config.app.get("scene_motion_pan", True))
        if pan_enabled:
            try:
                _seed = int(utils.md5(os.path.basename(image_path))[:8], 16)
            except Exception:
                _seed = 0
            direction = _seed % 4  # 0:sol->sağ 1:sağ->sol 2:üst->alt 3:alt->üst
            # GEÇİŞ (zoom-punch): klip başında ekstra zoom hızla sönerek her KESİMDE
            # enerjik bir "punch in" hissi verir. Klip süresini DEĞİŞTİRMEZ (overlap
            # yok) → ses/altyazı senkronu korunur. Sonra normal yavaş pan/zoom sürer.
            _pf = max(1, int(fps * 0.2))  # ~0.2sn punch
            z_expr = f"min(1.10+0.0006*on+0.09*max(0,1-on/{_pf}),1.24)"
            progress = f"(on/{denom})"
            x_center = "iw/2-(iw/zoom/2)"
            y_center = "ih/2-(ih/zoom/2)"
            if direction == 0:  # sol -> sağ
                x_expr, y_expr = f"(iw-iw/zoom)*{progress}", y_center
            elif direction == 1:  # sağ -> sol
                x_expr, y_expr = f"(iw-iw/zoom)*(1-{progress})", y_center
            elif direction == 2:  # üst -> alt
                x_expr, y_expr = x_center, f"(ih-ih/zoom)*{progress}"
            else:  # alt -> üst
                x_expr, y_expr = x_center, f"(ih-ih/zoom)*(1-{progress})"
            vf = (
                f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,"
                f"crop={big_w}:{big_h},"
                f"zoompan=z='{z_expr}':d={frames}:x='{x_expr}':y='{y_expr}':"
                f"s={video_width}x{video_height}:fps={fps},setsar=1"
            )
        else:
            # Eski davranış: ortalanmış basit zoom.
            max_zoom = round(1 + 0.03 * duration, 4)
            vf = (
                f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,"
                f"crop={big_w}:{big_h},"
                f"zoompan=z='min(1+0.001*on,{max_zoom})':d={frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={video_width}x{video_height}:fps={fps},setsar=1"
            )
        cmd = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", str(fps), "-i", image_path,
            "-t", f"{duration}",
            "-vf", vf,
            "-c:v", video_codec, "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if (
            result.returncode == 0
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 0
        ):
            return True
        logger.warning(
            f"ffmpeg zoompan başarısız (MoviePy'ye düşülüyor): "
            f"{(result.stderr or result.stdout or '')[:200]}"
        )
    except Exception as e:
        logger.warning(f"ffmpeg zoompan hatası (MoviePy'ye düşülüyor): {str(e)}")
    return False


def preprocess_video(materials: List[MaterialInfo], clip_duration=4, target_resolution=None):
    # WebUI 在某些二次生成场景下可能传入空素材列表，这里直接返回空结果，避免抛出 NoneType 异常。
    if not materials:
        return []

    # 仅返回通过预处理校验的素材，避免低分辨率图片继续进入后续的视频合成流程。
    valid_materials = []
    local_videos_dir = utils.storage_dir("local_videos", create=True)

    for material in materials:
        if not material.url:
            continue

        try:
            material_source_path = file_security.resolve_path_within_directory(
                local_videos_dir, material.url
            )
        except ValueError as exc:
            # local video_source 的素材路径来自 API 参数，必须限制在专用素材目录。
            # 允许用户传文件名，也兼容历史返回的绝对路径，但不允许逃逸到系统
            # 其他目录，避免任意文件读取或通过 MoviePy 探测本地敏感文件。
            logger.warning(
                f"skip unsafe local material: {material.url}, "
                f"local_videos_dir: {local_videos_dir}, error: {str(exc)}"
            )
            continue

        ext = utils.parse_extension(material_source_path)
        try:
            # 图片素材直接按图片方式读取，避免先走 VideoFileClip 误判后触发不稳定的回退分支。
            if ext in const.FILE_TYPE_IMAGES:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            else:
                clip = _open_video_clip_quietly(material_source_path)
        except Exception:
            # 非标准扩展名或探测失败时再回退到图片模式，兼容历史上直接传本地图片路径的情况。
            try:
                clip, material_source_path = _open_image_clip_with_fallback(
                    material_source_path
                )
            except Exception as exc:
                logger.warning(
                    f"skip unreadable local material: {material.url}, error: {str(exc)}"
                )
                continue
        try:
            width = clip.size[0]
            height = clip.size[1]
            if width < 480 or height < 480:
                logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
                # 探测到低分辨率素材后立即关闭资源，并且不要把该素材返回给后续流程。
                close_clip(clip)
                continue

            if ext in const.FILE_TYPE_IMAGES:
                logger.info(f"processing image: {material_source_path}")
                # 探测尺寸时已经打开过一次素材，这里先释放探测句柄，再重新创建用于导出的图片 clip。
                close_clip(clip)

                # Sahne-eşleştirmeli modda her görselin süresi, o segmentin süresine
                # eşittir (material.duration). Aksi halde varsayılan clip_duration kullanılır.
                img_duration = clip_duration
                if getattr(material, "duration", 0):
                    try:
                        img_duration = max(0.5, float(material.duration))
                    except (TypeError, ValueError):
                        img_duration = clip_duration

                # Süreyi dosya adına göm: sahne modunda aynı görsel farklı segmentlerde
                # FARKLI sürelerle kullanılabilir; aksi halde çıktı dosyaları çakışır
                # ve zaman çizelgesi kayar.
                duration_tag = int(round(img_duration * 1000))
                video_file = f"{material_source_path}.d{duration_tag}.mp4"

                # HIZLI YOL: hedef çözünürlük verildiyse (AI/sahne modu) ffmpeg zoompan
                # ile render et (~25x hızlı, aynı süre/kalite/zoom).
                rendered = False
                if target_resolution:
                    frame_w, frame_h = int(target_resolution[0]), int(target_resolution[1])
                    rendered = _render_image_clip_ffmpeg(
                        material_source_path, video_file, img_duration, frame_w, frame_h
                    )

                if not rendered:
                    # YEDEK: MoviePy ile render (yerel mod veya ffmpeg başarısızsa).
                    clip = (
                        ImageClip(material_source_path)
                        .with_duration(img_duration)
                        .with_position("center")
                    )
                    if target_resolution:
                        frame_w, frame_h = int(target_resolution[0]), int(target_resolution[1])
                        clip = _fit_clip_to_frame(clip, frame_w, frame_h)
                    else:
                        frame_w, frame_h = clip.size
                    zoom_clip = clip.resized(
                        lambda t: 1 + (img_duration * 0.03) * (t / img_duration)
                    ).with_position("center")
                    final_clip = CompositeVideoClip([zoom_clip], size=(frame_w, frame_h))
                    final_clip.write_videofile(video_file, fps=30, logger=None)
                    close_clip(clip)
                    close_clip(final_clip)

                material.url = video_file
                logger.success(f"image processed: {video_file}")
            else:
                # 普通视频素材只需要读取尺寸做校验，校验完成后立即释放句柄即可。
                close_clip(clip)
        except Exception:
            close_clip(clip)
            raise

        valid_materials.append(material)

    return valid_materials


def assemble_ordered_clips(
    combined_video_path: str,
    clip_paths: List[str],
    threads: int = 2,
) -> str:
    """Önceden işlenmiş klipleri (doğru süre/sıra/çözünürlükte) SIRAYLA birleştirir.

    Sahne-eşleştirmeli mod için kullanılır: klipler zaten seslendirme segmentlerinin
    süreleriyle ve sırasıyla üretildiğinden, karıştırma/dilimleme yapmadan düz
    birleştirme yeterlidir. Sessiz birleşik mp4 döner (ses sonradan generate_video ile eklenir).
    """
    if not clip_paths:
        raise ValueError("no clips to assemble")
    output_dir = os.path.dirname(combined_video_path)
    valid_clips = [p for p in clip_paths if p and os.path.exists(p)]
    if not valid_clips:
        raise ValueError("no valid clip files to assemble")
    logger.info(f"assembling {len(valid_clips)} scene clips in order with ffmpeg")

    # HIZLI YOL: klipler birebir aynı formatta (ffmpeg zoompan çıktısı) olduğundan
    # yeniden-encode yapmadan STREAM COPY ile birleştir (çok hızlı). Başarısız olursa
    # yeniden-encode'a düşülür.
    concat_list_file = os.path.join(output_dir, "ffmpeg-concat-copy.txt")
    try:
        with open(concat_list_file, "w", encoding="utf-8") as fp:
            for clip_file in valid_clips:
                abs_path = os.path.abspath(clip_file)
                fp.write(f"file '{_escape_ffmpeg_concat_path(abs_path)}'\n")
        cmd = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-c", "copy", combined_video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if (
            result.returncode == 0
            and os.path.exists(combined_video_path)
            and os.path.getsize(combined_video_path) > 0
        ):
            logger.success(f"scene video assembled (copy): {combined_video_path}")
            return combined_video_path
        logger.warning(
            f"stream-copy birleştirme başarısız, yeniden-encode'a düşülüyor: "
            f"{(result.stderr or result.stdout or '')[:150]}"
        )
    except Exception as e:
        logger.warning(f"stream-copy hatası, yeniden-encode'a düşülüyor: {str(e)}")
    finally:
        delete_files(concat_list_file)

    # YEDEK: yeniden-encode ile birleştir.
    concat_video_clips_with_ffmpeg(
        clip_files=valid_clips,
        output_file=combined_video_path,
        threads=threads,
        output_dir=output_dir,
    )
    logger.success(f"scene video assembled: {combined_video_path}")
    return combined_video_path


def _tr_upper(s: str) -> str:
    """Türkçe-duyarlı büyük harf (i->İ, ı->I)."""
    return str(s).replace("ı", "I").replace("i", "İ").upper()


def _ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h = cs // 360000
    m = (cs // 6000) % 60
    s = (cs // 100) % 60
    c = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _shift_srt(srt_path: str, out_path: str, offset_seconds: float) -> bool:
    """Bir .srt dosyasındaki tüm zaman damgalarını `offset_seconds` kadar ileri kaydırır."""
    import re as _re

    def _shift_ts(ts: str) -> str:
        h, m, rest = ts.split(":")
        s, ms = rest.split(",")
        total = (
            int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0 + offset_seconds
        )
        h2 = int(total // 3600)
        m2 = int((total % 3600) // 60)
        s2 = int(total % 60)
        ms2 = int(round((total - int(total)) * 1000))
        if ms2 >= 1000:
            s2 += 1
            ms2 = 0
        return f"{h2:02d}:{m2:02d}:{s2:02d},{ms2:03d}"

    try:
        with open(srt_path, encoding="utf-8") as f:
            content = f.read()

        def _repl(mo):
            return f"{_shift_ts(mo.group(1))} --> {_shift_ts(mo.group(2))}"

        new = _re.sub(
            r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", _repl, content
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new)
        return True
    except Exception as e:
        logger.warning(f"srt kaydırma hatası: {str(e)}")
        return False


def _prepend_audio_silence(audio_path: str, out_path: str, offset_seconds: float) -> bool:
    """Ses dosyasının başına `offset_seconds` kadar sessizlik ekler (intro için)."""
    try:
        ms = int(offset_seconds * 1000)
        cmd = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error", "-i", audio_path,
            "-af", f"adelay={ms}:all=1",
            "-c:a", audio_codec, "-b:a", audio_bitrate, out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
        logger.warning(f"ses gecikme hatası: {(r.stderr or '')[:150]}")
    except Exception as e:
        logger.warning(f"ses gecikme hatası: {str(e)}")
    return False


def clip_duration_seconds(path: str) -> float:
    """Bir video klibinin süresini saniye olarak döndürür (render etmeden)."""
    try:
        c = _open_video_clip_quietly(path)
        d = float(c.duration or 0)
        close_clip(c)
        return d
    except Exception:
        return 0.0


def _trim_clip_start(input_path: str, out_path: str, start_seconds: float) -> bool:
    """Trim a scene clip after an exact intro replacement duration."""
    try:
        cmd = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-ss", f"{max(0.0, float(start_seconds)):.3f}",
            "-i", input_path,
            "-an", "-c:v", video_codec, "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-r", "30", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if (
            result.returncode == 0
            and os.path.exists(out_path)
            and os.path.getsize(out_path) > 0
        ):
            return True
        logger.warning(f"scene prefix trim failed: {(result.stderr or '')[:150]}")
    except Exception as e:
        logger.warning(f"scene prefix trim failed: {str(e)}")
    return False


def replace_ordered_clip_prefix(
    clip_paths: List[str],
    intro_clip_path: str,
    intro_duration: float,
    output_dir: str,
) -> List[str]:
    """Replace an exact prefix duration with the intro while preserving scene timing."""
    if not intro_clip_path or not os.path.exists(intro_clip_path):
        return clip_paths

    remaining = max(0.0, float(intro_duration))
    result = [intro_clip_path]
    for idx, clip_path in enumerate(clip_paths):
        if remaining <= 0:
            result.extend(clip_paths[idx:])
            break
        duration = clip_duration_seconds(clip_path)
        if duration <= 0:
            continue
        if remaining >= duration - 0.05:
            remaining -= duration
            continue

        trimmed_path = os.path.join(output_dir, f"_intro-trimmed-scene-{idx + 1}.mp4")
        if _trim_clip_start(clip_path, trimmed_path, remaining):
            result.append(trimmed_path)
        else:
            logger.warning("intro trim fallback: keeping original scene clip")
            result.append(clip_path)
        result.extend(clip_paths[idx + 1:])
        break
    return result


def create_title_intro_clip(
    cover_image: str, title_text: str, out_path: str, duration: float,
    video_width: int, video_height: int, font_path: str, motion_clip: str = "",
) -> bool:
    """Konuya özel kapak (statik veya HAREKETLİ) + ortada büyük kalın başlık ile intro klibi üretir.

    `motion_clip` verilirse (Pollinations i2v çıktısı), kapak GERÇEKTEN hareket eder;
    aksi halde statik kapak görseline hafif zoom uygulanır. Başlık ekranın ortasında,
    büyük/kalın, beyaz + siyah çerçeve-gölge. Klip formatı sahne klipleriyle aynıdır.
    """
    ass_path = out_path + ".title.ass"
    base_path = out_path + ".base.mp4"
    try:
        fps = 30
        frames = max(1, int(round(duration * fps)))
        family = "Georgia"
        fontsize = int(video_height * 0.058)
        margin_lr = int(video_width * 0.08)
        title = _tr_upper(title_text).replace("\n", " ").strip()
        big_w, big_h = video_width * 2, video_height * 2

        if motion_clip and os.path.exists(motion_clip):
            # --- GEÇİŞ 1 (HAREKETLİ): i2v klibini çerçeveye otur + süreye kırp ---
            vf1 = (
                f"scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={video_width}:{video_height},fps={fps},setsar=1"
            )
            cmd1 = [
                get_ffmpeg_binary(), "-y", "-loglevel", "error", "-i", motion_clip,
                "-t", f"{duration}", "-vf", vf1,
                "-c:v", video_codec, "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-r", str(fps), base_path,
            ]
        else:
            # --- GEÇİŞ 1 (STATİK): LETTERBOX kapak (FADE YOK) ---
            # AI kapağı çoğunlukla kare (1024²) üretiliyor; 9:16'ya KIRPMAK yerine
            # bulanık-dolgu arka plan + ortalanmış NET kapak (contain) ile yerleştir.
            # Böylece kapaktaki çerçeve/altın kenar kesilmez, kenarlar siyah bant
            # yerine sinematik bulanık dolgu olur. Bütüne çok hafif yavaş zoom verilir.
            # İlk kare NET olmalı (TikTok kapağı siyah olmasın) -> fade-in yok.
            fc = (
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale={video_width}:{video_height}:force_original_aspect_ratio=increase,"
                f"crop={video_width}:{video_height},boxblur=22:1,eq=brightness=-0.06:saturation=1.06[bgb];"
                f"[fg]scale={video_width}:{video_height}:force_original_aspect_ratio=decrease[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={fps},"
                f"zoompan=z='min(1.0+0.0006*on,1.06)':d=1:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={video_width}x{video_height}:fps={fps},setsar=1[v]"
            )
            cmd1 = [
                get_ffmpeg_binary(), "-y", "-loglevel", "error",
                "-loop", "1", "-framerate", str(fps), "-i", cover_image,
                "-t", f"{duration}", "-filter_complex", fc, "-map", "[v]",
                "-c:v", video_codec, "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-r", str(fps), base_path,
            ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, check=False)
        if r1.returncode != 0 or not os.path.exists(base_path):
            logger.warning(f"intro kapak klibi üretilemedi: {(r1.stderr or '')[:200]}")
            return False

        # --- GEÇİŞ 2: modern sosyal medya kapak şablonunu ASS ile bindir ---
        title_style = (
            f"Style: T,{family},{fontsize},&H0000D7FF,&H000000FF,&H90000000,&H90000000,"
            f"-1,0,0,0,100,100,1,0,1,3,3,5,{margin_lr},{margin_lr},0,1"
        )
        header = (
            "[Script Info]\nScriptType: v4.00+\n"
            f"PlayResX: {video_width}\nPlayResY: {video_height}\nWrapStyle: 0\n\n"
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            f"{title_style}\n\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
        )
        # Başlık ilk kareden itibaren TAM görünür (fade-in yok); yalnızca sona doğru
        # hafifçe kaybolur (sahnelere yumuşak geçiş).
        title_y = int(video_height * 0.27)
        dialogue = (
            f"Dialogue: 0,{_ass_time(0)},{_ass_time(duration)},T,,0,0,0,,"
            f"{{\\pos({video_width // 2},{title_y})\\fad(0,250)}}{title}\n"
        )
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header + dialogue + "\n")

        # Profesyonel sinematik katman: köşelerde hafif vignette + üstte yumuşak
        # koyu degrade scrim (3 kademeli kutu = yumuşak gradyan). Bu hem başlığın
        # okunaklılığını artırır hem de AI kapağında kalan uyduruk yazı/artefaktları
        # gözden gizler. Başlık (ass) en üste bindirildiği için net kalır.
        _b1 = int(video_height * 0.46)
        _b2 = int(video_height * 0.34)
        _b3 = int(video_height * 0.20)
        _bot1 = int(video_height * 0.20)
        _bot2 = int(video_height * 0.11)
        scrim = (
            f"vignette=PI/5,"
            f"drawbox=x=0:y=0:w={video_width}:h={_b1}:color=black@0.16:t=fill,"
            f"drawbox=x=0:y=0:w={video_width}:h={_b2}:color=black@0.16:t=fill,"
            f"drawbox=x=0:y=0:w={video_width}:h={_b3}:color=black@0.18:t=fill,"
            # Alt şerit 2 kademeli + daha koyu: AI kapağının alt kenarındaki olası
            # uyduruk yazı/etiketleri tamamen gizler (kümülatif ~0.55 en altta).
            f"drawbox=x=0:y={video_height - _bot1}:w={video_width}:h={_bot1}:color=black@0.30:t=fill,"
            f"drawbox=x=0:y={video_height - _bot2}:w={video_width}:h={_bot2}:color=black@0.32:t=fill"
        )
        vf2 = f"{scrim},{_ass_filter_vf(ass_path)}"
        cmd2 = [
            get_ffmpeg_binary(), "-y", "-loglevel", "error", "-i", base_path,
            "-vf", vf2, "-c:v", video_codec, "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-r", str(fps), out_path,
        ]
        r2 = subprocess.run(
            cmd2, capture_output=True, text=True, check=False, cwd=_ROOT_DIR
        )
        if r2.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            logger.success(f"intro klibi üretildi: {out_path}")
            return True
        logger.warning(f"intro başlık bindirme başarısız: {(r2.stderr or '')[:200]}")
    except Exception as e:
        logger.warning(f"intro klibi hatası: {str(e)}")
    finally:
        for _f in (ass_path, base_path):
            try:
                if os.path.exists(_f):
                    delete_files(_f)
            except Exception:
                pass
    return False
