import math
import os.path
import re
from os import path
from typing import List, Tuple

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoAspect, VideoConcatMode, VideoParams
from app.services import llm, material, subtitle, video, voice, upload_post
from app.services import state as sm
from app.utils import utils


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
            video_script_prompt=params.video_script_prompt,
            custom_system_prompt=params.custom_system_prompt,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    repetition_reason = llm.script_repetition_reason(video_script)
    if repetition_reason:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(f"rejected repetitive video script: {repetition_reason}")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject, video_script=video_script, amount=5
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def generate_audio(task_id, params, video_script):
    '''
    Generate audio for the video script.
    If a custom audio file is provided, it will be used directly.
    There will be no subtitle maker object returned in this case.
    Otherwise, TTS will be used to generate the audio.
    Returns:
        - audio_file: path to the generated or provided audio file
        - audio_duration: duration of the audio in seconds
        - sub_maker: subtitle maker object if TTS is used, None otherwise
    '''
    logger.info("\n\n## generating audio")
    # /audio 和 /subtitle 请求模型不包含 custom_audio_file，
    # 这里统一做兼容读取，避免直调接口时抛属性错误。
    custom_audio_file = getattr(params, "custom_audio_file", None)
    if not custom_audio_file or not os.path.exists(custom_audio_file):
        if custom_audio_file:
            logger.warning(
                f"custom audio file not found: {custom_audio_file}, using TTS to generate audio."
            )
        else:
            logger.info("no custom audio file provided, using TTS to generate audio.")
        audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
        sub_maker = voice.tts(
            text=video_script,
            voice_name=voice.parse_voice_name(params.voice_name),
            voice_rate=params.voice_rate,
            voice_file=audio_file,
        )
        if sub_maker is None:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
            """.strip()
            )
            return None, None, None
        audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration.")
            return None, None, None
        return audio_file, audio_duration, sub_maker
    else:
        logger.info(f"using custom audio file: {custom_audio_file}")
        audio_duration = voice.get_audio_duration(custom_audio_file)
        if audio_duration == 0:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to get audio duration from custom audio file.")
            return None, None, None
        return custom_audio_file, audio_duration, None

def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    '''
    Generate subtitle for the video script.
    If subtitle generation is disabled or no subtitle maker is provided, it will return an empty string.
    Otherwise, it will generate the subtitle using the specified provider.
    Returns:
        - subtitle_path: path to the generated subtitle file
    '''
    logger.info("\n\n## generating subtitle")
    if not params.subtitle_enabled or sub_maker is None:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def _srt_time_to_seconds(t: str) -> float:
    """'HH:MM:SS,mmm' (SRT) zaman dizesini saniyeye çevirir."""
    t = t.strip().replace(",", ".")
    hh, mm, ss = t.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def _build_timed_segments(task_id, params, subtitle_path, sub_maker, video_script):
    """Seslendirmenin (start, end, text) segmentlerini SIRALI olarak kurar.

    Geçerli bir subtitle_path varsa onu kullanır; yoksa sub_maker'dan yalnızca
    ZAMANLAMA amaçlı geçici bir .srt üretir (ekran altyazısından bağımsız, gösterilmez).
    """
    srt_path = subtitle_path
    if not (srt_path and os.path.exists(srt_path)):
        if sub_maker is None:
            return []
        try:
            srt_path = path.join(utils.task_dir(task_id), "_scene_timing.srt")
            voice.create_subtitle(
                sub_maker=sub_maker, text=video_script, subtitle_file=srt_path
            )
        except Exception as e:
            logger.warning(f"failed to build timing subtitle: {str(e)}")
            return []

    try:
        lines = subtitle.file_to_subtitles(srt_path)
    except Exception as e:
        logger.warning(f"failed to read subtitle segments: {str(e)}")
        return []

    segments = []
    for item in lines:
        # item: (index, "HH:MM:SS,mmm --> HH:MM:SS,mmm", text)
        try:
            start_s, end_s = [p.strip() for p in item[1].split("-->")]
            text = item[2].replace("\n", " ").strip()
            if not text:
                continue
            segments.append(
                (_srt_time_to_seconds(start_s), _srt_time_to_seconds(end_s), text)
            )
        except Exception:
            continue
    return segments


def _group_timed_segments(
    segments: List[Tuple[float, float, str]],
    min_duration: float = 1.8,
    target_duration: float = 3.2,
    max_duration: float = 4.8,
    hard_max_duration: float = 5.5,
) -> List[Tuple[float, float, str]]:
    """Build short visual beats from subtitle timing without losing meaning.

    Subtitle cues are optimized for reading speed. Visual scenes need a different
    rhythm: connectors stay attached to meaningful text, new ideas open a scene,
    and long narration is split into additional topic-linked camera angles.
    """
    if not segments:
        return []

    min_duration = max(1.0, float(min_duration))
    target_duration = max(min_duration, float(target_duration))
    max_duration = max(target_duration, float(max_duration))
    hard_max_duration = max(max_duration, float(hard_max_duration))
    transition_re = re.compile(
        r"^(birinci(?:si)?|ikinci(?:si)?|üçüncü(?:sü)?|dördüncü(?:sü)?|"
        r"beşinci(?:si)?|altıncı(?:sı)?|yedinci(?:si)?|sekizinci(?:si)?|"
        r"dokuzuncu(?:su)?|onuncu(?:su)?|first|second|third|"
        r"fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
        re.IGNORECASE,
    )
    bridge_re = re.compile(
        r"^(bu|ve|ancak|çünkü|böylece|ayrıca|örneğin|son olarak|"
        r"birinci(?:si)?|ikinci(?:si)?|üçüncü(?:sü)?|dördüncü(?:sü)?|"
        r"beşinci(?:si)?|altıncı(?:sı)?|yedinci(?:si)?|sekizinci(?:si)?|"
        r"dokuzuncu(?:su)?|onuncu(?:su)?|this|and|but|because|for example)$",
        re.IGNORECASE,
    )

    def is_bridge_fragment(text: str) -> bool:
        clean = re.sub(r"[^\w\s]", "", text.casefold()).strip()
        words = clean.split()
        return bool(bridge_re.match(clean)) or (
            len(words) <= 2 and not any(char.isdigit() for char in clean)
        )

    def split_long_segment(
        start: float, end: float, text: str
    ) -> List[Tuple[float, float, str]]:
        duration = max(0.0, end - start)
        words = text.split()
        if duration <= hard_max_duration or len(words) < 5:
            return [(start, end, text)]

        pieces = max(2, math.ceil(duration / target_duration))
        words_per_piece = max(2, math.ceil(len(words) / pieces))
        chunks = [
            words[i : i + words_per_piece]
            for i in range(0, len(words), words_per_piece)
        ]
        result: List[Tuple[float, float, str]] = []
        cursor = start
        total_words = len(words)
        for idx, chunk in enumerate(chunks):
            chunk_end = (
                end
                if idx == len(chunks) - 1
                else cursor + duration * (len(chunk) / total_words)
            )
            result.append((cursor, chunk_end, " ".join(chunk)))
            cursor = chunk_end
        return result

    atomic_segments: List[Tuple[float, float, str]] = []
    for seg_start, seg_end, raw_text in segments:
        text = (raw_text or "").replace("\n", " ").strip()
        if text:
            atomic_segments.extend(
                split_long_segment(float(seg_start), float(seg_end), text)
            )

    grouped: List[Tuple[float, float, str]] = []
    start = 0.0
    end = 0.0
    parts: List[str] = []

    def flush():
        nonlocal start, end, parts
        if parts:
            grouped.append((start, end, " ".join(parts).strip()))
        start = 0.0
        end = 0.0
        parts = []

    for seg_start, seg_end, text in atomic_segments:
        bridge_fragment = is_bridge_fragment(text)
        if parts and (float(seg_end) - start) > hard_max_duration:
            flush()
        elif parts and transition_re.match(text) and (end - start) >= (min_duration / 2):
            flush()
        elif parts and not bridge_fragment and (end - start) >= target_duration:
            flush()

        if not parts:
            start = float(seg_start)
        end = max(float(seg_end), float(seg_start))
        parts.append(text)

        duration = end - start
        if duration >= hard_max_duration or (
            duration >= max_duration and not bridge_fragment
        ):
            flush()

    flush()

    # Move dangling connector words to the following scene. Subtitle timing can
    # split "Bu onları..." across cues; leaving "Bu" behind weakens both prompts.
    dangling_re = re.compile(r"\s+(bu|ve|ancak|çünkü|böylece)$", re.IGNORECASE)
    for i in range(len(grouped) - 1):
        match = dangling_re.search(grouped[i][2])
        if not match:
            continue
        connector = match.group(1)
        grouped[i] = (
            grouped[i][0],
            grouped[i][1],
            grouped[i][2][: match.start()].strip(),
        )
        grouped[i + 1] = (
            grouped[i + 1][0],
            grouped[i + 1][1],
            f"{connector} {grouped[i + 1][2]}".strip(),
        )

    # Merge weak, very short beats into a neighbour when that does not violate
    # the hard cap. This keeps the pace brisk without producing meaningless prompts.
    i = 0
    attempted_short_merges = set()
    while len(grouped) >= 2 and i < len(grouped):
        current = grouped[i]
        current_duration = current[1] - current[0]
        if current_duration >= min_duration:
            i += 1
            continue
        merge_key = (round(current[0], 3), round(current[1], 3), current[2])
        if merge_key in attempted_short_merges:
            i += 1
            continue
        attempted_short_merges.add(merge_key)

        if i + 1 < len(grouped):
            nxt = grouped[i + 1]
            replacements = split_long_segment(
                current[0],
                nxt[1],
                f"{current[2]} {nxt[2]}".strip(),
            )
            grouped[i : i + 2] = replacements
            if len(replacements) > 1:
                i += 1
            continue
        if i > 0:
            previous = grouped[i - 1]
            replacements = split_long_segment(
                previous[0],
                current[1],
                f"{previous[2]} {current[2]}".strip(),
            )
            grouped[i - 1 : i + 1] = replacements
            i = max(0, i - 1)
            if len(replacements) > 1:
                i += 1
            continue
        i += 1

    return grouped


def get_video_materials(
    task_id,
    params,
    video_terms,
    audio_duration,
    subtitle_path="",
    sub_maker=None,
    video_script="",
):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "pollinations_scene":
        logger.info("\n\n## generating scene-matched AI images (Pollinations)")
        segments = _build_timed_segments(
            task_id, params, subtitle_path, sub_maker, video_script
        )
        if not segments:
            # Zaman çizelgesi yoksa genel AI görsel moduna düş.
            logger.warning(
                "no timed segments available, falling back to generic AI images"
            )
            image_materials = material.generate_images_pollinations(
                task_id=task_id,
                search_terms=video_terms,
                video_aspect=params.video_aspect,
                audio_duration=audio_duration * params.video_count,
                max_clip_duration=params.video_clip_duration,
            )
        else:
            segments = _group_timed_segments(
                segments,
                min_duration=float(config.app.get("scene_min_duration", 1.8)),
                target_duration=float(config.app.get("scene_target_duration", 3.2)),
                max_duration=float(config.app.get("scene_max_duration", 4.8)),
                hard_max_duration=float(config.app.get("scene_hard_max_duration", 5.5)),
            )
            logger.info(f"grouped subtitle cues into {len(segments)} visual scenes")
            seg_texts = [t for _, _, t in segments]
            # KONU ANALİZİ: önce konuyu daralt (alan + tutarlı görsel dünya + tarih/mitoloji
            # uygun mu), sonra her sahne görselini bu dünyaya bağlı üret -> konudan kaymaz.
            topic_profile = llm.analyze_topic(params.video_subject, video_script)
            prompts = llm.generate_image_prompts_for_segments(
                seg_texts,
                params.video_subject,
                visual_world=topic_profile.get("visual_world", ""),
                allow_sensitive=topic_profile.get("allow_sensitive"),
            )
            image_materials = material.generate_segment_images(
                task_id=task_id,
                segments=segments,
                image_prompts=prompts,
                video_aspect=params.video_aspect,
                audio_duration=audio_duration,
            )
        if not image_materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to generate scene AI images, please try again.")
            return None
        target_res = VideoAspect(params.video_aspect).to_resolution()
        materials = video.preprocess_video(
            materials=image_materials,
            clip_duration=params.video_clip_duration,
            target_resolution=target_res,
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to process scene AI images into clips.")
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "pollinations":
        logger.info("\n\n## generating AI images with Pollinations")
        image_materials = material.generate_images_pollinations(
            task_id=task_id,
            search_terms=video_terms,
            video_aspect=params.video_aspect,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not image_materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to generate AI images, please try again.")
            return None
        # Üretilen görselleri zoom efektli video kliplerine dönüştür.
        materials = video.preprocess_video(
            materials=image_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error("failed to process AI images into clips.")
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_contact_mode=params.video_concat_mode,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    video_concat_mode = (
        params.video_concat_mode if params.video_count == 1 else VideoConcatMode.random
    )
    video_transition_mode = params.video_transition_mode

    # ----- INTRO (kapak görseli + ortada büyük başlık) -----
    # Sahne modunda, konuya özel bir kapak + başlık intro'su hazırla. İntro, videonun
    # ilk ~birkaç saniyesini KAPLAR (ilk sahne kliplerinin yerine geçer); bu sırada
    # dikkat çekici Türkçe hook SESLENDİRİLİR. Ses ve altyazı KAYDIRILMAZ -> senkron korunur.
    intro_clip_path = ""
    intro_dur = 0.0
    intro_enabled = bool(config.app.get("scene_intro_enabled", True))
    intro_target = max(1.2, min(float(config.app.get("scene_intro_duration", 2.2)), 3.0))
    if (
        params.video_source == "pollinations_scene"
        and intro_enabled
        and params.video_subject
        and downloaded_videos
    ):
        try:
            logger.info("\n\n## preparing intro (cover + title, hook narrated)")
            # Kapak, ilk sahnenin tüm süresini değil yalnızca başlığı okumaya yetecek
            # kısa ve sabit bir aralığı kaplar. İlk sahnenin kalan kısmı aşağıda korunur.
            intro_dur = round(intro_target, 3)
            _w, _h = VideoAspect(params.video_aspect).to_resolution()
            cover_prompt = llm.generate_cover_image_prompt(params.video_subject)
            cover = material.generate_cover_image(
                params.video_subject, params.video_aspect, cover_prompt=cover_prompt
            )

            # GERÇEK HAREKET: intro_motion açıksa, kapağı Pollinations video (i2v) ile
            # gerçekten hareket eden bir klibe çevir. Bakiye yoksa/başarısızsa statik kapağa düşer.
            _motion_clip = ""
            if bool(config.app.get("intro_motion_enabled", True)) and intro_dur > 0:
                try:
                    _req_dur = max(int(round(intro_dur)) + 1, 4)
                    _motion_prompt = (
                        f"{params.video_subject}, dramatic cinematic motion, "
                        f"slow camera movement, atmospheric, high detail"
                    )
                    _motion = path.join(utils.task_dir(task_id), "intro_motion.mp4")
                    if material.generate_motion_clip(
                        _motion_prompt, _motion, duration=_req_dur, width=_w, height=_h
                    ):
                        _motion_clip = _motion
                except Exception as e:
                    logger.warning(f"intro hareket klibi üretilemedi: {str(e)}")
                    _motion_clip = ""

            if (cover or _motion_clip) and intro_dur > 0:
                _font = path.join(
                    utils.font_dir(), params.font_name or "MicrosoftYaHeiBold.ttc"
                )
                _intro = path.join(utils.task_dir(task_id), "intro.mp4")
                if video.create_title_intro_clip(
                    cover, params.video_subject, _intro, intro_dur, _w, _h, _font,
                    motion_clip=_motion_clip,
                ):
                    intro_clip_path = _intro
                else:
                    intro_dur = 0.0
            else:
                intro_dur = 0.0
        except Exception as e:
            logger.warning(f"intro hazırlanamadı, intro'suz devam: {str(e)}")
            intro_clip_path = ""
            intro_dur = 0.0

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        if params.video_source == "pollinations_scene":
            # Sahne-eşleştirmeli mod: klipler segment süreleriyle/sırasıyla hazır;
            # karıştırmadan SIRAYLA birleştir. Varsa intro klibi, ilk kısa zaman aralığının
            # YERİNE geçer; ilk sahnenin kalan bölümü korunur ve süre eklenmez.
            _clip_list = video.replace_ordered_clip_prefix(
                clip_paths=downloaded_videos,
                intro_clip_path=intro_clip_path,
                intro_duration=intro_dur,
                output_dir=utils.task_dir(task_id),
            )
            video.assemble_ordered_clips(
                combined_video_path=combined_video_path,
                clip_paths=_clip_list,
                threads=params.n_threads,
            )
        else:
            video.combine_videos(
                combined_video_path=combined_video_path,
                video_paths=downloaded_videos,
                audio_file=audio_file,
                video_aspect=params.video_aspect,
                video_concat_mode=video_concat_mode,
                video_transition_mode=video_transition_mode,
                max_clip_duration=params.video_clip_duration,
                threads=params.n_threads,
            )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        # Ses ve altyazı KAYDIRILMAZ: hook, intro/kapak sahnesi üzerinde seslendirilir.
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id,
        params,
        video_terms,
        audio_duration,
        subtitle_path=subtitle_path,
        sub_maker=sub_maker,
        video_script=video_script,
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 仅完整视频生成流程才需要处理视频拼接模式；
    # 这样可以避免 /subtitle 和 /audio 这类请求访问不存在的字段。
    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    # 7. Cross-post to TikTok/Instagram (if enabled)
    cross_post_results = []
    if upload_post.upload_post_service.is_configured() and upload_post.upload_post_service.auto_upload:
        logger.info("\n\n## cross-posting videos to TikTok/Instagram")
        _subj = (params.video_subject or "").strip()
        _caption = (
            f"{_subj}\n\n"
            "#bunubiliyormuydun #ilginçbilgiler #biliyormuydun #keşfet #shorts #fyp"
            if _subj
            else "Bunu biliyor muydun abi? 🧠 #bunubiliyormuydun #ilginçbilgiler #keşfet #shorts #fyp"
        )
        for video_path in final_video_paths:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=_caption,
            )
            cross_post_results.append(result)
            if result.get('success'):
                logger.info(f"✅ Cross-posted: {video_path}")
            else:
                logger.warning(f"⚠️ Failed to cross-post: {video_path} - {result.get('error', 'Unknown error')}")

    # 7b. YouTube'a ücretsiz resmi API ile otomatik yükle (config ile aç/kapa).
    if config.app.get("youtube_auto_upload", False):
        try:
            from app.services import youtube_upload
            if youtube_upload.is_authorized():
                logger.info("\n\n## uploading videos to YouTube (free official API)")
                _subj = (params.video_subject or "").strip()
                _desc = (
                    f"{_subj}\n\n"
                    "#bunubiliyormuydun #ilginçbilgiler #biliyormuydun #keşfet #shorts #fyp"
                )
                _yt_privacy = str(config.app.get("youtube_privacy", "public")).strip() or "public"
                for video_path in final_video_paths:
                    yt = youtube_upload.upload_video(
                        video_path=video_path,
                        title=(_subj or "Bunu biliyor muydun abi?")[:100],
                        description=_desc,
                        tags=["bunubiliyormuydun", "ilginçbilgiler", "shorts", "keşfet"],
                        privacy=_yt_privacy,
                    )
                    cross_post_results.append(yt)
                    if yt.get("success"):
                        logger.info(f"✅ YouTube: {yt.get('url')}")
                    else:
                        logger.warning(f"⚠️ YouTube yükleme başarısız: {yt.get('error')}")
            else:
                logger.warning("YouTube auto-upload açık ama yetki yok (önce authorize gerekli).")
        except Exception as e:
            logger.warning(f"YouTube auto-upload hatası: {e}")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "cross_post_results": cross_post_results if cross_post_results else None,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
