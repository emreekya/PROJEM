import base64
import math
import os
import random
import threading
import time
from typing import List, Tuple
from urllib.parse import quote, urlencode

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect, VideoConcatMode
from app.utils import utils

# Thread-safe counter for API key rotation
_api_key_counter = 0
_api_key_lock = threading.Lock()

# Pollinations token rotasyonu için ayrı sayaç (birden çok hesap arasında dönüşüm).
_pollinations_token_counter = 0
_pollinations_token_lock = threading.Lock()

# Cloudflare günlük kotası (neuron) bittiğinde True olur; aynı çalışmada bir daha
# Cloudflare denenmez, doğrudan Pollinations'a geçilir (otomatik yedekleme).
# Workers Paid hesabında günlük ücretsiz tahsis aşılınca ücretli kullanım devam eder.
# Bu bayrak yalnızca Cloudflare gerçekten kota hatası döndürürse set edilir.
_cloudflare_quota_exhausted = False
# Kotası biten Cloudflare hesaplarını işaretlemek için küme (öncelik sırasıyla denenir).
_cloudflare_exhausted_accounts = set()


def _cloudflare_account_pairs() -> List[Tuple[str, str]]:
    """config'deki Cloudflare (account_id, api_key) çiftlerini döndürür.

    Hem tek string hem de eşit uzunlukta LİSTE desteklenir (indeksle eşlenir).
    Böylece birden çok ücretsiz hesabın kotası birleştirilebilir.
    """
    accounts = config.app.get("cloudflare_account_id")
    keys = config.app.get("cloudflare_api_key")
    acc_list = accounts if isinstance(accounts, list) else ([accounts] if accounts else [])
    key_list = keys if isinstance(keys, list) else ([keys] if keys else [])
    acc_list = [str(a).strip() for a in acc_list if a and str(a).strip()]
    key_list = [str(k).strip() for k in key_list if k and str(k).strip()]
    n = min(len(acc_list), len(key_list))
    return list(zip(acc_list[:n], key_list[:n]))


def _get_pollinations_token() -> str:
    """config'deki Pollinations token'ını döndürür.

    Tek string verilebilir veya birden çok token'lı bir LİSTE verilebilir.
    Liste verilmişse her çağrıda sırayla (round-robin) farklı token kullanılır;
    böylece rate-limit (402) birden çok hesaba dağıtılır.
    """
    tokens = config.app.get("pollinations_api_key")
    if not tokens:
        return ""
    if isinstance(tokens, str):
        return tokens.strip()
    tokens = [t.strip() for t in tokens if isinstance(t, str) and t.strip()]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    global _pollinations_token_counter
    with _pollinations_token_lock:
        _pollinations_token_counter += 1
        return tokens[_pollinations_token_counter % len(tokens)]


def _get_tls_verify() -> bool:
    # 默认开启 TLS 证书校验，防止素材搜索和下载过程被中间人篡改。
    # 仅在企业代理、自签证书等明确需要的场景下，允许用户通过
    # `config.toml` 显式设置 `tls_verify = false` 临时关闭。
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")

    if not tls_verify:
        logger.warning(
            "TLS certificate verification is disabled by config.app.tls_verify=false. "
            "Only use this in trusted proxy environments."
        )

    return bool(tls_verify)


def _is_cloudflare_daily_quota_exhausted(status_code: int, body: str) -> bool:
    """Gerçek günlük Neuron kotası hatasını geçici rate-limit yanıtından ayırır."""
    normalized_body = (body or "").lower()
    quota_terms = ("quota", "limit", "exceed", "allocation")
    return (
        status_code == 429
        and "neuron" in normalized_body
        and any(term in normalized_body for term in quota_terms)
    )


def get_api_key(cfg_key: str):
    api_keys = config.app.get(cfg_key)
    if not api_keys:
        raise ValueError(
            f"\n\n##### {cfg_key} is not set #####\n\nPlease set it in the config.toml file: {config.config_file}\n\n"
            f"{utils.to_json(config.app)}"
        )

    # if only one key is provided, return it
    if isinstance(api_keys, str):
        return api_keys

    global _api_key_counter
    with _api_key_lock:
        _api_key_counter += 1
        return api_keys[_api_key_counter % len(api_keys)]


def search_videos_pexels(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)
    video_orientation = aspect.name
    video_width, video_height = aspect.to_resolution()
    api_key = get_api_key("pexels_api_keys")
    headers = {
        "Authorization": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    # Build URL
    params = {"query": search_term, "per_page": 20, "orientation": video_orientation}
    query_url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url,
            headers=headers,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, 60),
        )
        response = r.json()
        video_items = []
        if "videos" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["videos"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["video_files"]
            # loop through each url to determine the best quality
            for video in video_files:
                w = int(video["width"])
                h = int(video["height"])
                if w == video_width and h == video_height:
                    item = MaterialInfo()
                    item.provider = "pexels"
                    item.url = video["link"]
                    item.duration = duration
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def search_videos_pixabay(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    aspect = VideoAspect(video_aspect)

    video_width, video_height = aspect.to_resolution()

    api_key = get_api_key("pixabay_api_keys")
    # Build URL
    params = {
        "q": search_term,
        "video_type": "all",  # Accepted values: "all", "film", "animation"
        "per_page": 50,
        "key": api_key,
    }
    query_url = f"https://pixabay.com/api/videos/?{urlencode(params)}"
    logger.info(f"searching videos: {query_url}, with proxies: {config.proxy}")

    try:
        r = requests.get(
            query_url, proxies=config.proxy, verify=_get_tls_verify(), timeout=(30, 60)
        )
        response = r.json()
        video_items = []
        if "hits" not in response:
            logger.error(f"search videos failed: {response}")
            return video_items
        videos = response["hits"]
        # loop through each video in the result
        for v in videos:
            duration = v["duration"]
            # check if video has desired minimum duration
            if duration < minimum_duration:
                continue
            video_files = v["videos"]
            # loop through each url to determine the best quality
            for video_type in video_files:
                video = video_files[video_type]
                w = int(video["width"])
                h = int(video["height"])
                if w == 0 or h == 0:
                    continue
                # Yön uyuşmalı: dikey video istendiğinde yatay klipleri alma.
                # Aksi halde yatay klip dikey çerçevede siyah bantlarla görünür.
                if (h >= w) != (video_height >= video_width):
                    break  # bu videonun tüm varyantları aynı yönde -> tamamını atla
                if w >= video_width:
                    item = MaterialInfo()
                    item.provider = "pixabay"
                    item.url = video["url"]
                    item.duration = duration
                    video_items.append(item)
                    break
        return video_items
    except Exception as e:
        logger.error(f"search videos failed: {str(e)}")

    return []


def save_video(video_url: str, save_dir: str = "") -> str:
    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url_without_query = video_url.split("?")[0]
    url_hash = utils.md5(url_without_query)
    video_id = f"vid-{url_hash}"
    video_path = f"{save_dir}/{video_id}.mp4"

    # if video already exists, return the path
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"video already exists: {video_path}")
        return video_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    # if video does not exist, download it
    with open(video_path, "wb") as f:
        f.write(
            requests.get(
                video_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(60, 240),
            ).content
        )

    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        clip = None
        try:
            clip = VideoFileClip(video_path)
            duration = clip.duration
            fps = clip.fps
            if duration > 0 and fps > 0:
                return video_path
        except Exception as e:
            logger.warning(f"invalid video file: {video_path} => {str(e)}")
            try:
                os.remove(video_path)
            except Exception as remove_error:
                logger.warning(
                    f"failed to remove invalid video file: {video_path}, error: {str(remove_error)}"
                )
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception as close_error:
                    logger.warning(
                        f"failed to close video clip: {video_path}, error: {str(close_error)}"
                    )
    return ""


def _generate_one_pollinations_image(
    prompt: str, width: int, height: int, seed: int, save_dir: str
) -> str:
    """Pollinations ile tek bir AI görsel üretir, local_videos klasörüne kaydeder ve dosya adını döndürür.

    Pollinations görsel API'si ücretsiz ve anahtarsızdır. Başarısız olursa boş string döner.
    """
    base = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {
        "width": width,
        "height": height,
        "seed": seed,
        "nologo": "true",
        "model": "flux",
    }
    # İsteğe bağlı Pollinations token'ı (auth.pollinations.ai'den ücretsiz alınır)
    # rate-limit'i (402) ciddi şekilde yükseltir. config.toml -> pollinations_api_key.
    # Birden çok token verilirse her çağrıda sırayla farklı hesap kullanılır.
    token = _get_pollinations_token()
    if token:
        params["token"] = token
    image_url = f"{base}?{urlencode(params)}"
    file_name = f"aiimg-{utils.md5(prompt + str(seed))}.jpg"
    file_path = os.path.join(save_dir, file_name)

    # Daha önce üretildiyse tekrar üretme (önbellek).
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_name

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            r = requests.get(
                image_url,
                headers=headers,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 180),
            )
            if r.status_code == 200 and len(r.content) > 2000:
                with open(file_path, "wb") as f:
                    f.write(r.content)
                if os.path.getsize(file_path) > 0:
                    return file_name
            else:
                logger.warning(
                    f"pollinations image attempt {attempt + 1} status {r.status_code}, "
                    f"size {len(r.content)}"
                )
        except Exception as e:
            logger.warning(f"pollinations image attempt {attempt + 1} failed: {str(e)}")
        # Rate-limit (402) ve geçici hatalara karşı üssel bekleme (en fazla ~12 sn).
        if attempt < max_attempts - 1:
            time.sleep(min(2.0 * (2 ** attempt), 12.0))
    return ""


def _generate_one_cloudflare_image(
    prompt: str, width: int, height: int, seed: int, save_dir: str,
    steps_override: int = 0,
) -> str:
    """Cloudflare Workers AI ile tek bir AI görsel üretir (HESAP bazlı, IP'den bağımsız).

    Ücretsiz katman günlük tahsis sunar; Workers Paid hesabında aynı API kimlik
    bilgileriyle tahsis üzerindeki kullanım ücretlendirilerek devam eder.
    Başarısız olursa boş string döner.
    """
    pairs = _cloudflare_account_pairs()
    if not pairs:
        logger.error(
            "cloudflare_account_id / cloudflare_api_key ayarlı değil (config.toml)"
        )
        return ""

    model = (
        config.app.get("cloudflare_image_model")
        or "@cf/black-forest-labs/flux-1-schnell"
    ).strip()
    # Adım (steps) sayısı neuron maliyetini doğrudan etkiler. flux-schnell az adımda
    # (1-4) çalışacak şekilde tasarlanmıştır; düşürmek günlük kotayı uzatır.
    try:
        steps = int(steps_override) if steps_override else int(
            config.app.get("cloudflare_image_steps", 4)
        )
    except (TypeError, ValueError):
        steps = 4
    steps = max(1, min(steps, 8))
    file_name = f"aiimg-{utils.md5(prompt + str(seed) + model + str(steps))}.jpg"
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_name

    payload = {"prompt": prompt, "steps": steps}
    # Boyut desteği: flux-1-schnell boyut almaz (kare üretip kırpılır), ama Leonardo
    # (Phoenix/Lucid) ve SDXL gibi modeller width/height destekler → doğrudan dikey
    # 9:16 üretip kırpmayı/letterbox bulanıklığını azaltır. flux-schnell modelinde
    # bu alanlar yok sayılır, sorun çıkarmaz.
    if "flux-1-schnell" not in model:
        payload["width"] = int(width)
        payload["height"] = int(height)

    # KATI ÖNCELİK SIRASI: hesaplar config'deki sırayla denenir.
    # Önce 1. hesap; yalnızca günlük Neuron kotası gerçekten bittiğinde işaretlenip
    # 2. hesaba geçilir. Geçici 429 rate-limit yanıtları hesabı kalıcı olarak elemez.
    # (Geçici hatada hesap işaretlenmez, sadece sıradaki denenir; sonraki görselde
    #  yine 1. hesap baştan denenir.)
    available = [(a, k) for (a, k) in pairs if a not in _cloudflare_exhausted_accounts]
    _gw = (config.app.get("cloudflare_ai_gateway") or "").strip()
    try:
        _gw_ttl = int(config.app.get("cloudflare_ai_gateway_cache_ttl", 0) or 0)
    except (TypeError, ValueError):
        _gw_ttl = 0
    for account_id, api_key in available:
        if _gw:
            # AI Gateway üzerinden: önbellek + maliyet analizi + hız sınırı.
            url = f"https://gateway.ai.cloudflare.com/v1/{account_id}/{_gw}/workers-ai/{model}"
        else:
            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {"Authorization": f"Bearer {api_key}"}
        if _gw and _gw_ttl > 0:
            # Aynı prompt+seed tekrar gelirse cache'den dön (neuron harcama).
            headers["cf-aig-cache-ttl"] = str(_gw_ttl)
        try:
            r = requests.post(
                url,
                headers=headers,
                json=payload,
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, 180),
            )
            if r.status_code == 200:
                content_type = r.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    # flux-1-schnell: {"result": {"image": "<base64 jpeg>"}, "success": true}
                    data = r.json()
                    img_b64 = (data.get("result") or {}).get("image")
                    if img_b64:
                        raw = base64.b64decode(img_b64)
                        if len(raw) > 2000:
                            with open(file_path, "wb") as f:
                                f.write(raw)
                            if os.path.getsize(file_path) > 0:
                                return file_name
                    logger.warning(f"cloudflare image: yanıtta görsel yok: {str(data)[:200]}")
                else:
                    # Bazı modeller (SDXL) ham PNG bayt döndürür.
                    if len(r.content) > 2000:
                        with open(file_path, "wb") as f:
                            f.write(r.content)
                        if os.path.getsize(file_path) > 0:
                            return file_name
            else:
                body = r.text[:200]
                # Bu hesabın günlük Neuron kotası bittiyse onu işaretle ve SONRAKİ
                # hesaba geç. Workers Paid hesabında tahsis üstü kullanım ücretli
                # devam ettiği için genel 429 yanıtlarını kota bitişi sayma.
                if _is_cloudflare_daily_quota_exhausted(r.status_code, body):
                    _cloudflare_exhausted_accounts.add(account_id)
                    logger.warning(
                        f"cloudflare hesabı (...{account_id[-6:]}) kotası bitti, "
                        f"sonraki hesaba geçiliyor"
                    )
                    continue
                if r.status_code == 429:
                    logger.warning(
                        f"cloudflare image rate-limit (geçici), hesap sonraki istekte "
                        f"yeniden denenecek: {body}"
                    )
                    continue
                logger.warning(f"cloudflare image status {r.status_code}: {body}")
        except Exception as e:
            logger.warning(f"cloudflare image request failed: {str(e)}")

    # Buraya gelindiyse hiçbir hesap görsel üretemedi.
    # Tüm hesapların kotası bittiyse global bayrağı set et -> Pollinations'a geç.
    if pairs and all(a in _cloudflare_exhausted_accounts for a, _ in pairs):
        global _cloudflare_quota_exhausted
        _cloudflare_quota_exhausted = True
        logger.error(
            "Tüm Cloudflare hesaplarının günlük kotası bitti -> Pollinations'a geçiliyor"
        )
    return ""


def _ai_image_provider() -> str:
    """Aktif AI görsel sağlayıcısını döndürür: 'cloudflare' veya 'pollinations'.

    config.app['ai_image_provider'] açıkça ayarlandıysa onu kullanır; aksi halde
    Cloudflare kimlik bilgileri varsa (IP-bağımsız, güvenilir) Cloudflare'i seçer.
    """
    explicit = (config.app.get("ai_image_provider") or "").strip().lower()
    if explicit in ("cloudflare", "pollinations"):
        return explicit
    if config.app.get("cloudflare_account_id") and config.app.get("cloudflare_api_key"):
        return "cloudflare"
    return "pollinations"


def _generate_one_ai_image(
    prompt: str, width: int, height: int, seed: int, save_dir: str,
    steps_override: int = 0,
) -> str:
    """Yapılandırmaya göre uygun sağlayıcıdan tek bir AI görsel üretir.

    Cloudflare seçiliyken günlük kota biterse (429), otomatik olarak Pollinations'a
    geçilir; böylece üretim yarıda kesilmez, sadece sağlayıcı değişir.

    `steps_override` verilirse (örn. kapak görseli için daha yüksek kalite), Cloudflare
    flux adım sayısı global ayar yerine bu değerle çalışır.
    """
    # Her görsele güçlü "yazı yok" eki: flux, diyagram/etiket benzeri sahnelerde
    # (örn. beyin-yemek görseli) uyduruk İngilizce etiketler basmaya çok meyilli.
    # Bu ek, hem sahne hem kapak görsellerinde uyduruk metin/etiket riskini düşürür.
    _no_text = (
        " | absolutely no text, no letters, no words, no captions, no labels, "
        "no numbers, no logos, no watermarks, no signs, no typography anywhere; "
        "no readable names or writing on trophies, jerseys, scoreboards, plaques, "
        "banners or screens; surfaces must be blank and wordless"
    )
    if _no_text.strip() not in prompt:
        prompt = prompt.rstrip() + _no_text

    if _ai_image_provider() == "cloudflare" and not _cloudflare_quota_exhausted:
        result = _generate_one_cloudflare_image(
            prompt, width, height, seed, save_dir, steps_override=steps_override
        )
        if result:
            return result
        # Kota bittiyse Pollinations'a düş (token'lar config'de varsa kullanılır).
        if _cloudflare_quota_exhausted:
            return _generate_one_pollinations_image(prompt, width, height, seed, save_dir)
        return result
    return _generate_one_pollinations_image(prompt, width, height, seed, save_dir)


def _ai_image_request_delay() -> float:
    """Sağlayıcıya göre istekler arası bekleme. Cloudflare hesap bazlı olduğu için kısa.

    Cloudflare kotası bitip Pollinations'a düşüldüyse daha uzun bekleme uygulanır.
    """
    if _ai_image_provider() == "cloudflare" and not _cloudflare_quota_exhausted:
        return 0.2
    return 0.8


def generate_images_pollinations(
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect = VideoAspect.portrait,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
) -> List[MaterialInfo]:
    """Pollinations AI ile konuya özel görseller üretir (stok video yerine).

    Üretilen görseller `local_videos` klasörüne kaydedilir; çağıran taraf bunları
    `video.preprocess_video` ile zoom efektli kliplere dönüştürür.
    """
    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()

    if not search_terms:
        search_terms = ["cinematic background"]

    # Sesi kaplayacak kadar görsel üret (+1 tampon).
    clip_seconds = max(1, int(max_clip_duration))
    needed = max(len(search_terms), math.ceil((audio_duration or clip_seconds) / clip_seconds) + 1)
    needed = min(needed, 40)  # aşırı uzun videolarda sınırla

    save_dir = utils.storage_dir("local_videos", create=True)

    # Her görsel için zenginleştirilmiş bir görsel istem (prompt) hazırla.
    jobs = []
    for i in range(needed):
        term = search_terms[i % len(search_terms)]
        prompt = (
            f"{term}, cinematic photography, highly detailed, dramatic lighting, "
            f"professional, ultra realistic, 4k"
        )
        jobs.append((prompt, 1000 + i))

    logger.info(f"generating {len(jobs)} AI images via Pollinations ({width}x{height})...")
    materials: List[MaterialInfo] = []

    # AI sağlayıcısından SIRAYLA üret (Pollinations'ta IP rate-limit'i, Cloudflare'de
    # nazik istek hızı için). Sağlayıcı config'e göre otomatik seçilir.
    for idx, (prompt, seed) in enumerate(jobs):
        file_name = _generate_one_ai_image(prompt, width, height, seed, save_dir)
        if file_name:
            item = MaterialInfo()
            item.provider = _ai_image_provider()
            item.url = file_name
            item.duration = max_clip_duration
            materials.append(item)
            logger.info(f"AI image ready: {file_name}")
        if idx < len(jobs) - 1:
            time.sleep(_ai_image_request_delay())

    logger.success(f"generated {len(materials)} AI images via Pollinations")
    return materials


def generate_motion_clip(
    prompt: str,
    out_path: str,
    duration: int = 4,
    width: int = 1080,
    height: int = 1920,
    image_url: str = "",
) -> bool:
    """Pollinations video API (gen.pollinations.ai) ile GERÇEKTEN hareket eden bir klip üretir.

    text-to-video (image_url boşsa) veya image-to-video (image_url verilirse).
    ÜCRETLİDİR (Pollen kredisi). Bakiye yoksa/başarısızsa False döner; çağıran taraf
    statik (zoom) kapağa düşer. Başarılıysa mp4 `out_path`'e yazılır.
    """
    token = _get_pollinations_token()
    if not token:
        logger.warning("pollinations video: token yok (Pollen gerekli)")
        return False
    model = (config.app.get("intro_motion_model") or "ltx-2").strip()
    params = {
        "model": model,
        "duration": int(duration),
        "width": int(width),
        "height": int(height),
        "seed": 7,
        "token": token,
    }
    if image_url:
        params["image"] = image_url
    url = f"https://gen.pollinations.ai/video/{quote(prompt)}?{urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }
    try:
        r = requests.get(
            url, headers=headers, proxies=config.proxy,
            verify=_get_tls_verify(), timeout=(30, 300),
        )
        content_type = r.headers.get("Content-Type", "").lower()
        if (
            r.status_code == 200
            and "json" not in content_type
            and len(r.content) > 10000
        ):
            with open(out_path, "wb") as f:
                f.write(r.content)
            if os.path.getsize(out_path) > 0:
                logger.success(f"pollinations video (hareket) üretildi: {out_path}")
                return True
        # Hata (genelde 402: yetersiz Pollen bakiyesi).
        logger.warning(
            f"pollinations video başarısız (statik kapağa düşülüyor): "
            f"{r.status_code} {r.text[:160]}"
        )
    except Exception as e:
        logger.warning(f"pollinations video hatası (statik kapağa düşülüyor): {str(e)}")
    return False


def generate_cover_image(
    video_subject: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    cover_prompt: str = "",
) -> str:
    """Konuya özel, temiz bir sosyal medya KAPAK görseli üretir.

    Tam dosya yolunu döndürür; üretilemezse boş string.
    """
    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()
    save_dir = utils.storage_dir("local_videos", create=True)
    visual_direction = (cover_prompt or "").strip() or (
        f"Create a clear contemporary editorial visual that directly represents '{video_subject}'. "
        "Use one relevant focal subject and a polished social-media thumbnail composition."
    )
    prompt = (
        f"Create a premium, cinematic vertical BACKGROUND IMAGE for this topic: "
        f"'{video_subject}'. Art direction: {visual_direction}. It is a clean photographic/illustrative "
        f"SCENE, NOT a poster, flyer or magazine. Leave calm empty negative space in the upper third "
        f"(a separate layer adds the headline). Place the topic-specific focal subject in the lower "
        f"two-thirds. Layered depth, cinematic studio lighting, high contrast, crisp details, sharp "
        f"focus, 4k, professional photography. Keep every object directly relevant to the topic. Do not "
        f"add unrelated religious, military, historical, mythological, fantasy or landmark imagery. "
        f"CRITICAL — ZERO TEXT: never include a poster, flyer, sign, banner, book, label or decorative "
        f"border with writing. The image must contain absolutely NO text, letters, words, captions, "
        f"titles, numbers, signs, logos, watermarks or typography anywhere. Pure wordless imagery only."
    )
    # Kapak tek bir görseldir (sahnelere göre nadir) -> kaliteyi maksimuma çek:
    # flux-schnell için en yüksek adım sayısı (8), daha az artefakt ve uyduruk yazı.
    file_name = _generate_one_ai_image(
        prompt, width, height, 7777, save_dir, steps_override=6
    )
    if file_name:
        return os.path.join(save_dir, file_name)
    return ""


def generate_segment_images(
    task_id: str,
    segments: List[Tuple[float, float, str]],
    image_prompts: List[str],
    video_aspect: VideoAspect = VideoAspect.portrait,
    audio_duration: float = 0.0,
) -> List[MaterialInfo]:
    """Her seslendirme segmenti için, o segmentin içeriğine özel bir AI görsel üretir.

    `segments`: (start, end, text) üçlülerinin SIRALI listesi (altyazı zaman çizelgesi).
    `image_prompts`: segmentlerle birebir hizalı İngilizce görsel istemleri.
    Dönen MaterialInfo listesi SEGMENT SIRASINI korur ve her birinin `duration` alanı
    o segmentin (bir sonraki segmente kadar olan) süresine eşittir; böylece klipler
    sırayla birleştirildiğinde seslendirmeyle hizalanır.
    """
    if not segments:
        return []

    aspect = VideoAspect(video_aspect)
    width, height = aspect.to_resolution()
    save_dir = utils.storage_dir("local_videos", create=True)

    n = len(segments)
    # Her segmentin görsel süresi: bu segmentin başından bir SONRAKİ segmentin başına kadar.
    # Son segment için: kendi (start, end) aralığı. Böylece klipler bitişik olur, kayma olmaz.
    durations: List[float] = []
    for i in range(n):
        start_i = segments[i][0]
        if i < n - 1:
            dur = segments[i + 1][0] - start_i
        else:
            # Son klip: ses sonuna kadar uzasın ki sondaki sessizlikte de görsel
            # kalsın (video süresi = ses süresi olur, kayma/kesilme olmaz).
            seg_dur = segments[i][1] - start_i
            tail = (audio_duration - start_i) if audio_duration else seg_dur
            dur = max(seg_dur, tail)
        durations.append(max(0.5, round(float(dur), 3)))

    logger.info(
        f"generating {n} scene-matched AI images via {_ai_image_provider()} ({width}x{height})..."
    )

    # Görseller SIRAYLA üretilir (Pollinations IP rate-limit'i / Cloudflare nazik hız).
    # Sağlayıcı config'e göre otomatik seçilir (Cloudflare varsa o, yoksa Pollinations).
    # (Sıralı üretim ayrıca WebUI log sink'inin alt-thread'den çağrılmasını da önler.)
    file_names: List[str] = ["" for _ in range(n)]
    for i in range(n):
        prompt = image_prompts[i] if i < len(image_prompts) else ""
        prompt = (prompt or "").strip() or segments[i][2].strip() or "cinematic scene"
        file_name = _generate_one_ai_image(prompt, width, height, 1000 + i, save_dir)
        file_names[i] = file_name
        if file_name:
            logger.info(f"scene image {i + 1}/{n} ready: {file_name}")
        if i < n - 1:
            time.sleep(_ai_image_request_delay())

    # Hiç görsel üretilemediyse boş dön (çağıran taraf hatayı ele alır).
    if not any(file_names):
        logger.error("no scene images could be generated")
        return []

    # Eksik (başarısız) görselleri komşu başarılı görselle doldur: önce önceki,
    # yoksa sonraki başarılı görsel. Böylece her segment için bir klip garanti olur
    # ve zaman çizelgesinde boşluk kalmaz.
    for i in range(n):  # ileri geçiş: önceki başarılıyı taşı
        if not file_names[i] and i > 0 and file_names[i - 1]:
            file_names[i] = file_names[i - 1]
    for i in range(n - 1, -1, -1):  # geri geçiş: sonraki başarılıyı baştaki boşluklara taşı
        if not file_names[i] and i < n - 1 and file_names[i + 1]:
            file_names[i] = file_names[i + 1]

    provider_name = _ai_image_provider()
    materials: List[MaterialInfo] = []
    for i in range(n):
        if not file_names[i]:
            continue
        item = MaterialInfo()
        item.provider = provider_name
        item.url = file_names[i]
        item.duration = durations[i]
        materials.append(item)

    logger.success(f"generated {len(materials)} scene-matched AI images")
    return materials


def download_videos(
    task_id: str,
    search_terms: List[str],
    source: str = "pexels",
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_contact_mode: VideoConcatMode = VideoConcatMode.random,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
) -> List[str]:
    valid_video_items = []
    valid_video_urls = []
    found_duration = 0.0
    # Birden fazla kaynağı tek modda birleştirebilmek için sağlayıcı listesi kuruyoruz.
    # "combined" modunda Pexels ve Pixabay aynı anda aranır, sonuçlar birleştirilir.
    if source == "combined":
        search_providers = [search_videos_pexels, search_videos_pixabay]
    elif source == "pixabay":
        search_providers = [search_videos_pixabay]
    else:
        search_providers = [search_videos_pexels]

    for search_term in search_terms:
        video_items = []
        for provider in search_providers:
            try:
                video_items.extend(
                    provider(
                        search_term=search_term,
                        minimum_duration=max_clip_duration,
                        video_aspect=video_aspect,
                    )
                )
            except Exception as e:
                # Bir kaynak (ör. eksik API anahtarı) başarısız olsa bile diğer kaynak çalışmaya devam etsin.
                logger.error(f"provider search failed for '{search_term}': {str(e)}")
        logger.info(f"found {len(video_items)} videos for '{search_term}'")

        for item in video_items:
            if item.url not in valid_video_urls:
                valid_video_items.append(item)
                valid_video_urls.append(item.url)
                found_duration += item.duration

    logger.info(
        f"found total videos: {len(valid_video_items)}, required duration: {audio_duration} seconds, found duration: {found_duration} seconds"
    )
    video_paths = []

    material_directory = config.app.get("material_directory", "").strip()
    if material_directory == "task":
        material_directory = utils.task_dir(task_id)
    elif material_directory and not os.path.isdir(material_directory):
        material_directory = ""

    concat_mode_value = getattr(video_contact_mode, "value", video_contact_mode)
    if concat_mode_value == VideoConcatMode.random.value:
        random.shuffle(valid_video_items)

    total_duration = 0.0
    for item in valid_video_items:
        try:
            logger.info(f"downloading video: {item.url}")
            saved_video_path = save_video(
                video_url=item.url, save_dir=material_directory
            )
            if saved_video_path:
                logger.info(f"video saved: {saved_video_path}")
                video_paths.append(saved_video_path)
                seconds = min(max_clip_duration, item.duration)
                total_duration += seconds
                if total_duration > audio_duration:
                    logger.info(
                        f"total duration of downloaded videos: {total_duration} seconds, skip downloading more"
                    )
                    break
        except Exception as e:
            logger.error(f"failed to download video: {utils.to_json(item)} => {str(e)}")
    logger.success(f"downloaded {len(video_paths)} videos")
    return video_paths


if __name__ == "__main__":
    download_videos(
        "test123", ["Money Exchange Medium"], audio_duration=100, source="pixabay"
    )
