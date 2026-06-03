"""
YouTube'a ücretsiz, resmi API ile otomatik video (Shorts) yükleme.

- Tek seferlik OAuth yetkilendirmesi: `client_secret.json` (Desktop OAuth) ile
  tarayıcı açılır, kullanıcı izin verir, kalıcı token `youtube_token.json`'a yazılır.
- Sonraki yüklemeler token'ı (gerekirse otomatik yenileyerek) kullanır.

Bağımlılıklar: google-api-python-client, google-auth-oauthlib, google-auth-httplib2
"""
import os

from loguru import logger

# Proje kökü
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
CLIENT_SECRET_FILE = os.path.join(_ROOT, "client_secret.json")
TOKEN_FILE = os.path.join(_ROOT, "youtube_token.json")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def is_authorized() -> bool:
    """Kalıcı token var mı (yani daha önce izin verilmiş mi)?"""
    return os.path.exists(TOKEN_FILE)


def is_configured() -> bool:
    """client_secret.json mevcut mu?"""
    return os.path.exists(CLIENT_SECRET_FILE)


def _load_credentials(interactive: bool = False):
    """Geçerli OAuth kimlik bilgilerini döndürür.

    `interactive=True` ve token yoksa, tarayıcı tabanlı tek seferlik yetkilendirme
    akışını çalıştırır (kullanıcı 'İzin Ver' demeli). Aksi halde sadece kayıtlı token'ı
    yükler/yeniler; yoksa None döner.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            logger.warning(f"youtube token okunamadı, yeniden yetkilendirme gerekebilir: {e}")
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds)
            return creds
        except Exception as e:
            logger.warning(f"youtube token yenilenemedi: {e}")
            creds = None

    if interactive:
        if not os.path.exists(CLIENT_SECRET_FILE):
            raise FileNotFoundError(
                "client_secret.json bulunamadı (proje kökünde olmalı)."
            )
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        # Yerel sunucu açıp tarayıcıda onay alır; kullanıcı 'İzin Ver' demeli.
        creds = flow.run_local_server(port=0, prompt="consent")
        _save_credentials(creds)
        return creds

    return None


def _save_credentials(creds) -> None:
    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    except Exception as e:
        logger.warning(f"youtube token kaydedilemedi: {e}")


def authorize() -> bool:
    """Tek seferlik tarayıcı yetkilendirmesini çalıştırır. Başarılıysa True."""
    try:
        creds = _load_credentials(interactive=True)
        return bool(creds and creds.valid)
    except Exception as e:
        logger.error(f"youtube yetkilendirme başarısız: {e}")
        return False


def upload_video(
    video_path: str,
    title: str,
    description: str = "",
    tags=None,
    privacy: str = "public",
    category_id: str = "27",  # 27 = Education; 24 = Entertainment
    made_for_kids: bool = False,
) -> dict:
    """Videoyu YouTube'a yükler. Dönen dict: {success, video_id, url} ya da {success:False, error}.

    YouTube başlığı en fazla 100 karakter; açıklama 5000 karakter.
    """
    if not os.path.exists(video_path):
        return {"success": False, "error": f"video bulunamadı: {video_path}"}

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except Exception as e:
        return {"success": False, "error": f"google kütüphaneleri yok: {e}"}

    creds = _load_credentials(interactive=False)
    if not creds:
        return {
            "success": False,
            "error": "YouTube yetkisi yok. Önce authorize() çalıştırılmalı.",
        }

    body = {
        "snippet": {
            "title": (title or "Video")[:100],
            "description": (description or "")[:4900],
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }
    last_err = ""
    for attempt in range(2):  # geçici hatada bir kez daha dene
        try:
            youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
            media = MediaFileUpload(
                video_path, chunksize=-1, resumable=True, mimetype="video/*"
            )
            request = youtube.videos().insert(
                part="snippet,status", body=body, media_body=media
            )
            logger.info(
                f"YouTube'a yükleniyor (deneme {attempt + 1}): "
                f"{os.path.basename(video_path)} ..."
            )
            response = None
            while response is None:
                _status, response = request.next_chunk()
            video_id = response.get("id")
            url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
            logger.success(f"✅ YouTube'a yüklendi: {url}")
            return {"success": True, "video_id": video_id, "url": url}
        except Exception as e:
            last_err = str(e)
            logger.warning(f"YouTube yükleme denemesi {attempt + 1} başarısız: {last_err}")
    logger.error(f"YouTube yükleme hatası (tüm denemeler): {last_err}")
    return {"success": False, "error": last_err}
