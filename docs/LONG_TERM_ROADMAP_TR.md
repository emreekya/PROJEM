# Uzun Vadeli MoneyPrinterTurbo Ürün Yol Haritası 🚀

Amaç: Mevcut MoneyPrinterTurbo tabanını, senin markana özel profesyonel içerik üretim paneline dönüştürmek.

## Faz 1 — Production VPS

- Docker Compose production dosyası
- Nginx reverse proxy
- SSL
- `.env.production`
- `config.toml` güvenli yapı
- Storage backup mantığı

## Faz 2 — Marka ve Türkçe deneyim

- README Türkçeleştirme
- Uygulama adını markana göre değiştirme
- Türkçe varsayılan promptlar
- “Bunu biliyor muydun abi?” tarzı kanal sesi
- Burç, tarih, ilginç bilgi, hayat hack, psikoloji kategorileri
- Otomatik kapak başlığı üretimi

## Faz 3 — İçerik üretim paneli

- Video fikir havuzu
- Senaryo onay ekranı
- Görsel stil seçimi
- Ses seçimi
- Altyazı stili seçimi
- Tek tıkla render
- Üretilen videolar arşivi
- Başlık/açıklama/hashtag üretici

## Faz 4 — YouTube/TikTok/Reels paketleme

- YouTube Shorts metadata üretimi
- Video açıklaması
- Hashtag setleri
- Kapak fotoğrafı üretimi
- Planlı içerik takvimi
- Manuel upload hazırlık paketi

## Faz 5 — Profesyonel SaaS mimari

Önerilen mimari:

```text
Frontend: React / Next.js
Backend: FastAPI
Queue: Redis
Worker: Celery veya RQ
Storage: Local disk veya S3 uyumlu depolama
Database: PostgreSQL
Reverse Proxy: Nginx
Deployment: Docker Compose
```

## Faz 6 — Otomasyon

- Günlük 10 konu önerisi
- Onaylanan konulardan otomatik senaryo
- Otomatik görsel/video materyal seçimi
- Otomatik render kuyruğu
- Başarısız render retry sistemi
- Log paneli
- Kota/API key takip paneli

## Faz 7 — Çok kullanıcı / ekip

- Admin panel
- Kullanıcı rolleri
- Kullanıcı başına render limiti
- Proje/kanal bazlı çalışma alanları
- İçerik onay süreci
