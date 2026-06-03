<div align="center">
<h1 align="center">MoneyPrinterTurbo 💸</h1>

<h3><a href="README.md">简体中文</a> | <a href="README-en.md">English</a> | Türkçe</h3>

Yalnızca bir video <b>konusu</b> ya da <b>anahtar kelime</b> vermeniz yeterli; uygulama video metnini, video görsellerini, altyazıları ve arka plan müziğini tamamen otomatik üretip bunları birleştirerek HD bir kısa video oluşturur.

</div>

## Özellikler 🎯

- Eksiksiz **MVC mimarisi**, temiz kod yapısı, `API` ve `Web arayüzü` desteği
- Video metninin **yapay zeka ile otomatik üretimi**; metni kendiniz de yazabilirsiniz
- Çoklu **HD video** boyutları
  - Dikey 9:16, `1080x1920`
  - Yatay 16:9, `1920x1080`
- **Toplu video üretimi**: tek seferde birden çok video üretip en beğendiğinizi seçin
- **Klip süresi** ayarı ile görsel geçiş sıklığını kontrol etme
- **Türkçe**, Çince ve İngilizce video metni desteği
- Çok sayıda **seslendirme (TTS)** seçeneği ve **anlık ön dinleme**
- **Altyazı üretimi**: `font`, `konum`, `renk`, `boyut` ve `çerçeve (stroke)` ayarlanabilir
- **Arka plan müziği**: rastgele veya belirli bir dosya, ses seviyesi ayarlanabilir
- **Telifsiz HD** video kaynakları; kendi **yerel görsellerinizi** de kullanabilirsiniz
- Çoklu model desteği: **OpenAI**, **Moonshot**, **Azure**, **gpt4free**, **one-api**, **Tongyi Qianwen**, **Google Gemini**, **Ollama**, **DeepSeek**, **MiniMax**, **Pollinations**, **ModelScope** ve daha fazlası

## Hızlı Başlangıç 🚀

### Gereksinimler
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/) kurulu olmalı
- Bir LLM sağlayıcısına ait API anahtarı (ör. DeepSeek, Moonshot, OpenAI)
- Video kaynağı için [Pexels](https://www.pexels.com/api/) ve/veya [Pixabay](https://pixabay.com/api/docs/) API anahtarı

### Kurulum (Windows)

```powershell
# 1) Bağımlılıkları kurun
pip install -r requirements.txt

# 2) Yapılandırma dosyasını oluşturun (ilk çalıştırmada otomatik kopyalanır)
#    config.example.toml -> config.toml
#    config.toml içinde API anahtarlarınızı doldurun

# 3) Web arayüzünü başlatın
.\webui.bat

# veya API servisini başlatın
python main.py
```

Web arayüzü açıldığında uygulama **varsayılan olarak Türkçe** gelir. Dili sağ üstteki **Language / 语言** menüsünden değiştirebilirsiniz.

### Docker ile

```bash
docker-compose up
```

## Yapılandırma ⚙️

Tüm ayarlar `config.toml` dosyasında bulunur. Önemli bölümler:

- `[app]` — LLM sağlayıcısı, model adı ve API anahtarları
- `[azure]` / TTS ayarları — seslendirme sağlayıcıları
- `[ui]` — arayüz dili (`language = "tr"`), günlük gizleme, altyazı konumu
- `[ui]` Upload-Post — üretilen videoları TikTok/Instagram'a otomatik paylaşma

## Proje Yapısı 🗂️

```
app/            FastAPI tabanlı çekirdek (MVC)
  controllers/  API uç noktaları (v1: video, llm)
  services/     iş mantığı (llm, material, voice, subtitle, video, task)
  models/       şema ve sabitler
  utils/        yardımcı araçlar
webui/          Streamlit web arayüzü
  i18n/         dil dosyaları (tr.json dahil)
resource/       fontlar, müzik ve örnek görseller
main.py         API servisi giriş noktası
```

## Lisans 📝

Bu proje [MIT lisansı](LICENSE) ile dağıtılmaktadır. Orijinal proje: [harry0703/MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo).
