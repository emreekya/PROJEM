# MoneyPrinterTurbo VPS Production Kurulum Rehberi 🇹🇷

Bu rehber, projeyi cPanel/Turhost paylaşımlı hosting yerine VPS/VDS üzerinde güvenli şekilde çalıştırmak içindir.

## 1. Önerilen VPS

Minimum:
- 2 vCPU
- 4 GB RAM
- 60 GB disk
- Ubuntu 22.04 veya 24.04

Önerilen:
- 4 vCPU
- 8-16 GB RAM
- 100+ GB disk

Video üretimi CPU/RAM/disk kullanır. Çok video üretilecekse 8 GB RAM altı yorabilir.

---

## 2. Domain planı

Önerilen yapı:

```text
video.senindomain.com      -> WebUI
api-video.senindomain.com  -> API, istersen kapalı tut
```

İlk aşamada sadece WebUI aç. API'yi dışarı açmak zorunda değilsin.

---

## 3. Sunucuya bağlan

Windows PowerShell:

```powershell
ssh root@SUNUCU_IP
```

---

## 4. Sunucuyu hazırla

Repo içindeki scripti kullanabilirsin:

```bash
chmod +x scripts/deploy_vps.sh
./scripts/deploy_vps.sh
```

Manuel yapmak istersen:

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install ca-certificates curl gnupg git nginx ufw certbot python3-certbot-nginx
curl -fsSL https://get.docker.com | sudo sh
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable
```

---

## 5. Projeyi VPS'e çek

```bash
cd /opt
sudo mkdir -p moneyprinterturbo
sudo chown -R $USER:$USER moneyprinterturbo
cd moneyprinterturbo
git clone https://github.com/emreekya/PROJEM.git .
```

Private repo ise GitHub token veya SSH key gerekebilir.

---

## 6. Ortam dosyalarını hazırla

```bash
cp .env.production.example .env.production
cp config.example.toml config.toml
mkdir -p storage
```

> `config.toml` ve `.env.production` gerçek key içereceği için GitHub'a gönderilmez.

`.gitignore` içinde `config.toml` zaten ignore edilmiş olmalı.

---

## 7. config.toml içine API keyleri gir

Örnek:

```toml
[app]
llm_provider = "gemini"
gemini_api_key = "GERCEK_KEY_BURAYA"
gemini_model_name = "gemini-2.5-flash"
pexels_api_keys = ["GERCEK_PEXELS_KEY"]
```

Cloudflare AI görsel kullanacaksan:

```toml
ai_image_provider = "cloudflare"
cloudflare_account_id = "GERCEK_ACCOUNT_ID"
cloudflare_api_key = "GERCEK_CLOUDFLARE_TOKEN"
cloudflare_image_model = "@cf/black-forest-labs/flux-1-schnell"
```

---

## 8. Docker ile başlat

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

Kontrol:

```bash
docker ps
docker logs -f moneyprinterturbo-webui
docker logs -f moneyprinterturbo-api
```

Lokal port testi:

```bash
curl http://127.0.0.1:8501/_stcore/health
curl http://127.0.0.1:8080/docs
```

---

## 9. Nginx ayarla

WebUI config örneğini kopyala:

```bash
sudo cp nginx/moneyprinterturbo-webui.conf.example /etc/nginx/sites-available/moneyprinterturbo-webui
sudo nano /etc/nginx/sites-available/moneyprinterturbo-webui
```

`video.senindomain.com` yazan yerleri kendi domaininle değiştir.

Aktifleştir:

```bash
sudo ln -s /etc/nginx/sites-available/moneyprinterturbo-webui /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 10. SSL kur

```bash
sudo certbot --nginx -d video.senindomain.com
```

---

## 11. Güncelleme

Yeni kod çekmek için:

```bash
cd /opt/moneyprinterturbo
git pull
docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build
```

---

## 12. Backup

En önemli klasörler:

```text
config.toml
.env.production
storage/
```

Basit backup:

```bash
tar -czf backup-moneyprinterturbo-$(date +%F).tar.gz config.toml .env.production storage
```

---

## 13. Güvenlik notları

- API keyleri GitHub'a koyma.
- `config.toml` sunucuda kalsın.
- API subdomainini herkese açma; gerekmedikçe kapalı tut.
- Sadece 80/443 ve SSH açık olsun.
- Streamlit/API portları dış dünyaya direkt açılmasın.
- Docker compose dosyasında portlar `127.0.0.1` ile bağlıdır, bu doğru.
