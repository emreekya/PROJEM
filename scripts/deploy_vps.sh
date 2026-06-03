#!/usr/bin/env bash
set -euo pipefail

# MoneyPrinterTurbo Ubuntu VPS hızlı kurulum scripti.
# Root veya sudo yetkili kullanıcı ile çalıştır.
#
# Kullanım:
#   chmod +x scripts/deploy_vps.sh
#   ./scripts/deploy_vps.sh

PROJECT_DIR="${PROJECT_DIR:-/opt/moneyprinterturbo}"

echo "==> Sistem güncelleniyor..."
sudo apt update
sudo apt -y upgrade

echo "==> Temel paketler kuruluyor..."
sudo apt -y install ca-certificates curl gnupg git nginx ufw certbot python3-certbot-nginx

echo "==> Docker kuruluyor..."
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
fi

echo "==> Docker Compose kontrol ediliyor..."
docker compose version

echo "==> Firewall ayarlanıyor..."
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

echo "==> Proje klasörü hazırlanıyor: ${PROJECT_DIR}"
sudo mkdir -p "${PROJECT_DIR}"
sudo chown -R "$USER":"$USER" "${PROJECT_DIR}"

echo ""
echo "Kurulum temel paketleri tamamladı."
echo "Sonraki adımlar:"
echo "1) Repo dosyalarını ${PROJECT_DIR} içine gönder veya git clone yap."
echo "2) cp .env.production.example .env.production"
echo "3) cp config.example.toml config.toml"
echo "4) config.toml içine API keylerini sadece VPS üzerinde yaz."
echo "5) docker compose -f docker-compose.prod.yml --env-file .env.production up -d --build"
echo "6) Nginx configlerini /etc/nginx/sites-available içine kopyala."
echo ""
