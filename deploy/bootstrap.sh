#!/usr/bin/env bash
# One-shot EC2 bootstrap for the Job Application Agent.
#
# Tested on Amazon Linux 2023 and Ubuntu 22.04 (t3.micro / t2.micro, free tier).
# Run it after SSHing into a fresh instance:
#
#   curl -fsSL <raw-url>/deploy/bootstrap.sh -o bootstrap.sh
#   REPO=https://github.com/tushar-k14/job-agent.git BRANCH=main bash bootstrap.sh
#
# Or clone the repo first and run: bash deploy/bootstrap.sh
#
# It: adds swap (critical on 1 GB RAM), installs Docker, clones the repo, and starts
# the app with docker compose. You still need to drop your API keys into .env.
set -euo pipefail

REPO="${REPO:-https://github.com/tushar-k14/job-agent.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-$HOME/job-agent}"

echo "==> 1/5 Adding 2 GB swap (safe on 1 GB instances during pip/docker build)"
if ! sudo swapon --show | grep -q '/swapfile'; then
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "    swap already present, skipping"
fi

echo "==> 2/5 Installing Docker + compose plugin"
if ! command -v docker >/dev/null 2>&1; then
  if command -v dnf >/dev/null 2>&1; then           # Amazon Linux 2023
    sudo dnf -y install docker
    sudo systemctl enable --now docker
    # compose plugin
    sudo mkdir -p /usr/libexec/docker/cli-plugins
    sudo curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
      -o /usr/libexec/docker/cli-plugins/docker-compose
    sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose
  else                                              # Ubuntu/Debian
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-compose-plugin git
    sudo systemctl enable --now docker
  fi
  sudo usermod -aG docker "$USER" || true
fi

echo "==> 3/5 Cloning repo ($BRANCH)"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only || true
fi
cd "$APP_DIR"

echo "==> 4/5 Preparing .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "    !! Edit $APP_DIR/.env and set DEEPSEEK_API_KEY / GEMINI_API_KEY, then re-run:"
  echo "       cd $APP_DIR && sudo docker compose up -d --build"
  echo ""
  exit 0
fi

echo "==> 5/5 Building + starting (docker compose)"
# `sudo` because the group change above needs a re-login to take effect this session.
sudo docker compose up -d --build

IP=$(curl -fsSL http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<public-ip>")
echo ""
echo "==> Done. App should be reachable at:  http://${IP}:8501"
echo "    (Ensure the instance security group allows inbound TCP 8501.)"
