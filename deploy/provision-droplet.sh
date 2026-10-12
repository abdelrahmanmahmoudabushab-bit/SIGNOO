#!/bin/bash
# One-time setup of a fresh Ubuntu 22.04/24.04 droplet. Run as root over SSH:
#
#   bash provision-droplet.sh
#
# It installs Docker, opens only SSH/HTTP/HTTPS, adds swap (training needs headroom on a small
# droplet), and turns on automatic security updates. It does not start SIGNOO; see
# deploy/README.md for that.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git ufw fail2ban unattended-upgrades docker.io docker-compose-v2

systemctl enable --now docker
systemctl enable --now fail2ban

# Firewall: SSH for you, 80/443 for Caddy. Postgres and the API publish no ports at all
# (deploy/docker-compose.yml), and Docker bypasses ufw for published ports, so the compose
# file is the real guard; this is the second line.
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp
ufw --force enable

# Swap, only if there is none. A 2 GB droplet running a training job next to Postgres runs out
# of memory without it, and the kernel's answer to that is to kill Postgres.
if ! swapon --show | grep -q .; then
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

dpkg-reconfigure -f noninteractive unattended-upgrades

echo
echo "done. Next:"
echo "  1. point your domain's A record at this droplet's IP"
echo "  2. git clone the repo, cp deploy/.env.example deploy/.env, fill it in"
echo "  3. docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d --build"
