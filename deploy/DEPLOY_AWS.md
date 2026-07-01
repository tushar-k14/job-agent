# Deploying to AWS Free Tier (EC2)

This guide gets the Job Application Agent running on a free-tier **EC2 `t3.micro`**
(1 GB RAM, 750 hrs/month free for 12 months) with a public URL you can demo.

> **RAM reality check:** 1 GB is tight for `chromadb`/`onnxruntime`. Two safe paths:
> 1. **Keep memory on** — use `requirements.txt`; the bootstrap adds 2 GB swap so the
>    build survives. Best demo (shows the vector-memory feature).
> 2. **Lighter build** — use `requirements-cloud.txt` (no Chroma); memory degrades to a
>    no-op, everything else works. Fastest/most reliable on 1 GB.
> Pick one; both are covered below.

---

## 1. Launch the instance (AWS Console)

1. **EC2 → Launch instance.**
2. **AMI:** Amazon Linux 2023 (or Ubuntu 22.04) — both free-tier eligible.
3. **Instance type:** `t3.micro` (or `t2.micro`) — "Free tier eligible".
4. **Key pair:** create/download one (you'll SSH with it).
5. **Network / Security group — add inbound rules:**
   | Type | Port | Source | Why |
   |------|------|--------|-----|
   | SSH | 22 | *My IP* | your admin access only |
   | Custom TCP | 8501 | *My IP* (recommended) or 0.0.0.0/0 | the Streamlit app |
6. **Storage:** default 8 GB is enough (30 GB is free-tier max).
7. Launch.

> **Security note:** the app has **no login**. If you open 8501 to `0.0.0.0/0`, anyone
> with the IP can use it and burn your DeepSeek/Gemini credits. For an interview, prefer
> *My IP*, or only run it around the call. HTTPS + auth options are in §5.

## 2. Bootstrap the app (SSH)

```bash
ssh -i your-key.pem ec2-user@<public-ip>       # Ubuntu: ubuntu@<public-ip>

# Fetch and run the bootstrap (installs Docker + swap, clones, prepares .env)
curl -fsSL https://raw.githubusercontent.com/tushar-k14/job-agent/main/deploy/bootstrap.sh -o bootstrap.sh
REPO=https://github.com/tushar-k14/job-agent.git BRANCH=main bash bootstrap.sh
```

The first run stops after creating `.env`. Add your keys and start it:

```bash
cd ~/job-agent
nano .env          # set DEEPSEEK_API_KEY and GEMINI_API_KEY
sudo docker compose up -d --build
```

**Lighter build (1 GB struggling?):** point the image at the no-Chroma requirements:
```bash
# one-off: use the cloud requirements for the build
sed -i 's#requirements.txt#requirements-cloud.txt#' Dockerfile
sudo docker compose up -d --build
```

## 3. Open it

```
http://<public-ip>:8501
```

Persisted data (SQLite + Chroma) lives in `~/job-agent/data` via the compose volume, so
it survives `docker compose restart` and reboots.

## 4. Demo tips

- Pre-seed a few runs before the call so the **Observability** page has data.
- Use **paste-JD mode** (100% reliable); don't demo live LinkedIn scraping (blocked by design).
- Keep the GitHub **green CI run** open in a tab.
- `sudo docker compose logs -f` to watch requests live if asked.

## 5. Optional hardening (nice-to-have, not required for a demo)

- **HTTPS + port 80:** put [Caddy](https://caddyserver.com) in front — it auto-provisions
  a TLS cert if you point a free domain (e.g. DuckDNS) at the instance.
- **Basic auth:** front the app with nginx + `htpasswd`, or use Streamlit's built-in
  authentication, so the public URL isn't wide open.
- **Cost hygiene:** `t3.micro` is free for 12 months / 750 hrs — one always-on instance
  fits. `sudo docker compose down` + stop the instance when not demoing to be safe.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker compose` build killed / hangs | Swap missing — re-run bootstrap, or use `requirements-cloud.txt` |
| Can't reach `:8501` | Security-group inbound rule for 8501 not added, or wrong source IP |
| `permission denied` on docker | Use `sudo docker ...` (group change needs re-login) |
| App loads but LLM calls fail | Keys not set in `.env`, or you didn't rebuild after editing it |
