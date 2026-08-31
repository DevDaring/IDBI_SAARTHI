# SAARTHI — Server Setup Notes

How this server hosts SAARTHI, what was installed, and where every credential
comes from. **No secret values are written in this file** — only their sources.

---

## 1. Live URL

**https://koushikdeb.duckdns.org** — trusted HTTPS (real certificate, no browser
warning), no port number, publicly reachable, no login.

Server public IP: `154.12.116.152` (static — does not change on reboot).

---

## 2. How the URL + HTTPS were created

1. **Free subdomain (DuckDNS).** `koushikdeb.duckdns.org` was created on
   duckdns.org and pointed at the server IP `154.12.116.152`. This is what makes
   the address permanent and free (no domain purchase).
2. **Automatic trusted certificate (Caddy + ACME).** The Caddy web server
   requests a certificate over the **HTTP-01** challenge on port 80 and serves it
   on port 443. The certificate was issued by **ZeroSSL** (Caddy's ACME provider;
   Let's Encrypt is the fallback) and **auto-renews** every ~60 days. HTTP-01 was
   used because DuckDNS's nameservers are unreliable for the DNS-01 method (see
   note below).
3. **Reverse proxy.** Caddy (port 443) forwards requests to the SAARTHI app
   server, **gunicorn**, which listens only on `127.0.0.1:8080` (localhost). So
   the app itself is never exposed directly — only Caddy faces the internet.

```
Browser ──HTTPS:443──▶ Caddy (TLS, ZeroSSL cert) ──HTTP──▶ gunicorn 127.0.0.1:8080 ──▶ SAARTHI (Flask + React)
```

> **Earlier approaches (superseded):** a self-signed cert on `:8443`, then a
> free **Cloudflare Quick Tunnel** (trusted but with an ephemeral URL), then a
> **certbot DNS-01** attempt (failed — DuckDNS nameservers repeatedly timed out
> for the certificate authority). The final Caddy + HTTP-01 setup is the one in
> use; the tunnel is stopped.

---

## 3. What was installed

| Software | Version | Where | Why | Needed root? |
| --- | --- | --- | --- | --- |
| **Caddy** | 2.11.4 | `/usr/local/bin/caddy` | Reverse proxy + automatic HTTPS on 80/443 | yes (install + bind low ports) |
| **gunicorn** | 26.0 | Python venv | Production app server for Flask + React build | no |
| **Python venv + libs** | 3.11 | `/home/Debz/Hackathon/IDBI_Hackathon/.venv` | Flask, LightGBM, SHAP, lifelines, fairlearn, certbot, … | no |
| **Node.js** | 20.18 | `~/.local/opt/node-v20.18.0-linux-x64` | Building the React frontend (one-time) | no |
| **certbot** | 5.6 | Python venv | DNS-01 cert attempt (superseded by Caddy) | no |
| **cloudflared** | 2026.6 | `~/.local/bin/cloudflared` | Earlier tunnel (now stopped/unused) | no |

Two **systemd services** were created (see §4) — both auto-start on boot.

---

## 4. Auto-restart (systemd)

Both run as user `Debz`, are **enabled** (start automatically when the server
boots) and set to **restart on crash**:

| Service | Unit file | Runs |
| --- | --- | --- |
| `saarthi` | `/etc/systemd/system/saarthi.service` | gunicorn (the app) on `127.0.0.1:8080` |
| `caddy` | `/etc/systemd/system/caddy.service` | Caddy on `:80` + `:443` |

Operate:
```bash
sudo systemctl status  saarthi caddy      # health
sudo systemctl restart saarthi            # restart the app
sudo systemctl restart caddy              # restart the proxy
sudo journalctl -u caddy   -f             # proxy / cert logs
sudo journalctl -u saarthi -f             # app logs
```
Config: app env `/home/Debz/Host/saarthi/saarthi.env`, proxy
`/home/Debz/Host/saarthi/Caddyfile`, gunicorn `.../gunicorn_conf.py`.
Certificate renewal is fully automatic (Caddy).

---

## 5. Credentials — sources only (NO values here)

None of the following secret values are stored in this document. This lists
**what** each credential is and **where it is read from**.

| Credential | Read from / stored at | Used for | Exposed to browser? |
| --- | --- | --- | --- |
| **LLM API keys** — DeepSeek, Mistral, OpenRouter, Gemini, NanoGPT | your `Codes/.env`; loaded by `backend/config.py`; a copy lives at `Host/saarthi/backend/.env` (permissions `600`) | server-side LLM calls (mapping, explanations, judge) | **No** — server-side only |
| **DuckDNS token** | your DuckDNS account page; saved to `Host/saarthi/.duckdns_token` (permissions `600`) | the earlier DNS-01 cert attempt (Caddy does **not** need it) | No |
| **ACME registration email** | your email address | registering with ZeroSSL / Let's Encrypt for the certificate (not a secret) | No |
| **sudo password** | you typed it into the chat; used **only** in-session to install Caddy, create the systemd services, and open the firewall — **not written to any file on disk** | system install steps | No |
| **App login (HTTP Basic Auth)** | `Host/saarthi/.hostenv` (permissions `600`) — currently **blank/disabled** at your request (public demo) | optional gate on the whole app | No |
| **Cloudflare API key** | present in your `Codes/.env` as `Cloudflare_API_Key` — **not used** (we chose DuckDNS + Caddy instead) | n/a | No |
| **Kaggle username / key / token** | your `Codes/.env`; read case-insensitively by `scripts/download_data.py` | one-time dataset downloads | No |

### Security reminders
- **Rotate your sudo password** (`passwd`) — it appeared in the chat transcript.
- Consider **regenerating the DuckDNS token** from its dashboard for the same reason.
- Secret files (`.env`, `.hostenv`, `.duckdns_token`, `saarthi.env`) are `chmod 600`
  and must never be committed to git or served by the web server.
- The app is currently **open (no login)** so anyone with the URL can run analyses,
  which uses your LLM API credits. Re-enable Basic Auth via `.hostenv` +
  `saarthi.env` if you want to restrict it.
