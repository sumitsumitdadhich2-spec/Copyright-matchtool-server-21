# Qwen2.5-VL AWS GPU server setup

Runs [vLLM](https://github.com/vllm-project/vllm)'s OpenAI-compatible API
server for `Qwen/Qwen2.5-VL-7B-Instruct`, used by the main app's
`server/vlm-verify.ts` to gate matched segments through a scene-verification
check. This runs on a separate AWS GPU instance — it is not part of the
Replit deployment.

## 1. Provision the EC2 instance

- Ubuntu 22.04 LTS, NVIDIA GPU (e.g. `g6.xlarge` or `g5.xlarge`).
- Open inbound port `8000` (ideally restricted to your Replit app's
  outbound IP / a VPN, not `0.0.0.0/0` — this endpoint has no auth built in).
- Install the NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  and Docker + Docker Compose plugin.

## 2. Deploy the stack

```bash
sudo mkdir -p /opt/vlm-server
sudo cp docker-compose.yml vllm-qwen.service /opt/vlm-server/
cd /opt/vlm-server

# Optional — only needed if your Hugging Face account requires a token
# for this model:
echo "HUGGING_FACE_HUB_TOKEN=hf_xxx" > .env

docker compose up -d
docker compose logs -f   # first run downloads ~16 GB — this takes a while
```

The model is cached to the `huggingface-cache` Docker volume, so subsequent
`docker compose restart` / container crashes / `docker compose up -d` do
**not** re-download it.

## 3. Survive reboots

`restart: unless-stopped` in `docker-compose.yml` brings the container back
after a crash as long as the Docker daemon is running. To also make sure the
whole stack comes back up after an **EC2 instance reboot**, install the
systemd unit:

```bash
sudo cp vllm-qwen.service /etc/systemd/system/vllm-qwen.service
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen.service
```

## 4. Verify it's serving

```bash
curl http://localhost:8000/v1/models
```

Should return a JSON list including `Qwen/Qwen2.5-VL-7B-Instruct`.

## 5. Point the Replit app at it

In the Replit app, set the environment variable:

```
VLM_ENDPOINT_URL=http://<ec2-public-ip-or-dns>:8000/v1/chat/completions
```

Optional tuning (defaults shown):

```
VLM_CONFIDENCE_THRESHOLD=80
VLM_MAX_ATTEMPTS=10
```

If this variable is unset or the endpoint is unreachable at request time,
the app logs a warning once and returns the original hash-matched segments
untouched — matching keeps working with the GPU server off.

## Cost note

A `g6.xlarge` (NVIDIA L4) or `g5.xlarge` (NVIDIA A10G) runs continuously
once started (`restart: unless-stopped` will keep restarting it — it will
NOT auto-shut-down to save cost). Stop the instance from the AWS console
when you don't need VLM verification, or add your own scheduled
start/stop automation if usage is intermittent.
