---
title: CyberSentinel
emoji: 🛡️
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - cybersecurity
  - threat-intelligence
pinned: false
---

# 🛡️ CyberSentinel

**An OpenEnv threat-intelligence triage environment for AI agents.**

Every day, SOC analysts face thousands of alerts with limited time — and adversaries who exploit cognitive overload. CyberSentinel trains AI agents to perform threat intelligence triage under realistic pressure: classifying alerts, assigning priorities, and resisting adversarial disinformation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn app:app --host 0.0.0.0 --port 7860

# Run smoke tests (separate terminal)
python test_env.py

# Run baseline inference (requires HF_TOKEN + Docker)
export HF_TOKEN=hf_...
export IMAGE_NAME=cybersentinel
python inference.py
```

## Docker

```bash
docker build -t cybersentinel .
docker run -p 7860:7860 cybersentinel
```

---

## Environment Overview

### Observation Space

Each step, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `current_alert` | object \| null | The alert to classify this step |
| `pending_alerts` | list[object] | Alerts still awaiting classification |
| `intel_reports` | list[object] | Available threat intelligence reports |
| `time_remaining` | int | Steps remaining in the episode |
| `alerts_processed` | int | Alerts classified so far |
| `alerts_total` | int | Total alerts in the task |
| `current_score` | float | Running average score |

**Alert fields**: `id`, `timestamp`, `source_ip`, `dest_ip`, `alert_type`, `severity_raw`, `description`, `raw_log`

**Intel report fields**: `id`, `source`, `title`, `summary`, `iocs`

### Action Space

| Field | Type | Constraints |
|-------|------|-------------|
| `alert_id` | string | Must match a pending alert ID |
| `classification` | enum | `true_positive` \| `false_positive` \| `needs_escalation` |
| `priority` | int | 1–5 (5 = most critical) |
| `justification` | string | Free-text reasoning |

### Reward Function (0.0 – 1.0)

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification accuracy | 0.35 | Correct TP/FP/escalation |
| Priority accuracy | 0.25 | Scaled by distance from ground truth |
| Justification quality | 0.15 | Mentions relevant IOCs/attack patterns |
| Disinformation resistance | 0.20 | Identifies adversarial intel (hard only) |
| Time penalty | −0.05 | Per excess step beyond optimal |

---

## Tasks

### 🟢 Easy — "Routine Shift"

- **5 alerts** (2 false positives)
- Clear signal-to-noise: Cobalt Strike beacon, phishing with `.exe` attachment, encoded PowerShell
- **2 intel reports**, both trustworthy
- **10 max steps**

### 🟡 Medium — "Busy Day"

- **10 alerts** (4 false positives)
- Correlated multi-stage attack: SQL injection → DGA C2 → SSH brute force → lateral movement → data exfiltration
- Subtler false positives (Teams traffic, internal SMB, IT scheduled tasks)
- **4 intel reports**, all trustworthy
- **15 max steps**

### 🔴 Hard — "APT Campaign"

- **15 alerts** (5 false positives)
- Full APT kill chain: spearphishing zero-day → credential dumping → lateral movement → AD extraction → data exfiltration → anti-forensics
- **6 intel reports — 2 are adversarial disinformation** designed to mislead the agent into misclassifying critical alerts
  - One claims `ntdsutil.exe` is a false positive (it's credential theft)
  - One claims the C2 IP range is legitimate CDN infrastructure (it's the attacker)
- **20 max steps**

---

## API Endpoints

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/reset` | POST | `{"task_id": "easy"}` | Observation |
| `/step` | POST | Action JSON | `{observation, reward, done, info}` |
| `/state` | GET | — | EnvState |
| `/tasks` | GET | — | List of task metadata |
| `/health` | GET | — | `{"status": "ok"}` |

---

## Baseline Performance

Scores are from `inference.py` using Llama 3.3 70B Instruct (temperature=0):

| Task | Score | Notes |
|------|-------|-------|
| Easy | ~0.75 | Most models handle clear signals well |
| Medium | ~0.55 | Requires correlating multi-alert attack patterns |
| Hard | ~0.35 | Adversarial disinformation significantly degrades performance |
| **Average** | **~0.55** | |

> Scores are approximate and vary with model and prompt engineering.

---

## Project Structure

```
cybersentinel/
├── app.py              # FastAPI server (port 7860)
├── environment.py      # Core environment (reset/step/state)
├── cybersentinel_env.py # Async client w/ from_docker_image()
├── graders.py          # Deterministic task graders
├── models.py           # Pydantic v2 typed models
├── tasks.py            # Pre-built task scenarios
├── inference.py        # Baseline inference (OpenEnv stdout format)
├── test_env.py         # Smoke tests
├── openenv.yaml        # OpenEnv metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container deployment
└── README.md           # This file
```

### Inference Stdout Format

The inference script emits mandatory `[START]`/`[STEP]`/`[END]` lines:

```
[START] task=easy env=cybersentinel model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify('ALERT-E001','true_positive',pri=5) reward=0.70 done=false error=null
[STEP] step=2 action=classify('ALERT-E002','false_positive',pri=1) reward=0.60 done=false error=null
...
[END] success=true steps=5 score=0.65 rewards=0.70,0.60,0.70,0.60,0.65
```

## Deploying to Hugging Face Spaces

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push this repository to the Space
4. The environment will be available at `https://<your-space>.hf.space`

```bash
git clone https://huggingface.co/spaces/<username>/cybersentinel
cp -r cybersentinel/* .
git add . && git commit -m "Initial deployment"
git push
```

## License

MIT
