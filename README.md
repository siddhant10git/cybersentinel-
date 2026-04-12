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
  - soc-analyst
  - adversarial-robustness
pinned: false
---

# 🛡️ CyberSentinel v2

**An OpenEnv threat-intelligence triage environment for AI agents — built for the OpenEnv Hackathon.**

Every day, SOC analysts face thousands of alerts with limited time — and adversaries who exploit cognitive overload. CyberSentinel trains AI agents to perform threat intelligence triage under realistic pressure: classifying alerts, assigning priorities, correlating multi-stage attack chains, and resisting adversarial disinformation.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run smoke tests (separate terminal)
python test_env.py

# Run baseline inference (requires HF_TOKEN)
export HF_TOKEN=hf_...
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
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
| `running_reward_avg` | float | Running average of per-step rewards |
| `steps_warning` | bool | **True when ≤ 3 steps remain** — prioritise! |

**Alert fields**: `id`, `timestamp`, `source_ip`, `dest_ip`, `alert_type`, `severity_raw`, `description`, `raw_log`

**Intel report fields**: `id`, `source`, `title`, `summary`, `iocs`

> ⚠️ Ground-truth labels and `is_disinformation` flags are **never** exposed in observations.

### Action Space

| Field | Type | Constraints |
|-------|------|-------------|
| `alert_id` | string | Must match a pending alert ID |
| `classification` | enum | `true_positive` \| `false_positive` \| `needs_escalation` |
| `priority` | int | 1–5 (5 = most critical) |
| `justification` | string | Free-text reasoning (max 2000 chars) |

### Reward Function (0.0 – 1.0)

| Component | Weight | Description |
|-----------|--------|-------------|
| Classification accuracy | **0.35** | Correct TP/FP/escalation classification |
| Priority accuracy | **0.25** | Scaled linearly by distance from ground truth |
| Justification quality | **0.15** | Mentions relevant IOCs and attack patterns |
| Disinformation resistance | **0.20** | Identifies/avoids adversarial intel (any task with disinfo) |
| Cross-alert correlation | **+0.10** | References a sibling alert in the same attack chain |
| Time penalty | **−0.10/step** | Per excess step beyond optimal, capped −0.30 |

**Hard task kill-chain bonus**: +0.10 added to final score if ≥ 6 of 9 APT campaign alerts are correctly classified.

---

## Tasks

### 🟢 Easy — "Routine Shift"

- **7 alerts** (3 false positives, 4 true positives)
- Attack chain: Cobalt Strike beacon → phishing email → encoded PowerShell C2 → DNS beacon to known C2 domain
- 3 intel reports — all trustworthy. One cross-links the phishing/PowerShell chain for correlation bonus.
- **12 max steps**

### 🟡 Medium — "Busy Day"

- **12 alerts** (5 FP, 5 TP, 2 NEEDS_ESCALATION) — presented in **shuffled order**
- Correlated multi-stage attack: SQL injection → DGA C2 → SSH brute force → data exfiltration → lateral movement
- Subtle FPs: Teams traffic, internal SMB, IT scheduled tasks, IT WMI remote management
- 4 intel reports — one is a draft with slightly incorrect thresholds (tests skepticism)
- **18 max steps**

### 🔴 Hard — "APT Campaign"

- **17 alerts** (6 FP, 9 TP, 2 NE) — full APT kill chain
- Attack chain: spearphishing zero-day → PDF exploit → credential dumping → WMI lateral movement → AD extraction → DNS tunneling → 2.3 GB exfiltration → anti-forensics
- **7 intel reports — 3 are adversarial disinformation** planted by the attacker:
  - Claims `ntdsutil.exe` is a benign AD maintenance tool (it's credential theft)
  - Claims the C2 IP range is legitimate CDN infrastructure (it's the attacker's C2)
  - Uses a fake MITRE ATT&CK reference to suppress persistence detection
- **Kill-chain bonus** (+0.10) for correctly classifying ≥ 6 of 9 campaign alerts
- **25 max steps**

### 🟠 Insider Threat — "The Trusted Adversary"

- **8 alerts** (3 FP, 3 TP, 2 NE)
- A finance director is staging data exfiltration via legitimate cloud storage (OneDrive, Dropbox, personal Gmail)
- **No external C2, no malware, all traffic uses valid certificates** — raw log evidence requires behavioural reasoning
- DLP violation, cross-role access, and volume anomalies are the key signals
- **14 max steps**

### ⚪ Zero Alert Shift — "False Positive Crucible"

- **5 alerts — ALL false positives**
- Tests over-trigger bias: Microsoft 365 syncs, Windows Update, AD replication, scheduled scans, HR automation
- Classifying everything as TP scores 0. Classifying everything as FP scores near-perfect.
- **8 max steps**

---

## API Endpoints

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/reset` | POST | `{"task_id": "easy"}` | Observation |
| `/step` | POST | Action JSON | `{observation, reward, done, info}` |
| `/state` | GET | — | EnvState |
| `/tasks` | GET | — | List of task metadata |
| `/grade` | POST | `{"task_id": ..., "actions": [...]}` | GradeResult |
| `/health` | GET | — | `{"status": "ok", "tasks_available": [...]}` |
| `/docs` | GET | — | Interactive Swagger UI |

---

## Baseline Performance

Scores from `inference.py` using Qwen2.5-72B-Instruct (temperature=0):

| Task | Difficulty | Score | Notes |
|------|-----------|-------|-------|
| Easy | easy | ~0.80 | Clear signals, all intel trustworthy |
| Zero | easy | ~0.90 | All FPs — tests over-trigger bias |
| Medium | medium | ~0.60 | Correlated patterns, shuffled order |
| Insider | medium+ | ~0.45 | Behavioural reasoning, no C2 signatures |
| Hard | hard | ~0.40 | 3 disinformation reports, kill-chain bonus |
| **Average** | — | **~0.63** | |

> Scores are approximate and vary with model and prompt engineering. CoT mode (`COT_MODE=1`) improves hard task performance significantly.

---

## Advanced Inference Options

```bash
# Chain-of-thought reasoning mode (better on hard/insider tasks)
export COT_MODE=1
python inference.py

# Run only specific tasks
export TASK_IDS=hard,insider
python inference.py

# Adjust success threshold
export SUCCESS_THRESHOLD=0.6
python inference.py
```

---

## Project Structure

```
cybersentinel/
├── server/
│   └── app.py              # FastAPI server v2.0.0 (port 7860)
├── environment.py           # Core environment (reset/step/state)
├── cybersentinel_env.py     # Async client w/ from_docker_image()
├── graders.py               # GraderConfig + deterministic grading
├── models.py                # Pydantic v2 typed models
├── tasks.py                 # 5 pre-built task scenarios
├── inference.py             # Baseline inference (OpenEnv stdout format)
├── test_env.py              # Smoke tests (13 tests)
├── openenv.yaml             # OpenEnv metadata
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container deployment
└── README.md                # This file
```

### Inference Stdout Format

```
[START] task=hard env=cybersentinel model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=classify('ALERT-H001','true_positive',pri=5) reward=0.75 done=false error=null
[STEP] step=2 action=classify('ALERT-H002','true_positive',pri=5) reward=0.70 done=false error=null
...
[END] success=true steps=17 score=0.62 rewards=0.75,0.70,...
[SUMMARY] tasks=5 avg_score=0.6300 scores={"easy":0.80,"zero":0.90,...}
```

---

## What Makes This Challenging

1. **Adversarial disinformation**: 3 planted intel reports in the hard task actively mislead agents. The reward function penalises agents that cite disinformation reports to misclassify TPs as FPs.

2. **Kill-chain correlation**: Attacks span multiple alerts. Justifications that cross-reference sibling alerts earn a correlation bonus.

3. **Insider threat without signatures**: No malware, no C2, valid certs — requires reasoning about behavioural anomalies and role-based access policies.

4. **Over-trigger bias test**: The zero-alert shift ensures agents don't over-classify. A model that sees every alert as suspicious will score near 0 on this task.

5. **`NEEDS_ESCALATION` is meaningful**: Some cases genuinely require a senior analyst. Correctly escalating rather than guessing earns 60% credit.

---

## Deploying to Hugging Face Spaces

```bash
# Push to HF Space
git remote add hf https://huggingface.co/spaces/<username>/cybersentinel
git push hf main
```

## License

MIT
