# System Architecture Flow Chart

This document describes the runtime architecture of `evacuAI-main` (StampedeGuard Pro).

## High-Level Flow

```mermaid
flowchart TD
    U[Operator / Browser Dashboard] -->|HTTP GET/POST| R[Flask Routes<br/>`api/routes.py`]
    U -->|SSE subscribe| SSE[/`/sse` stream/]
    R -->|Render UI| T[Templates<br/>`templates/*.html`]
    R -->|Video stream request| VF[/`/video_feed/<camera>`/]
    VF --> G[Frame Generator<br/>`services/vision_engine.py`]

    C1[Camera Sources<br/>IP Cam / Webcam / Video Files] --> G
    S[Config + Thresholds<br/>`config/settings.py`] --> G
    S --> R

    G --> D[YOLOv8 Person Detection]
    D --> O[Optical Flow + Turbulence + Acceleration]
    O --> RK[Risk Scoring Engine]
    RK --> ST[(In-Memory Shared State<br/>`core/state.py`)]

    ST --> R
    R --> M[/`/metrics`, `/camera_health`, `/entry_count`/]
    R --> GH[/`/graph/<camera>` + incidents pages/]

    RK --> AL[Alert Decision Logic<br/>`services/alert_engine.py`]
    AL --> IL[(Incident Log + SSE Queues)]
    IL --> SSE
    SSE --> U

    AL --> TG[Telegram Service<br/>`telegram_service.py`]
    AL --> TW[Twilio SMS Service<br/>`twilio_service.py`]
    AL --> BZ[Local Beep Alarm]
```

## Component Responsibilities

- `app.py`: Flask entry point and blueprint registration.
- `api/routes.py`: Dashboard, camera streams, metrics APIs, SSE endpoint, and control actions.
- `services/vision_engine.py`: Per-frame pipeline (orientation, YOLO detection, optical flow analytics, risk computation, overlays, stream encoding).
- `core/state.py`: Global thread-shared runtime state (metrics, history, incident logs, alert cooldowns, SSE client queues).
- `services/alert_engine.py`: Critical/high alert persistence logic, cooldown checks, incident recording, and outbound notification triggers.
- `telegram_service.py` + `twilio_service.py`: External notification adapters.
- `config/settings.py`: Camera sources, ROI definitions, density thresholds, flow parameters, and alert timing constants.

## Runtime Sequence (Per Camera)

1. Client requests `GET /video_feed/<camera>`.
2. Route invokes `generate_frames(camera)` from `vision_engine`.
3. Engine reads camera frame, applies orientation + resize.
4. YOLOv8 detects persons; ROI density and optical-flow features are computed.
5. Composite risk score and risk level are derived.
6. Shared state is updated (`camera_metrics`, `crowd_history`, health/fps).
7. Alert engine evaluates persistence + cooldown and triggers notifications if needed.
8. Annotated JPEG frame is streamed back to dashboard in multipart format.

## External Integrations

- **Telegram Bot API** for text/image/voice alerts.
- **Twilio API** for SMS alerts.
- `.env` secrets loaded at startup (`python-dotenv`).
