\# Shadow-KYC



\### Multi-Layer Video KYC Integrity and Deepfake Risk Analysis



Shadow-KYC is a video integrity analysis application designed to identify suspicious indicators during remote Know Your Customer (KYC) verification. It examines facial behavior, identity consistency, visual artifacts, facial geometry, mouth movement, and environmental signals through a seven-layer analysis pipeline.



Rather than relying on a single facial recognition result, the system combines multiple signals to produce a session-level risk assessment with supporting indicators and evidence.



> Shadow-KYC is a security and fraud-analysis prototype. Its results are risk indicators, not guaranteed proof of authenticity or fraud.



\---



\## Overview



Remote video verification can be exposed to replay attacks, presentation attacks, synthetic facial media, face manipulation, and other forms of video tampering.



Shadow-KYC provides an additional video integrity analysis layer that can be used alongside existing KYC workflows.



The application supports:

\- Live video session analysis.

\- Recorded video analysis.

\- Facial landmark and temporal analysis.

\- Multi-layer risk evaluation.

\- Evidence and report generation.

\- Customer and officer authentication workflows.

\- KYC session and document workflows.



\## Seven-Layer Analysis Pipeline



The application uses a single orchestrated pipeline consisting of seven complementary analysis layers. These are components of one application, not autonomous AI agents.



| Layer | Component | Purpose |

|---|---|---|

| L1 | Temporal Liveness | Examines blinking, head movement, facial landmarks and frozen-frame behavior. |

| L2 | Face Match | Monitors facial representation consistency and identity drift. |

| L3 | Texture and Artifact Analysis | Examines texture and frequency characteristics, including presentation and synthetic-media indicators. |

| L4 | Geometry Integrity | Evaluates facial landmark relationships, proportions and geometric changes. |

| L5 | Lip-Sync Integrity | Examines mouth movement and related speech activity indicators. |

| L6 | Environment Integrity | Checks background movement, lighting changes and visual inconsistencies. |

| L7 | Meta-Fusion | Combines layer outputs, contradiction rules and critical security vetoes into a risk assessment. |



\### Analysis Flow



```text

&#x20;            Live Camera / Recorded Video

&#x20;                        |

&#x20;                        v

&#x20;              Frame Sampling

&#x20;                        |

&#x20;                        v

&#x20;                Face Detection

&#x20;                        |

&#x20;                        v

&#x20;               Session Orchestrator

&#x20;                        |

&#x20;         +--------------+--------------+

&#x20;         |              |              |

&#x20;         v              v              v

&#x20;      L1 Liveness    L2 Face Match   L3 Texture

&#x20;         |              |              |

&#x20;         +--------------+--------------+

&#x20;                        |

&#x20;         +--------------+--------------+

&#x20;         |              |              |

&#x20;         v              v              v

&#x20;      L4 Geometry    L5 Lip-Sync    L6 Environment

&#x20;         |              |              |

&#x20;         +--------------+--------------+

&#x20;                        |

&#x20;                        v

&#x20;                  L7 Meta-Fusion

&#x20;                        |

&#x20;                        v

&#x20;             Risk Assessment and Evidence

&#x20;                        |

&#x20;                        v

&#x20;                Session Report

```



\## Risk Evaluation



The meta-fusion engine combines the available layer signals using weighted risk contributions.



\### Layer weights



| Layer | Weight |

|---|---:|

| L1 – Temporal Liveness | 0.25 |

| L2 – Face Match | 0.20 |

| L3 – Texture Analysis | 0.25 |

| L4 – Geometry | 0.10 |

| L5 – Lip-Sync | 0.10 |

| L6 – Environment | 0.10 |

| \*\*Total\*\* | \*\*1.00\*\* |



Unavailable layer values are handled through active-weight normalization rather than automatically treating every unavailable signal as suspicious.



\### Additional security checks



The fusion engine also evaluates combinations of suspicious indicators. Examples include:

\- Unnatural facial stability alongside background motion.

\- Face-swap indicators combined with frozen-frame replay.

\- Facial smoothing combined with geometric distortion.

\- Frozen mouth movement alongside background activity.



Critical spoof indicators can trigger security vetoes that escalate the risk assessment independently of the ordinary weighted score.



\### Risk categories



The frame-level fusion engine uses these thresholds:



| Risk score | Classification |

|---|---|

| Below 0.30 | LOW RISK |

| 0.30 to below 0.60 | MEDIUM RISK |

| 0.60 and above | HIGH RISK |



Session-level decisions can also consider active liveness completion, deepfake indicators, and the number of flagged frames.



\## Technology Stack



\### Backend

\- Python

\- FastAPI

\- Uvicorn

\- OpenCV

\- MediaPipe

\- NumPy

\- SciPy

\- ONNX model support

\- WebSockets



\### Frontend

\- React 18

\- Vite

\- React Router

\- Chart.js

\- React Chart.js 2

\- Lucide React



\### Authentication and Data

\- JWT-based authentication

\- Argon2 password hashing

\- Optional Supabase integration

\- Local backend persistence



\### Model Assets

\- MediaPipe Face Landmarker

\- MiniFASNet-V2 ONNX model



\## Project Structure



```text

Shadow-KYC/

│

├── ShadowKYC2-main/

│   └── ShadowKYC2-main/

│       ├── app/

│       │   ├── api/

│       │   ├── core/

│       │   │   ├── config.py

│       │   │   ├── orchestrator.py

│       │   │   ├── session\_manager.py

│       │   │   └── types.py

│       │   ├── models/

│       │   ├── modules/

│       │   │   ├── layer\_1\_temporal\_liveness.py

│       │   │   ├── layer\_2\_face\_match.py

│       │   │   ├── layer\_3\_texture\_artifact.py

│       │   │   ├── layer\_4\_geometry.py

│       │   │   ├── layer\_5\_lipsync.py

│       │   │   ├── layer\_6\_environment.py

│       │   │   ├── layer\_7\_meta\_fusion.py

│       │   │   ├── challenge\_manager.py

│       │   │   ├── deepfake\_detector.py

│       │   │   ├── evidence\_manager.py

│       │   │   └── report\_generator.py

│       │   ├── auth.py

│       │   ├── database.py

│       │   ├── main.py

│       │   └── supabase\_client.py

│       │

│       ├── frontend/

│       │   ├── public/

│       │   └── src/

│       │       ├── components/

│       │       ├── lib/

│       │       ├── pages/

│       │       ├── App.jsx

│       │       └── main.jsx

│       │

│       ├── docs/

│       ├── tests/

│       ├── evidence/

│       ├── reports/

│       ├── temp\_uploads/

│       ├── requirements.txt

│       ├── .env.example

│       └── supabase\_schema.sql

│

├── run.bat

├── run.ps1

└── README.md

```



\## Getting Started



\### Prerequisites



\- Python 3.13

\- Node.js and npm

\- Git

\- A webcam for live camera analysis



\### 1. Clone the repository



```powershell

git clone https://github.com/Gowsh7k/Shadow-KYC.git

cd Shadow-KYC

```



\### 2. Navigate to the application directory



```powershell

cd ShadowKYC2-main\\ShadowKYC2-main

```



\### 3. Set up the Python environment



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 4. Configure environment variables



Create a local `.env` file:



```powershell

Copy-Item .env.example .env

```



Set a secure `SECRET\_KEY` in the `.env` file.



The configuration template also provides optional Supabase settings. Local backend authentication is available when Supabase is not configured.



Never commit the actual `.env` file or secret credentials.



\### 5. Start the backend



From the application directory:



```powershell

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

```



Backend:



```text

http://127.0.0.1:8000

```



API documentation:



```text

http://127.0.0.1:8000/docs

```



\### 6. Start the frontend



Open a second PowerShell window:



```powershell

cd Shadow-KYC\\ShadowKYC2-main\\ShadowKYC2-main\\frontend

npm install

npm run dev

```



The frontend uses Vite for development. Its proxy configuration forwards supported backend requests to the FastAPI server.



\## API Capabilities



The FastAPI application exposes endpoints covering:



| Area | Example routes |

|---|---|

| Health | `/health`, `/ping` |

| Authentication | `/auth/signup`, `/auth/login`, `/auth/logout` |

| Video analysis | `/analyze-session`, `/upload-recording` |

| Live sessions | `/session/create`, `/ws/session/{session\_id}/{role}` |

| KYC applications | `/kyc/applications` |

| KYC sessions | `/kyc/sessions` |

| Documents | `/documents/upload`, `/documents` |

| Verification | `/verification/results` |

| Decisions | `/kyc/decisions` |

| Queries | `/queries` |

| Audit | `/audit` |



For request parameters, response schemas, and interactive testing, use the FastAPI documentation at `/docs`.



\## Running Tests



From the application directory:



```powershell

python -m pytest tests/ -v

```



The repository contains tests covering backend behavior, deepfake analysis, and LiveGuard-related functionality.



\## Security and Privacy Considerations



Video KYC data can contain sensitive personal and biometric information.



Before any real-world deployment:

\- Use strong secrets and restrict access to credentials.

\- Configure appropriate CORS origins.

\- Protect uploaded recordings and evidence files.

\- Apply access controls to customer and officer workflows.

\- Define secure retention and deletion policies.

\- Evaluate tenant and session isolation.

\- Validate detection performance using representative datasets.

\- Ensure that human review and applicable KYC policies remain part of the decision process.



\## Limitations



The system should be treated as a prototype and decision-support layer.



Detection performance may be affected by:

\- Poor lighting and camera quality.

\- Face occlusion and extreme head angles.

\- Compression and frame loss.

\- Legitimate unusual facial movement.

\- Changes in recording environments.

\- New synthetic-media techniques.



The risk score should not be treated as definitive proof of identity or fraud. It should not be the sole basis for financial, identity, or compliance decisions.



\## License



Add or confirm the project's chosen license before public redistribution or production use.



\---



\*\*Project Goal:\*\* Make video KYC harder to fool by evaluating not only the identity being presented, but also whether the video itself demonstrates consistent and trustworthy characteristics.



