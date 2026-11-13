# ShadowKYC2 — Complete Application Repair Report

**Project root:** `D:\ShadowKYC2-main\ShadowKYC2-main\`
**Backend venv:** `D:\ShadowKYC2-main\venv` (Python 3.13.5)
**Node:** v24.18.0 / npm 12.2.1
**Report date:** 2026-11-05

> No passwords, tokens, API keys, password hashes or identity information are recorded in this report.

---

## 1. Problems discovered (baseline)

### Backend startup / filesystem paths
- `app/main.py:282` mounted `StaticFiles(directory="frontend/dist/assets")` using a **relative path** → crashed when started from any CWD other than the project root, and crashed when the frontend build was missing.
- No validation that static directories exist before mounting.
- The catch-all route blindly returned `frontend/dist/index.html` even when the build was absent.
- `database.DB_FILE = "db.json"` and `session_manager.SESSIONS_FILE = "sessions_state.json"` were relative → broke when CWD differed.

### Computer vision / MediaPipe
- `mediapipe 1.2.2` installed is the **Tasks-only build**: `mp.solutions` does **not** exist. All seven-layer code imported `mp.solutions.face_detection` / `mp.solutions.face_mesh` → `AttributeError` → `SessionOrchestrator` failed to initialize (orchestrator never started).
- `mediapipe.python.solutions` does not exist either.
- `opencv-python` / `cv2` was **not installed** in the venv → Haar Cascade fallback could not load; OpenCV-dependent layers (L3, L6) could not run.
- `argon2`, `itsdangerous`, `python-dotenv`, `pillow` were missing in the venv (argon2 since installed; dotenv/pillow already present in user site).
- No bundled MediaPipe model files. `deepfake_model.tflite` is a placeholder path; does not exist.
- L1/L4/L5 silently degraded to a fixed score (0.7/0.5) when MediaPipe was missing instead of reporting UNAVAILABLE.

### Authentication
- `/auth/signup` **always returned HTTP 403** ("Signup is disabled").
- `/auth/login` permitted **only** `client@test.com` and `tenant@test.com`, role tied to email.
- Passwords verified with **unsalted SHA-256**; debug prints leaked the stored hash and input hash to server logs.
- Tokens were permanent `secrets.token_hex(32)` with **no expiry**, no revocation, no logout endpoint.
- No password confirmation, no email normalization, no duplicate-email rejection.
- Public signup could create privileged roles (no backend role enforcement).

### Database / persistence
- `save_async()` used **daemon threads** that could be killed mid-write → `db.json` was observed corrupted (`Expecting ',' delimiter`).
- `_load_db()` printed an error and continued with empty stores on corrupt JSON (no backup of corrupt file).
- Supabase writes fire-and-forget, correctly disabled when env vars missing.

### Frontend
- `Login.jsx` / `Signup.jsx` already called `/auth/login` and `/auth/signup` (backend blocked signup).
- `App.jsx` `ProtectedRoute` trusted `localStorage` token presence only, no expiry check.
- `vite.config.js` proxy targets `http://127.0.0.1:8000` — consistent with backend routes (no change needed).
- No `frontend/.env`; Supabase not configured → local backend auth path used by design.

### Verification layers
- L1, L4, L5 required MediaPipe FaceMesh (legacy `solutions`) — unavailable with MediaPipe 1.2.2 Tasks build.
- L2 (HOG-PCA), L3 (OpenCV/FFT), L6 (OpenCV), L7 (rule fusion) work once dependencies installed.
- No layer reported `UNAVAILABLE`; missing model = degraded numeric score (could be misread as a prediction).
- No input validation for missing frames / bad image formats.

### Dependencies / build
- `requirements.txt` listed packages not installed/incompatible with Python 3.13 + MediaPipe 1.2.2 (e.g. `tensorflow`, `opencv-python`).
- Frontend `node_modules/` present and `esbuild`/`vite` working; production build succeeds.

---

## 2. Fixes applied

| Area | Fix |
|------|-----|
| **Paths** | All filesystem paths resolved via `config.BASE_DIR` (project root). `database.DB_FILE` and `session_manager.SESSIONS_FILE` now absolute. Static mounts (`/evidence`, `/assets`, `/uploads`) validate existence before mounting and skip gracefully when missing. SPA catch-all returns index.html when the build exists, otherwise a helpful 404 setup message. |
| **Static files** | `/evidence`, `/assets`, `/uploads` mounted only when the corresponding directory exists. `/ping` returns `{"status":"ok","ping":"pong"}` (200). `/docs`, `/health` preserved. |
| **Startup/shutdown** | `startup()` writes `startup_test.log` to project root, connects DB, persists session state, bulk-syncs to Supabase (fire-and-forget). `shutdown()` closes DB and saves session state. |
| **Auth** | Argon2id (`argon2-cffi`) for new passwords. Existing demo accounts keep original hashes via legacy SHA-256 fallback (passwords never overwritten). JWT access tokens (HS256) with expiry + `jti`, persisted and revocable. `/auth/logout` revokes the token. `/auth/signup` enabled (customer `user` role only; tenant rejected 403; duplicate email 409; email normalized; password ≥ 8 chars). Role enforced on backend (frontend role not trusted). No password/token logging. |
| **Database** | `_save_db()` now writes to a temp file then `os.replace()` (atomic) so crashes can never corrupt `db.json`. `save_async()` now performs an atomic synchronous save. `_load_db()` backs up a corrupt `db.json` to `.corrupt` and starts empty instead of crashing. |
| **MediaPipe / CV** | Face detection via OpenCV Haar Cascade (bundled `cv2.data.haarcascades`). Facial landmarks via MediaPipe Tasks `FaceLandmarker` using bundled `app/models/face_landmarker.task` (downloaded). Legacy `mp.solutions` no longer referenced. `FaceDetector.diagnostics()` reports every detector's availability. |
| **Layers L1/L4/L5** | Use MediaPipe Tasks landmarks; report `UNAVAILABLE` (never a fabricated prediction) when the landmarker is missing. Frozen-frame/hash checks work without MediaPipe. |
| **Layers L2/L3/L6** | Always available (OpenCV/NumPy). Return structured status (`PASS`/`FAIL`/`ERROR`). |
| **Layer L7** | Fusion treats missing/unavailable layers as neutral (goodness 1.0) — never treated as fraud. Returns `layer_status` per layer. |
| **Orchestrator** | Shares one `FaceDetector` across L1/L4/L5. Emits `detectors` and `layer_status` in results. Neutral-scoring for unavailable layers. |
| **Frontend** | New `src/lib/auth.js` (token get/set/clear, JWT client-side expiry check, `isAuthenticated`). `App.jsx` `ProtectedRoute` uses `isAuthenticated()`. Global fetch + XHR interceptor clears auth and redirects to `/login` on 401. `Login.jsx`/`Signup.jsx`/`Home.jsx`/`TenantMonitor.jsx` use `setAuth`/`clearAuth`. Production build regenerated. |
| **Config** | `requirements.txt` pinned to tested versions. `.env.example` created (SECRET_KEY, token expiry, CORS, optional Supabase). `.gitignore` now excludes `.env`, `frontend/.env`, `temp_uploads/`, `evidence/`, `reports/`, `sessions_state.json`, logs. |
| **Tests** | `tests/test_backend.py` — 24 tests: startup/health, signup, duplicate rejection, tenant rejection, validation, demo login, invalid login, role mismatch, token logout/revocation, missing token, face detector, MediaPipe fallback, all 7 layers, missing-frame validation, orchestrator, DB preservation. |

---

## 3. Test results

### Backend tests (`pytest tests/test_backend.py -v`)
**24 passed, 0 failed.**

| Test | Result |
|------|--------|
| app imports + /ping /health /docs | PASS |
| signup new customer | PASS |
| duplicate email rejected (409) | PASS |
| tenant signup rejected (403) | PASS |
| signup validation (400) | PASS |
| demo login client | PASS |
| demo login tenant | PASS |
| invalid login (401) | PASS |
| role mismatch login (403) | PASS |
| token logout + revocation (401 after logout) | PASS |
| missing token rejected (401) | PASS |
| face detector initialization | PASS |
| MediaPipe landmarker / fallback | PASS |
| L1 structured result | PASS |
| L2 structured result | PASS |
| L3 structured result | PASS |
| L4 structured result | PASS |
| L5 structured result | PASS |
| L6 structured result | PASS |
| L7 fusion structured result | PASS |
| missing-frame input validation | PASS |
| orchestrator single frame | PASS |
| database preserves existing users | PASS |
| startup/shutdown handlers | PASS |

### End-to-end integration test (`test_integration.py` against live server)
**All scenarios PASS:**
signup → login(new account) → login(demo client) → login(demo tenant) → invalid login → role mismatch → no-token 401 → session create with token → logout → revoked-token 401 → duplicate signup → SPA catch-all serves index.html (200).

### CV pipeline (synthetic image)
- FaceDetector: Haar Cascade available, MediaPipe Tasks FaceLandmarker available (478 landmarks detected).
- L1 0.80, L2 1.00, L3 0.75, L4 1.00, L5 1.00, L6 1.00 → Fusion risk 0.0875 LOW_RISK.
- Missing frame → layers return `ERROR` status (never fabricate).

### Frontend build
`npm run build` → `✓ built in ~2s` (dist/ regenerated with index-CdDep4B5.js).

### Startup from different working directory
Verified: server launched with CWD=`C:\Users\Public` and `PYTHONPATH` pointing at the project root; `/ping`, `/health`, `/docs` all returned 200; `db.json` loaded from the project root (not CWD).

---

## 4. Remaining limitations

- **MediaPipe landmarker model** (`face_landmarker.task`, 3.7 MB) is bundled in `app/models/`. It was downloaded from Google storage; if the file is absent, L1/L4/L5 correctly report `UNAVAILABLE`.
- **Face detection** uses OpenCV Haar Cascade (reliable, offline). Landmark-based liveness/geometry/lipsync require the MediaPipe model; without them those layers report `UNAVAILABLE` rather than predicting.
- **No deepfake classifier** — `deepfake_model.tflite` remains a placeholder; no claim of validated accuracy is made (consistent with the project disclaimer).
- **Tokens** use `SECRET_KEY`; a development default is used only when the env var is unset (a startup warning is printed). Production must set `SECRET_KEY`.
- **Supabase** is optional; when not configured, all writes are local-only (correctly identified as local persistence).
- **Session state** (`sessions_state.json`) and **db.json** are local files; for multi-server production deployments they would need a shared store (out of scope for this repair).
- The system is a **risk-analysis layer**, not a bank-grade or regulatorially compliant deepfake detector (no independent validation performed).
- `db.json` may accumulate test-created accounts during repeated test runs; existing demo accounts are never modified.

---

## 5. Files changed

Backend:
- `app/main.py` — paths, static mounts, auth (signup/login/logout), startup/shutdown, pyzrab guard
- `app/auth.py` — rewritten: Argon2id, JWT, expiry, revocation, legacy SHA-256 fallback
- `app/database.py` — atomic saves, robust load, absolute DB path
- `app/core/session_manager.py` — absolute sessions file path
- `app/core/config.py` — statuses, model path, BASE_DIR
- `app/core/orchestrator.py` — shared FaceDetector, diagnostics, neutral scoring, detectors output
- `app/modules/face_detector.py` — Haar + MediaPipe Tasks landmarker, diagnostics
- `app/modules/layer_1_temporal_liveness.py` — MediaPipe Tasks, UNAVAILABLE reporting
- `app/modules/layer_2_face_match.py` — l2_status in all paths
- `app/modules/layer_3_texture_artifact.py` — l3_status
- `app/modules/layer_4_geometry.py` — MediaPipe Tasks, UNAVAILABLE reporting
- `app/modules/layer_5_lipsync.py` — MediaPipe Tasks, UNAVAILABLE reporting
- `app/modules/layer_6_environment.py` — l6_status
- `app/modules/layer_7_meta_fusion.py` — neutral unavailable layers, layer_status
- `app/models/face_landmarker.task` — bundled MediaPipe model
- `requirements.txt` — pinned/tested dependencies
- `.env.example` — new
- `.gitignore` — secrets and runtime data excluded
- `README.md` — Windows setup/startup commands

Frontend:
- `frontend/src/lib/auth.js` — new
- `frontend/src/lib/supabase.js` — unchanged (already degrades gracefully)
- `frontend/src/App.jsx` — ProtectedRoute uses isAuthenticated()
- `frontend/src/main.jsx` — global 401 → clear auth + redirect to /login
- `frontend/src/pages/Login.jsx` — setAuth helper
- `frontend/src/pages/Signup.jsx` — setAuth after signup
- `frontend/src/pages/Home.jsx` — clearAuth on logout
- `frontend/src/pages/TenantMonitor.jsx` — clearAuth on logout
- `frontend/dist/` — regenerated production build

Tests:
- `tests/__init__.py`, `tests/test_backend.py` — new
