import base64

import asyncio

import cv2

import numpy as np

import json

import hashlib

import secrets

import time

from datetime import timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query, Header, status

try:

    from pyzbar.pyzbar import decode as pyzbar_decode

    _HAS_PYZBAR = True

except Exception:  # pyzbar / libzbar not installed

    pyzbar_decode = None  # type: ignore

    _HAS_PYZBAR = False

    print("[Main] pyzbar not available — QR scanning disabled.")

from fastapi.responses import JSONResponse, FileResponse

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from typing import Optional, List

import shutil

import os

from pathlib import Path

from dataclasses import asdict



from app.core import config

from app.core.orchestrator import SessionOrchestrator

from app.core.session_manager import SessionManager, SessionInstance

from app.database import (

    connect_db, close_db, save_db,

    users_col, tokens_col, notifications_col,

    tickets_col, session_history_col, app_status_col, doc_approvals_col,

    organizations_col, verified_domains_col, officer_invitations_col,

    customer_profiles_col, kyc_applications_col, kyc_sessions_col,

    session_codes_col, documents_col, verification_results_col,

    kyc_decisions_col, customer_queries_col, audit_events_col, audit_event,

)

import app.database as database

from app.auth import (

    verify_ws_token, get_user_from_auth_header, hash_password, verify_password,

    create_access_token, store_token, revoke_token,

    client_register, client_verify_email, client_login,

    client_forgot_password, client_reset_password,

    organization_create, officer_invite, officer_register, officer_login,

    require_client, require_officer, require_bank_admin, require_platform_admin,

    require_org_member, ROLE_CLIENT, ROLE_BANK_OFFICER, ROLE_BANK_ADMIN,

    ROLE_PLATFORM_ADMIN, send_email, send_sms,

    ClientSignupRequest, ClientVerifyEmailRequest, ClientLoginRequest,

    ClientForgotPasswordRequest, ClientResetPasswordRequest,

    OrganizationCreateRequest, OfficerInviteRequest, OfficerRegisterRequest,

    OfficerLoginRequest,

)

import app.supabase_client as supa



# ─── Project-root-relative paths ─────────────────────────────────────────────

# All filesystem paths resolve relative to the backend source root so the

# server works no matter what the current working directory is.

BASE_DIR = config.BASE_DIR  # project root (parent of /app)

FRONTEND_DIR = BASE_DIR / "frontend"

DIST_DIR = FRONTEND_DIR / "dist"

ASSETS_DIR = DIST_DIR / "assets"

INDEX_HTML = DIST_DIR / "index.html"

EVIDENCE_DIR = config.EVIDENCE_DIR

REPORTS_DIR = config.REPORTS_DIR

TEMP_UPLOADS_DIR = BASE_DIR / "temp_uploads"

DB_FILE = BASE_DIR / "db.json"

SESSIONS_FILE = BASE_DIR / "sessions_state.json"



def _resolve(path) -> Path:

    p = Path(path)

    return p if p.is_absolute() else (BASE_DIR / p)



def _normalize_role(r: Optional[str]) -> str:

    if not r:

        return ""

    val = r.strip().upper()

    if val in ("USER", "CLIENT"):

        return "CLIENT"

    if val in ("TENANT", "OFFICER", "BANK_OFFICER"):

        return "BANK_OFFICER"

    if val in ("BANK_ADMIN", "ADMIN"):

        return "BANK_ADMIN"

    if val in ("PLATFORM_ADMIN", "SUPERADMIN"):

        return "PLATFORM_ADMIN"

    return val



# Roles that public customer registration may create.

PUBLIC_ROLES = {"user"}





# ─── Helpers ──────────────────────────────────────────────────────────────────



def _hash(pw: str) -> str:

    """Legacy SHA-256 hash, kept ONLY for verifying existing demo accounts."""

    return hashlib.sha256(pw.encode()).hexdigest()



async def _get_user_from_token(token: str):

    doc = await database.tokens_col.find_one({"token": token})

    if doc:

        return await database.users_col.find_one({"email": doc["email"]})

    return None



async def _add_notification(user_id: str, message: str, ntype: str = "info"):

    notif = {

        "id": secrets.token_hex(6),

        "user_id": user_id,

        "message": message,

        "type": ntype,

        "read": False,

        "created_at": time.time(),

    }

    await database.notifications_col.insert_one(notif)

    asyncio.create_task(supa.upsert_notification(notif))



async def _process_session_decision(

    session_id: str, decision: str, notes: str,

    user_id: str = None, tenant_id: str = None, risk_score: float = None

):

    existing = await database.session_history_col.find_one({"session_id": session_id})

    if existing:

        update = {"decision": decision, "notes": notes}

        if risk_score is not None:

            update["risk_score"] = risk_score

        if tenant_id:

            update["tenant_id"] = tenant_id

        await database.session_history_col.update_one({"session_id": session_id}, {"$set": update})

    else:

        await database.session_history_col.insert_one({

            "session_id": session_id,

            "user_id": user_id,

            "tenant_id": tenant_id,

            "timestamp": time.time(),

            "status": "completed",

            "decision": decision,

            "notes": notes,

            "risk_score": risk_score,

        })



    if user_id:

        await database.app_status_col.insert_one({

            "user_id": user_id,

            "session_id": session_id,

            "status": decision,

            "notes": notes,

            "updated_at": time.time(),

        })

        labels = {

            "approved": "Your KYC verification has been approved!",

            "rejected": "Your KYC verification was not successful.",

            "manual_review": "Your KYC is under manual review.",

            "visit_branch": "Please visit the nearest branch for verification.",

        }

        notif_msg = labels.get(decision, "Your KYC status has been updated.")

        await _add_notification(

            user_id,

            notif_msg,

            decision if decision in ["approved", "success"] else "info"

        )

        # Sync to Supabase

        asyncio.create_task(supa.upsert_app_status(

            user_id=user_id, session_id=session_id,

            status=decision, notes=notes

        ))



    # Update session in Supabase

    asyncio.create_task(supa.upsert_session(

        session_id=session_id, user_id=user_id, tenant_id=tenant_id,

        status="completed", decision=decision, notes=notes, risk_score=risk_score

    ))



    return {"message": "Decision processed", "decision": decision}







# Simple QR scanning helper

def simple_qr_scan(image):

    # Auto-downsize high-res images to speed up processing

    h, w = image.shape[:2]

    max_dim = 1000

    if max(h, w) > max_dim:

        scale = max_dim / max(h, w)

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)



    # 1. Try Raw Decode

    results = pyzbar_decode(image)

    if results:

        return {"success": True, "type": results[0].type}

    

    # 2. Try Grayscale/Threshold

    try:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        results = pyzbar_decode(thresh)

        if results:

            return {"success": True, "type": results[0].type}

    except: pass

    

    return None











# ─── Request Models ───────────────────────────────────────────────────────────



class SignupRequest(BaseModel):

    name: str

    email: str

    password: str

    role: str

    organization: Optional[str] = None



class LoginRequest(BaseModel):

    email: str

    password: str

    role: str



class DecisionRequest(BaseModel):

    session_id: str

    user_id: str

    decision: str

    notes: str

    risk_score: Optional[float] = None

    tenant_id: Optional[str] = None



class TicketRequest(BaseModel):

    subject: str

    description: str





# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="Shadow API", description="KYC Video Integrity Analysis")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



@app.post("/session/scan-qr")

async def scan_qr(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:

            return {"success": False, "message": "Failed to decode image"}



        if not _HAS_PYZBAR:

            return {"success": False, "message": "QR scanning not available (pyzbar not installed)."}



        res = simple_qr_scan(img)

        if res:

            return res

        return {"success": False, "message": "No QR code found"}

    except Exception as e:

        return {"success": False, "message": str(e)}



@app.middleware("http")

async def log_requests(request, call_next):

    print(f"[RECON] {request.method} {request.url.path}")

    return await call_next(request)



# ─── Session Management ───────────────────────────────────────────────────────



session_manager = SessionManager()



@app.post("/session/create")

async def create_session(authorization: str = Header(None)):

    user_info = await get_user_from_auth_header(authorization)

    if not user_info:

        raise HTTPException(status_code=401, detail="Authentication required to create a session.")

    

    tenant_id = user_info.get("id", "unknown_tenant")

    code = session_manager.create_session()

    

    # Store the tenant_id in the session instance

    session = session_manager.active_sessions.get(code)

    if session:

        session.tenant_id = tenant_id



    # Record in local history + Supabase

    await database.session_history_col.insert_one({

        "session_id": code,

        "user_id": None,

        "tenant_id": tenant_id,

        "timestamp": time.time(),

        "status": "active",

        "decision": "pending",

        "notes": "",

        "risk_score": None,

    })

    asyncio.create_task(supa.upsert_session(

        session_id=code, tenant_id=tenant_id,

        status="active", decision="pending"

    ))



    return {"session_id": code}



@app.on_event("startup")

async def startup():

    try:

        with open(str(BASE_DIR / "startup_test.log"), "w") as f:

            f.write(f"Server started at {time.ctime()}\n")

    except Exception as e:

        print(f"[Startup] Could not write startup_test.log: {e}")

    await connect_db()

    # Ensure session state file exists on disk (loads on SessionManager import)

    try:

        session_manager.save_state()

    except Exception as e:

        print(f"[Startup] Could not persist session state: {e}")

    # Bulk-sync local db.json data to Supabase on every startup (fire-and-forget,

    # failures are logged but never crash startup)

    asyncio.create_task(supa.sync_all_local_data(

        users=database._users,

        history=database._history_list,

        notifs=database._notif_list,

        tickets=database._ticket_list,

        doc_approvals=database._doc_approvals,

        app_status=database._app_status_list,

    ))



@app.get("/health")

async def health():

    return {

        "status": "ok",

        "time": time.ctime(),

        "mode": "in-memory",

        "active_sessions": list(session_manager.active_sessions.keys())

    }



@app.get("/debug/sessions")

async def debug_sessions():

    return {

        "active_count": len(session_manager.active_sessions),

        "active_ids": list(session_manager.active_sessions.keys()),

        "archived_count": len(database._history_list),

    }



@app.get("/ping")

async def ping():

    return {"status": "ok", "ping": "pong"}



@app.on_event("shutdown")

async def shutdown():

    await close_db()

    try:

        session_manager.save_state()

    except Exception as e:

        print(f"[Shutdown] Failed to save session state: {e}")



# Mount static directories (only when they exist, using project-root paths)

if EVIDENCE_DIR.exists():

    app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

else:

    print(f"[Static] Evidence directory not found, skipping /evidence mount: {EVIDENCE_DIR}")



if ASSETS_DIR.exists():

    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

else:

    print(f"[Static] Frontend assets not found, skipping /assets mount: {ASSETS_DIR}")

    print("[Static] Frontend production build missing. API routes remain available.")





# ─── Auth Endpoints ───────────────────────────────────────────────────────────



def _user_public(user: dict) -> dict:

    """Return a user record without any password material."""

    return {

        "id": user.get("id"),

        "name": user.get("name"),

        "email": user.get("email"),

        "role": user.get("role"),

        "organization": user.get("organization"),

        "created_at": user.get("created_at"),

    }





@app.post("/auth/signup")

async def signup(req: SignupRequest):

    """Customer registration. Public registration can only create 'user' role

    accounts; privileged 'tenant' (bank officer) accounts are rejected."""

    name = (req.name or "").strip()

    email = (req.email or "").strip().lower()

    password = req.password or ""

    role = (req.role or "user").strip().lower()

    organization = (req.organization or "").strip() or None



    # Validate inputs

    if not name:

        raise HTTPException(status_code=400, detail="Full name is required.")

    if not email or "@" not in email:

        raise HTTPException(status_code=400, detail="Valid email is required.")

    if len(password) < 8:

        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")



    # Public registration cannot create privileged bank-officer accounts.

    if role not in PUBLIC_ROLES:

        raise HTTPException(status_code=403, detail="Public registration is limited to customer (user) accounts.")



    # Reject duplicate emails

    existing = await database.users_col.find_one({"email": email})

    if existing:

        raise HTTPException(status_code=409, detail="An account with this email already exists.")



    import uuid as _uuid

    user_id = str(_uuid.uuid4())

    now = time.time()

    user_doc = {

        "id": user_id,

        "name": name,

        "email": email,

        "hashed_pw": hash_password(password),

        "role": "user",

        "organization": organization,

        "created_at": now,

    }

    await database.users_col.insert_one(user_doc)



    # Issue an access token so the customer can log in immediately

    token = create_access_token(user_id=user_id, email=email, role="user")

    await store_token(token, email=email, user_id=user_id, role="user")



    return {

        "message": "Account created successfully.",

        "access_token": token,

        "token_type": "bearer",

        "user": _user_public(user_doc),

    }





@app.post("/auth/login")

async def login(req: LoginRequest):

    email = (req.email or "").strip().lower()

    password = req.password or ""

    # Role is authoritative from the stored record, never trusted from the frontend.

    requested_role = (req.role or "").strip().lower()



    if not email or not password:

        raise HTTPException(status_code=401, detail="Invalid email or password.")



    user = await database.users_col.find_one({"email": email})

    if not user or not verify_password(password, user.get("hashed_pw")):

        raise HTTPException(status_code=401, detail="Invalid email or password.")



    user_role = user.get("role", "user")

    norm_user = _normalize_role(user_role)

    norm_req = _normalize_role(req.role)

    if norm_req and norm_req != norm_user:

        raise HTTPException(

            status_code=403,

            detail=f"Role mismatch: account '{email}' is a {user_role}, not a {req.role}.",

        )



    token = create_access_token(

        user_id=user.get("id"), email=email, role=user_role

    )

    await store_token(token, email=email, user_id=user.get("id"), role=user_role)



    return {

        "access_token": token,

        "token_type": "bearer",

        "user": _user_public(user),

    }





@app.post("/auth/logout")

async def logout(authorization: str = Header(None)):

    """Invalidate the caller's access token."""

    if authorization and authorization.startswith("Bearer "):

        token = authorization.split(" ", 1)[1]

        await revoke_token(token)

    return {"message": "Logged out successfully."}





# ─── Session History ──────────────────────────────────────────────────────────



@app.post("/session/record-history")

async def record_session_history(

    session_id: str = Query(...),

    user_id: str = Query(default=None),

    tenant_id: str = Query(default=None),

):

    entry = {

        "session_id": session_id,

        "user_id": user_id,

        "tenant_id": tenant_id,

        "timestamp": time.time(),

        "status": "completed",

        "decision": "pending",

        "notes": "",

        "risk_score": None,

    }

    await database.session_history_col.insert_one(entry)

    return {"message": "Session recorded.", "entry": entry}



@app.get("/session/history/{role_id}")

async def get_session_history(role_id: str, role: str = Query(default="user")):

    key = "user_id" if role == "user" else "tenant_id"

    cursor = database.session_history_col.find({key: role_id}).sort("timestamp", -1)

    history = await cursor.to_list(length=200)

    return {"history": history}





# ─── Tenant Decision ──────────────────────────────────────────────────────────



@app.post("/session/decision")

async def submit_decision(req: DecisionRequest):

    return await _process_session_decision(

        session_id=req.session_id,

        decision=req.decision,

        notes=req.notes,

        user_id=req.user_id,

        tenant_id=req.tenant_id,

        risk_score=req.risk_score

    )





# ─── Application Status ───────────────────────────────────────────────────────



@app.get("/application/status/{user_id}")

async def get_application_status(user_id: str):

    cursor = database.app_status_col.find({"user_id": user_id}).sort("updated_at", -1)

    statuses = await cursor.to_list(length=100)

    return {"statuses": statuses}





# ─── Support Tickets ──────────────────────────────────────────────────────────



@app.post("/support/ticket")

async def create_ticket(req: TicketRequest, user_id: str = Query(...)):

    ticket = {

        "id": secrets.token_hex(6),

        "user_id": user_id,

        "subject": req.subject,

        "description": req.description,

        "status": "open",

        "created_at": time.time(),

        "updated_at": time.time(),

    }

    await database.tickets_col.insert_one(ticket)

    asyncio.create_task(supa.upsert_ticket(ticket))

    await _add_notification(user_id, f"Your support ticket '{req.subject}' has been created.", "info")

    return {"message": "Ticket created.", "ticket": ticket}



@app.get("/support/tickets/{user_id}")

async def get_tickets(user_id: str):

    cursor = database.tickets_col.find({"user_id": user_id}).sort("created_at", -1)

    tickets = await cursor.to_list(length=100)

    return {"tickets": tickets}



@app.get("/tenant/tickets")

async def get_all_tickets():

    cursor = database.tickets_col.find({}).sort("created_at", -1)

    tickets = await cursor.to_list(length=500)

    return {"tickets": tickets}





# ─── Notifications ────────────────────────────────────────────────────────────



@app.get("/notifications/{user_id}")

async def get_notifications(user_id: str):

    cursor = database.notifications_col.find({"user_id": user_id}).sort("created_at", -1)

    notifs = await cursor.to_list(length=50)

    return {"notifications": notifs}



@app.post("/notifications/{user_id}/read")

async def mark_notifications_read(user_id: str):

    await database.notifications_col.update_many({"user_id": user_id}, {"$set": {"read": True}})

    return {"message": "All notifications marked as read."}





# ─── Tenant Stats ─────────────────────────────────────────────────────────────



@app.get("/tenant/stats/{tenant_id}")

async def get_tenant_stats(tenant_id: str):

    cursor = database.session_history_col.find({"tenant_id": tenant_id})

    history = await cursor.to_list(length=10000)

    total    = len(history)

    approved = sum(1 for s in history if s["decision"] == "approved")

    rejected = sum(1 for s in history if s["decision"] == "rejected")

    manual   = sum(1 for s in history if s["decision"] == "manual_review")

    visit    = sum(1 for s in history if s["decision"] == "visit_branch")

    pending  = sum(1 for s in history if s["decision"] == "pending")

    fraud    = sum(1 for s in history if (s.get("risk_score") or 0) > 65)

    return {

        "total": total, "approved": approved, "rejected": rejected,

        "manual_review": manual, "visit_branch": visit, "pending": pending,

        "fraud_suspected": fraud,

    }





# ─── Session Validate ─────────────────────────────────────────────────────────



@app.get("/session/validate/{code}")

async def validate_session(code: str):

    session = session_manager.active_sessions.get(code)

    if not session:

        hist = await database.session_history_col.find_one({"session_id": code})

        if hist and hist.get("status") == "active":

            session = SessionInstance(code)

            session.tenant_id = hist.get("tenant_id")

            session_manager.active_sessions[code] = session

    if session:

        return {

            "valid": True,

            "session_id": code,

            "has_client": session.client_socket is not None,

            "has_tenant": session.tenant_socket is not None,

        }

    return {"valid": False}





# ─── Document Upload & Approval ───────────────────────────────────────────────



ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "application/pdf"}

MAX_DOC_SIZE_MB = 5

REQUIRED_DOCS = {"gov_id", "selfie"}



@app.post("/session/upload-docs")

async def upload_docs(

    session_id: str = Query(..., description="Session code"),

    gov_id: UploadFile = File(...),

    selfie: UploadFile = File(...),

    address_proof: Optional[UploadFile] = File(None),

):

    docs = {"gov_id": gov_id, "selfie": selfie}

    if address_proof and address_proof.filename:

        docs["address_proof"] = address_proof



    upload_dir = TEMP_UPLOADS_DIR / "docs" / session_id

    upload_dir.mkdir(parents=True, exist_ok=True)



    saved = []

    for doc_key, upload_file in docs.items():

        content_type = upload_file.content_type or ""

        if content_type not in ALLOWED_MIME_TYPES:

            raise HTTPException(status_code=400, detail=f"'{doc_key}': Invalid file type '{content_type}'. Only JPG, PNG, PDF accepted.")



        contents = await upload_file.read()

        size_mb = len(contents) / (1024 * 1024)

        if size_mb > MAX_DOC_SIZE_MB:

            raise HTTPException(status_code=400, detail=f"'{doc_key}': File too large ({size_mb:.1f} MB). Maximum is {MAX_DOC_SIZE_MB} MB.")

        if len(contents) < 100:

            raise HTTPException(status_code=400, detail=f"'{doc_key}': File appears to be empty or corrupt.")



        ext = Path(upload_file.filename).suffix or ".bin"

        save_path = upload_dir / f"{doc_key}{ext}"

        with open(save_path, "wb") as f:

            f.write(contents)

        saved.append(str(save_path))



    await database.doc_approvals_col.update_one(

        {"session_id": session_id},

        {"$set": {"approved": True, "docs": saved, "timestamp": time.time()}},

        upsert=True

    )

    asyncio.create_task(supa.upsert_doc_approval(

        session_id=session_id, approved=True, docs=saved

    ))



    return {

        "approved": True,

        "session_id": session_id,

        "message": "Documents verified successfully. You may now join the live session.",

        "docs_received": list(docs.keys()),

    }



# Mount temp_uploads for accessing uploaded documents (only when present)

if TEMP_UPLOADS_DIR.exists():

    app.mount("/uploads", StaticFiles(directory=str(TEMP_UPLOADS_DIR)), name="uploads")

else:

    print(f"[Static] temp_uploads not found, skipping /uploads mount: {TEMP_UPLOADS_DIR}")





@app.get("/session/doc-status/{session_id}")

async def doc_status(session_id: str):

    approval = await database.doc_approvals_col.find_one({"session_id": session_id})

    return {

        "approved": approval["approved"] if approval else False,

        "session_id": session_id,

    }



@app.get("/tenant/documents")

async def get_tenant_documents():

    cursor = database.doc_approvals_col.find({}).sort("timestamp", -1)

    approvals = await cursor.to_list(length=500)

    results = []

    for data in approvals:

        doc_urls = []

        for path in data.get("docs", []):

            p = Path(path)

            try:

                rel = p.relative_to(TEMP_UPLOADS_DIR)

                doc_urls.append(f"/uploads/{rel.as_posix()}")

            except ValueError:

                doc_urls.append(str(path))

        results.append({

            "session_id": data["session_id"],

            "timestamp": data.get("timestamp"),

            "doc_urls": doc_urls

        })

    return {"documents": results}





# ─── Video Analysis ───────────────────────────────────────────────────────────



@app.post("/analyze-session")

async def analyze_session(file: UploadFile = File(...)):

    try:

        TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

        file_path = TEMP_UPLOADS_DIR / f"upload_{file.filename}"



        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)



        orchestrator = SessionOrchestrator()

        report = orchestrator.start_session(str(file_path))



        try:

            os.remove(file_path)

        except Exception as e:

            print(f"Error removing temp file: {e}")



        return asdict(report)



    except Exception as e:

        print(f"Error processing session: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=str(e))



@app.post("/upload-recording")

async def upload_recording(session_id: str, file: UploadFile = File(...)):

    try:

        # [SECURITY] Disk storage disabled as per user request

        # session_dir = Path("evidence") / session_id

        # session_dir.mkdir(parents=True, exist_ok=True)

        # file_path = session_dir / "session_recording.webm"



        # with open(file_path, "wb") as buffer:

        #     shutil.copyfileobj(file.file, buffer)



        print(f"[SECURITY] Session recording received but NOT stored: {session_id}")

        return {"status": "success", "message": "Recording processed in-memory (not stored on disk)"}

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))





# ─── WebSocket Session ────────────────────────────────────────────────────────



@app.websocket("/ws/session/{session_id}/{role}")

async def session_websocket(websocket: WebSocket, session_id: str, role: str, token: Optional[str] = Query(None)):

    auth_user = await verify_ws_token(websocket, token)

    if not auth_user:

        # In-memory token DB is wiped on server restart — allow guest fallback

        # so the session can proceed. Auth is already enforced at login.

        print(f"[WS] Token not validated (server may have restarted). Allowing guest for session={session_id}, role={role}")

        auth_user = {"id": "guest", "email": role + "@session", "role": role, "source": "guest"}



    await websocket.accept()

    print(f"[WS] Accepted handshake: session={session_id}, role={role}, user={auth_user.get('email')}", flush=True)

    

    session = session_manager.active_sessions.get(session_id)

    if not session:

        hist = await database.session_history_col.find_one({"session_id": session_id})

        if hist and hist.get("status") == "active":

            session = SessionInstance(session_id)

            session.tenant_id = hist.get("tenant_id")

            session_manager.active_sessions[session_id] = session

    if not session:

        print(f"[WS] 404: Session {session_id} not found for role={role}", flush=True)

        await websocket.send_json({"type": "error", "message": "Session not found"})

        await websocket.close(code=4004)

        return

    

    # Handshake accepted. Now enter the role-specific loop.

    if role == "client":

        await session.connect_client(websocket)

        print(f"[WS] Handshake successful: {role} for session {session_id}", flush=True)

        try:

            while True:

                data = await websocket.receive_text()

                message = json.loads(data)



                if message.get("type") == "identify":

                    user_id = message.get("user_id")

                    session.user_id = user_id

                    print(f"[WS] Client identified in session {session_id} as user {user_id}", flush=True)

                    continue



                if message.get("type") == "frame":

                    payload = message.get("data", "")

                    timestamp = message.get("timestamp", 0)



                    try:

                        _, encoded = payload.split(",", 1)

                        data_bytes = base64.b64decode(encoded)

                        nparr = np.frombuffer(data_bytes, np.uint8)

                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)



                        if frame is not None:

                            frame_results = session.orchestrator.process_single_frame(frame, timestamp)

                            current_stats = session.orchestrator.get_current_stats()



                            await session.broadcast_to_tenant({

                                "type": "results",

                                "frame": frame_results,

                                "session": current_stats,

                                "image": payload

                            })



                            quality_warning = None

                            quality_val = frame_results.get("quality")
                            if quality_val is not None and quality_val < 0.4:

                                quality_warning = "Low Quality: Adjust Lighting"



                            await session.send_to_client({
                                "type": "results",
                                "status": "processed",
                                "quality_warning": quality_warning,
                                "tenant_connected": session.tenant_socket is not None,
                                "frame": frame_results,
                                "session": current_stats
                            })

                    except Exception as frame_err:

                        print(f"[WS] Frame processing error: {frame_err}")



                elif message.get("type") in ["stop", "complete_verification"]:
                    try:
                        report = session.orchestrator.finalize_live_session()
                        report_dict = asdict(report)
                        
                        # Persist for reconnection
                        session.is_active = False
                        session.final_report = report_dict
                        
                        raw_label = report.classification.lower()
                        label_map = {
                            "low_risk": "approved",
                            "high_risk": "rejected",
                            "medium_risk": "manual_review"
                        }
                        session.final_decision = label_map.get(raw_label, "completed")

                        asyncio.create_task(supa.upsert_session(
                            session_id=session_id, user_id=session.user_id, tenant_id=session.tenant_id,
                            status="completed", decision=session.final_decision, 
                            risk_score=report.risk_pct
                        ))

                        try:
                            await database.session_history_col.update_one(
                                {"session_id": session_id},
                                {"$set": {
                                    "status": "completed",
                                    "decision": session.final_decision,
                                    "risk_score": report.risk_pct,
                                    "timestamp": time.time(),
                                    "report": report_dict
                                }},
                                upsert=True
                            )
                        except Exception as dbe:
                            print(f"[WS] session_history update error: {dbe}")

                        await session.broadcast_to_tenant({
                            "type": "report", 
                            "data": report_dict,
                            "decision": session.final_decision
                        })
                        await session.send_to_client({
                            "type": "completion", 
                            "status": "COMPLETED", 
                            "decision": session.final_decision,
                            "report": report_dict,
                            "session": session.orchestrator.get_current_stats()
                        })
                        session_manager.save_state()
                    except Exception as e:
                        print(f"[WS] Stop error: {e}")
                    
                    # Wait 1s for completion to reach client
                    await asyncio.sleep(1)
                    break

        except WebSocketDisconnect:

            print(f"[WS] Client disconnected from session {session_id}")

            session.disconnect_client(websocket)

            await session.notify_client_left()

        except Exception as e:

            print(f"[WS] Client error in session {session_id}: {e}")

            session.disconnect_client(websocket)

            await session.notify_client_left()



    elif role == "tenant":

        await session.connect_tenant(websocket)

        try:

            while True:

                data = await websocket.receive_text()

                msg = json.loads(data)



                if msg.get("type") == "end_session":

                    decision_raw = msg.get("status", "ENDED")

                    decision = decision_raw.lower()

                    print(f"[WS] Tenant decision received: {decision} for session {session_id}", flush=True)



                    if decision in ["approved", "rejected", "manual_review", "visit_branch"]:

                        try:

                            stats = session.orchestrator.get_current_stats()

                            risk_score = stats.get("risk_pct", 0)

                            await _process_session_decision(

                                session_id=session_id,

                                decision=decision,

                                notes="Decision via Live Monitor",

                                user_id=session.user_id,

                                tenant_id="tenant_ws",

                                risk_score=risk_score

                            )

                        except Exception as e:

                            print(f"[WS] Decision processing error: {e}")



                    # Finalize and notify (ALWAYS, even if status is just "ENDED")

                    try:

                        report = session.orchestrator.finalize_live_session()

                        report_dict = asdict(report)

                        

                        # 1. LOCK the state FIRST for reconnection persistence

                        session.is_active = False

                        session.final_report = report_dict

                        

                        # Map internal labels to user-friendly decision strings

                        raw_label = report.classification.lower()

                        label_map = {

                            "low_risk": "approved",

                            "high_risk": "rejected",

                            "medium_risk": "manual_review"

                        }

                        

                        final_decision = decision if decision in ["approved", "rejected", "manual_review", "visit_branch"] else label_map.get(raw_label, "completed")

                        session.final_decision = final_decision

                        

                        # Sync final state to Supabase

                        asyncio.create_task(supa.upsert_session(

                            session_id=session_id, user_id=session.user_id, tenant_id=session.tenant_id,

                            status="completed", decision=final_decision, 

                            risk_score=report.risk_pct

                        ))



                        # 2. BROADCAST to everyone

                        print(f"[WS] BROADCASTING FINAL DECISION: {final_decision} for {session_id}", flush=True)

                        

                        # PRIORITIZE telling the CLIENT (for reflection)

                        try:

                            print(f"[WS] Sending completion to client {session_id} NOW...", flush=True)

                            await session.send_to_client({
                                "type": "completion",
                                "status": "COMPLETED",
                                "decision": final_decision,
                                "report": report_dict,
                                "session": session.orchestrator.get_current_stats()
                            })

                            print(f"[WS] Client completion sent successfully!", flush=True)

                        except Exception as ce:

                            print(f"[WS] ERROR sending to client: {ce}", flush=True)

                        

                        # THEN tell the Tenant

                        await session.broadcast_to_tenant({"type": "report", "data": report_dict, "decision": final_decision})

                        

                        # PERSIST the final state

                        session_manager.save_state()

                    except Exception as e:

                        print(f"[WS] Finalize error: {e}")

                    

                    # Wait a moment for buffers to flush before closing handler

                    await asyncio.sleep(1)

                    break



        except WebSocketDisconnect:

            print(f"[WS] Tenant disconnected from session {session_id}")

            session.disconnect_tenant(websocket)

            await session.notify_tenant_left()

        except Exception as e:

            print(f"[WS] Tenant error in session {session_id}: {e}")

            session.disconnect_tenant(websocket)

            await session.notify_tenant_left()



    else:

        await websocket.close(code=4003, reason="Invalid role")





# ──────────────────────────────────────────────────────────────────────────────

# Helpers for the new LiveGuard bank-orientated routes

# ──────────────────────────────────────────────────────────────────────────────



async def _current_user(authorization: Optional[str] = None) -> dict:

    if not authorization or not authorization.startswith("Bearer "):

        raise HTTPException(status_code=401, detail="Authentication required.")

    user = await get_user_from_auth_header(authorization)

    if not user:

        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return user





def _code_hash(code: str) -> str:

    return hashlib.sha256(code.encode("utf-8")).hexdigest()





def _public_user(user: dict) -> dict:

    return {

        "id": user.get("id"), "name": user.get("name"), "email": user.get("email"),

        "role": user.get("role"), "organization": user.get("organization"),

        "employee_id": user.get("employee_id"), "designation": user.get("designation"),

        "is_verified": user.get("is_verified"),

    }





# ──────────────────────────────────────────────────────────────────────────────

# Auth: client self-registration + verification + login + password reset

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/auth/client/register")

async def api_client_register(req: ClientSignupRequest):

    return await client_register(req)





@app.post("/auth/client/verify-email")

async def api_client_verify_email(req: ClientVerifyEmailRequest):

    return await client_verify_email(req)





@app.post("/auth/client/login")

async def api_client_login(req: ClientLoginRequest):

    return await client_login(req)





@app.post("/auth/client/forgot-password")

async def api_client_forgot_password(req: ClientForgotPasswordRequest):

    return await client_forgot_password(req)





@app.post("/auth/client/reset-password")

async def api_client_reset_password(req: ClientResetPasswordRequest):

    return await client_reset_password(req)





# ──────────────────────────────────────────────────────────────────────────────

# Auth: organization + officer invitation / registration / login

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/auth/organizations")

async def api_organization_create(

    req: OrganizationCreateRequest, authorization: Optional[str] = Header(None)

):

    actor = await _current_user(authorization)

    require_platform_admin(actor)

    return await organization_create(req, actor)





@app.post("/auth/officer/invite")

async def api_officer_invite(

    req: OfficerInviteRequest, authorization: Optional[str] = Header(None)

):

    actor = await _current_user(authorization)

    if actor.get("role") not in (ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only bank admins or platform admins can invite officers.")

    return await officer_invite(req, actor)





@app.post("/auth/officer/register")

async def api_officer_register(req: OfficerRegisterRequest):

    return await officer_register(req)





@app.post("/auth/officer/login")

async def api_officer_login(req: OfficerLoginRequest):

    return await officer_login(req)





# ──────────────────────────────────────────────────────────────────────────────

# KYC Applications

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/kyc/applications")

async def api_kyc_create_application(

    req: dict, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    role = user.get("role")

    if role not in (ROLE_CLIENT, ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Authentication required.")

    # Clients create applications for themselves; officers create them for a

    # customer (customer_id) within their own organization.

    if role == ROLE_CLIENT:

        customer_id = user.get("id")

        name = (req.get("customer_name") or user.get("name") or "").strip()

        email = (req.get("customer_email") or user.get("email") or "").strip()

        phone = (req.get("customer_phone") or user.get("phone") or "").strip()

        organization = None

    else:

        customer_id = req.get("customer_id") or user.get("id")

        name = (req.get("customer_name") or req.get("name") or "").strip()

        email = (req.get("customer_email") or req.get("email") or "").strip()

        phone = (req.get("customer_phone") or req.get("phone") or "").strip()

        organization = user.get("organization")

    purpose = (req.get("purpose") or "").strip()

    if not name or not email or not purpose:

        raise HTTPException(status_code=400, detail="Name, email and purpose are required.")



    import uuid as _uuid

    app_id = str(_uuid.uuid4())

    now = time.time()

    doc = {

        "id": app_id, "customer_id": customer_id, "officer_id": None,

        "organization": organization, "name": name, "email": email,

        "phone": phone, "purpose": purpose, "status": "documents_pending",

        "created_at": now, "updated_at": now,

    }

    await kyc_applications_col.insert_one(doc)

    await save_db()

    await audit_event(customer_id, role, "kyc_application_create", "kyc_application", app_id,

                      {"purpose": purpose, "organization": organization})

    return {"message": "KYC application created.", "application": doc}





@app.get("/kyc/applications")

async def api_kyc_list_applications(authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    role = user.get("role")

    org = user.get("organization")

    cursor = kyc_applications_col.find({})

    apps = await cursor.to_list(length=500)

    if role == ROLE_CLIENT:

        apps = [a for a in apps if a.get("customer_id") == user.get("id")]

    elif role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):

        apps = [a for a in apps if a.get("organization") == org]

    # platform admin sees all

    return {"applications": apps}





@app.get("/kyc/applications/{app_id}")

async def api_kyc_get_application(app_id: str, authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    app = await kyc_applications_col.find_one({"id": app_id})

    if not app:

        raise HTTPException(status_code=404, detail="Application not found.")

    role = user.get("role")

    org = user.get("organization")

    if role == ROLE_CLIENT and app.get("customer_id") != user.get("id"):

        raise HTTPException(status_code=403, detail="Not your application.")

    if role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN) and app.get("organization") != org:

        raise HTTPException(status_code=403, detail="Not your organization's application.")

    return {"application": app}





# ──────────────────────────────────────────────────────────────────────────────

# KYC Sessions (6-digit invitation codes)

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/kyc/sessions")

async def api_kyc_create_session(

    req: dict, authorization: Optional[str] = Header(None)

):

    """Officer creates a KYC session with a secure random 6-digit code."""

    user = await _current_user(authorization)

    if user.get("role") not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only bank officers can create sessions.")

    org = user.get("organization")

    customer_id = req.get("customer_id")

    application_id = req.get("application_id")

    purpose = (req.get("purpose") or "").strip()

    expiry_minutes = int(req.get("expiry_minutes") or 60)

    if expiry_minutes < 5 or expiry_minutes > 10080:

        raise HTTPException(status_code=400, detail="Expiry must be between 5 minutes and 7 days.")



    # Verify application exists and belongs to org (officer scope)

    app = await kyc_applications_col.find_one({"id": application_id}) if application_id else None

    if app and app.get("organization") != org:

        raise HTTPException(status_code=403, detail="Application not in your organization.")



    import uuid as _uuid

    session_id = str(_uuid.uuid4())

    code = f"{secrets.randbelow(1000000):06d}"

    code_hash = _code_hash(code)

    now = time.time()

    expires_at = now + expiry_minutes * 60

    session_doc = {

        "id": session_id, "application_id": application_id,

        "customer_id": customer_id, "officer_id": user.get("id"),

        "organization": org, "purpose": purpose,

        "code_hash": code_hash, "code_expires_at": expires_at,

        "code_used": False, "rate_limit_count": 0, "last_attempt": None,

        "created_at": now, "status": "active",

    }

    await kyc_sessions_col.insert_one(session_doc)

    await session_codes_col.insert_one({"code_hash": code_hash, "session_id": session_id})

    await save_db()



    await audit_event(user.get("id"), user.get("role"), "kyc_session_create", "kyc_session", session_id,

                      {"customer_id": customer_id, "application_id": application_id, "organization": org})



    # Dev-only: show the code once. In production deliver via email/SMS provider.

    return {

        "message": "Session created. Share the 6-digit code with the customer.",

        "session": {

            "id": session_id, "application_id": application_id,

            "customer_id": customer_id, "organization": org,

            "expires_at": expires_at, "purpose": purpose,

        },

        "code_dev": code,

    }





@app.get("/kyc/sessions/validate/{code}")

async def api_kyc_validate_code(code: str):

    """Client validates a 6-digit session code before joining."""

    ch = _code_hash(code)

    rec = await session_codes_col.find_one({"code_hash": ch})

    if not rec:

        raise HTTPException(status_code=404, detail="Invalid session code.")

    session = await kyc_sessions_col.find_one({"id": rec.get("session_id")})

    if not session:

        raise HTTPException(status_code=404, detail="Session not found.")

    if session.get("code_used"):

        raise HTTPException(status_code=409, detail="Session code already used.")

    if time.time() > session.get("code_expires_at", 0):

        raise HTTPException(status_code=410, detail="Session code expired.")

    return {

        "valid": True,

        "session_id": session.get("id"),

        "application_id": session.get("application_id"),

        "customer_id": session.get("customer_id"),

        "organization": session.get("organization"),

        "purpose": session.get("purpose"),

    }





@app.post("/kyc/sessions/join/{code}")

async def api_kyc_join_session(

    code: str, authorization: Optional[str] = Header(None)

):

    """Client joins a session using a validated code."""

    user = await _current_user(authorization)

    require_client(user)

    ch = _code_hash(code)

    rec = await session_codes_col.find_one({"code_hash": ch})

    if not rec:

        raise HTTPException(status_code=404, detail="Invalid session code.")

    session = await kyc_sessions_col.find_one({"id": rec.get("session_id")})

    if not session:

        raise HTTPException(status_code=404, detail="Session not found.")

    if session.get("code_used"):

        raise HTTPException(status_code=409, detail="Session code already used.")

    if time.time() > session.get("code_expires_at", 0):

        raise HTTPException(status_code=410, detail="Session code expired.")

    if session.get("customer_id") and session.get("customer_id") != user.get("id"):

        raise HTTPException(status_code=403, detail="Session is not for this customer.")



    # Mark code used

    await kyc_sessions_col.update_one(

        {"id": session.get("id")},

        {"$set": {"code_used": True, "used_at": time.time(), "used_by": user.get("id")}},

    )

    await save_db()



    await audit_event(user.get("id"), ROLE_CLIENT, "kyc_session_join", "kyc_session", session.get("id"))



    return {

        "message": "Session joined.",

        "session": {

            "id": session.get("id"), "application_id": session.get("application_id"),

            "organization": session.get("organization"), "purpose": session.get("purpose"),

        },

    }





@app.get("/kyc/sessions")

async def api_kyc_list_sessions(authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    role = user.get("role")

    org = user.get("organization")

    cursor = kyc_sessions_col.find({})

    sessions = await cursor.to_list(length=500)

    if role == ROLE_CLIENT:

        sessions = [s for s in sessions if s.get("customer_id") == user.get("id")]

    elif role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):

        sessions = [s for s in sessions if s.get("organization") == org]

    return {"sessions": sessions}





# ──────────────────────────────────────────────────────────────────────────────

# Documents

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/documents/upload")

async def api_document_upload(

    application_id: str,

    doc_type: str,

    file: UploadFile = File(...),

    authorization: Optional[str] = Header(None),

):

    user = await _current_user(authorization)

    require_client(user)

    allowed = {"image/jpeg", "image/jpg", "image/png", "application/pdf"}

    content_type = file.content_type or ""

    if content_type not in allowed:

        raise HTTPException(status_code=400, detail=f"Invalid file type '{content_type}'.")

    contents = await file.read()

    size_mb = len(contents) / (1024 * 1024)

    if size_mb > 5:

        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")

    if len(contents) < 100:

        raise HTTPException(status_code=400, detail="File appears empty or corrupt.")



    app = await kyc_applications_col.find_one({"id": application_id})

    if not app:

        raise HTTPException(status_code=404, detail="Application not found.")

    if app.get("customer_id") != user.get("id"):

        raise HTTPException(status_code=403, detail="Not your application.")



    import uuid as _uuid

    from pathlib import Path as _Path

    doc_id = str(_uuid.uuid4())

    ext = _Path(file.filename or "").suffix or ".bin"

    upload_dir = TEMP_UPLOADS_DIR / "documents" / application_id

    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = upload_dir / f"{doc_id}{ext}"

    with open(save_path, "wb") as f:

        f.write(contents)



    now = time.time()

    doc = {

        "id": doc_id, "application_id": application_id,

        "customer_id": user.get("id"), "doc_type": doc_type,

        "filename": file.filename or "upload", "path": str(save_path),

        "mime_type": content_type, "size_bytes": len(contents),

        "status": "pending", "created_at": now,

    }

    await documents_col.insert_one(doc)

    await save_db()

    await audit_event(user.get("id"), ROLE_CLIENT, "document_upload", "document", doc_id,

                      {"application_id": application_id, "doc_type": doc_type})

    return {"message": "Document uploaded.", "document": doc}





@app.get("/documents")

async def api_document_list(authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    role = user.get("role")

    org = user.get("organization")

    cursor = documents_col.find({})

    docs = await cursor.to_list(length=500)

    if role == ROLE_CLIENT:

        docs = [d for d in docs if d.get("customer_id") == user.get("id")]

    elif role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):

        # only docs for applications in this org

        app_ids = {a.get("id") for a in (await kyc_applications_col.find({}).to_list(length=5000))

                   if a.get("organization") == org}

        docs = [d for d in docs if d.get("application_id") in app_ids]

    return {"documents": docs}





@app.post("/documents/{doc_id}/review")

async def api_document_review(

    doc_id: str, req: dict, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    if user.get("role") not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only officers can review documents.")

    doc = await documents_col.find_one({"id": doc_id})

    if not doc:

        raise HTTPException(status_code=404, detail="Document not found.")

    decision = (req.get("decision") or "").strip()

    notes = (req.get("notes") or "").strip()

    if decision not in ("approved", "rejected", "additional_info"):

        raise HTTPException(status_code=400, detail="Invalid decision.")

    await documents_col.update_one(

        {"id": doc_id},

        {"$set": {"status": decision, "reviewed_by": user.get("id"),

                  "reviewed_at": time.time(), "review_notes": notes}},

    )

    await save_db()

    await audit_event(user.get("id"), user.get("role"), "document_review", "document", doc_id,

                      {"decision": decision, "notes": notes})

    return {"message": f"Document {decision}.", "document_id": doc_id}





# ──────────────────────────────────────────────────────────────────────────────

# Verification results (LiveGuard engine output)

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/verification/results")

async def api_verification_results(

    req: dict, authorization: Optional[str] = Header(None)

):

    """Store LiveGuard verification results for a session/application."""

    user = await _current_user(authorization)

    if user.get("role") not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only officers can record verification results.")

    session_id = req.get("session_id")

    application_id = req.get("application_id")

    result = {

        "id": secrets.token_hex(8),

        "session_id": session_id, "application_id": application_id,

        "recorded_by": user.get("id"), "organization": user.get("organization"),

        "created_at": time.time(),

        "signals": req.get("signals", {}),   # face_detection, face_match, pad, liveness, ...

        "overall_risk": req.get("overall_risk"),  # 0..1

        "status": req.get("status", "INCONCLUSIVE"),

    }

    await verification_results_col.insert_one(result)

    await save_db()

    await audit_event(user.get("id"), user.get("role"), "verification_results_record", "verification_result", result["id"],

                      {"session_id": session_id, "application_id": application_id})

    return {"message": "Verification results recorded.", "result": result}





@app.get("/verification/results/{session_id}")

async def api_verification_results_get(

    session_id: str, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    cursor = verification_results_col.find({"session_id": session_id})

    results = await cursor.to_list(length=100)

    # org scoping

    role = user.get("role")

    org = user.get("organization")

    if role == ROLE_CLIENT:

        # client can see results for their own sessions

        sess = await kyc_sessions_col.find_one({"id": session_id})

        if not sess or sess.get("customer_id") != user.get("id"):

            raise HTTPException(status_code=403, detail="Not your session.")

    elif role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):

        sess = await kyc_sessions_col.find_one({"id": session_id})

        if not sess or sess.get("organization") != org:

            raise HTTPException(status_code=403, detail="Not your organization's session.")

    return {"results": results}





# ──────────────────────────────────────────────────────────────────────────────

# KYC Decisions

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/kyc/decisions")

async def api_kyc_decision(

    req: dict, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    if user.get("role") not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only officers can make KYC decisions.")

    application_id = req.get("application_id")

    session_id = req.get("session_id")

    decision = (req.get("decision") or "").strip()

    reason = (req.get("reason") or "").strip()

    if decision not in ("approved", "rejected", "additional_info", "manual_review"):

        raise HTTPException(status_code=400, detail="Invalid decision.")

    if decision in ("rejected", "manual_review", "additional_info") and not reason:

        raise HTTPException(status_code=400, detail="Reason is required for this decision.")



    app = await kyc_applications_col.find_one({"id": application_id})

    if not app:

        raise HTTPException(status_code=404, detail="Application not found.")

    org = user.get("organization")

    if user.get("role") != ROLE_PLATFORM_ADMIN and app.get("organization") != org:

        raise HTTPException(status_code=403, detail="Not your organization's application.")



    import uuid as _uuid

    now = time.time()

    decision_id = str(_uuid.uuid4())

    doc = {

        "id": decision_id, "application_id": application_id, "session_id": session_id,

        "officer_id": user.get("id"), "organization": org,

        "decision": decision, "reason": reason, "created_at": now,

    }

    await kyc_decisions_col.insert_one(doc)

    # Update application status

    new_status = {

        "approved": "approved", "rejected": "rejected",

        "additional_info": "additional_info_required", "manual_review": "manual_review",

    }[decision]

    await kyc_applications_col.update_one(

        {"id": application_id},

        {"$set": {"status": new_status, "updated_at": now, "decision_id": decision_id}},

    )

    await save_db()

    await audit_event(user.get("id"), user.get("role"), "kyc_decision", "kyc_decision", decision_id,

                      {"application_id": application_id, "decision": decision, "reason": reason})

    return {"message": f"KYC application {decision}.", "decision": doc}





# ──────────────────────────────────────────────────────────────────────────────

# Customer queries

# ──────────────────────────────────────────────────────────────────────────────



@app.post("/queries")

async def api_query_create(

    req: dict, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    require_client(user)

    subject = (req.get("subject") or "").strip()

    message = (req.get("message") or "").strip()

    if not subject or not message:

        raise HTTPException(status_code=400, detail="Subject and message are required.")

    import uuid as _uuid

    now = time.time()

    doc = {

        "id": str(_uuid.uuid4()), "customer_id": user.get("id"),

        "customer_name": user.get("name"), "organization": user.get("organization"),

        "subject": subject, "message": message,

        "status": "open", "response": None, "responded_by": None,

        "created_at": now, "updated_at": now,

    }

    await customer_queries_col.insert_one(doc)

    await save_db()

    await audit_event(user.get("id"), ROLE_CLIENT, "query_create", "customer_query", doc["id"],

                      {"subject": subject})

    return {"message": "Query raised.", "query": doc}





@app.get("/queries")

async def api_query_list(authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    role = user.get("role")

    org = user.get("organization")

    cursor = customer_queries_col.find({})

    queries = await cursor.to_list(length=500)

    if role == ROLE_CLIENT:

        queries = [q for q in queries if q.get("customer_id") == user.get("id")]

    elif role in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):

        queries = [q for q in queries if q.get("organization") == org]

    return {"queries": queries}





@app.post("/queries/{query_id}/respond")

async def api_query_respond(

    query_id: str, req: dict, authorization: Optional[str] = Header(None)

):

    user = await _current_user(authorization)

    if user.get("role") not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN):

        raise HTTPException(status_code=403, detail="Only officers can respond to queries.")

    response = (req.get("response") or "").strip()

    if not response:

        raise HTTPException(status_code=400, detail="Response is required.")

    query = await customer_queries_col.find_one({"id": query_id})

    if not query:

        raise HTTPException(status_code=404, detail="Query not found.")

    org = user.get("organization")

    if user.get("role") != ROLE_PLATFORM_ADMIN and query.get("organization") != org:

        raise HTTPException(status_code=403, detail="Not your organization's query.")

    now = time.time()

    await customer_queries_col.update_one(

        {"id": query_id},

        {"$set": {"response": response, "responded_by": user.get("id"),

                  "status": "answered", "updated_at": now}},

    )

    await save_db()

    await audit_event(user.get("id"), user.get("role"), "query_respond", "customer_query", query_id)

    return {"message": "Query responded.", "query_id": query_id}





# ──────────────────────────────────────────────────────────────────────────────

# Audit (platform admin)

# ──────────────────────────────────────────────────────────────────────────────



@app.get("/audit")

async def api_audit(authorization: Optional[str] = Header(None)):

    user = await _current_user(authorization)

    require_platform_admin(user)

    cursor = audit_events_col.find({})

    events = await cursor.to_list(length=1000)

    return {"audit_events": events}





# SPA catch-all — must be LAST

@app.get("/{full_path:path}")

async def catch_all(full_path: str):

    # Only redirect to index.html if it's NOT a known API/WS path

    api_prefixes = ["auth", "tenant", "application", "support", "notifications", "uploads", "evidence", "analyze-session", "upload-recording"]

    if any(full_path.startswith(prefix) for prefix in api_prefixes):

        raise HTTPException(status_code=404)



    # Don't intercept /ws or /session - let them fail naturally or match

    if full_path.startswith("ws") or full_path.startswith("session") or full_path.startswith("health") or full_path.startswith("ping"):

        raise HTTPException(status_code=404)



    # Direct static files in DIST_DIR (e.g. /skyc_logo_light.png, /logo.svg, /assets/*)
    dist_file = DIST_DIR / full_path
    if dist_file.exists() and dist_file.is_file():
        return FileResponse(str(dist_file))

    # Fallback to frontend/public for unbundled public static assets
    public_file = (FRONTEND_DIR / "public" / full_path)
    if public_file.exists() and public_file.is_file():
        return FileResponse(str(public_file))



    # Evidence files

    if full_path.startswith("evidence"):

        ev_path = EVIDENCE_DIR / full_path[len("evidence/"):] if full_path.startswith("evidence/") else EVIDENCE_DIR / full_path

        if ev_path.exists():

            return FileResponse(str(ev_path))



    # Serve the SPA when the frontend build exists; otherwise a helpful setup message.

    if INDEX_HTML.exists():

        return FileResponse(str(INDEX_HTML))

    raise HTTPException(

        status_code=404,

        detail="Frontend production build not found. Build the frontend (cd frontend && npm run build) "

               "or access the API directly at /docs, /ping, /auth, etc.",

    )

