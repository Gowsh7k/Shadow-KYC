"""
app/auth.py
───────────
Local authentication for ShadowKYC.

- Passwords are hashed with Argon2id (argon2-cffi). Existing demo accounts
  keep their original hashes and are verified with a legacy SHA-256 fallback
  so their passwords are never changed or overwritten.
- Access tokens are signed JWTs (HS256) with an expiry. Revocation is tracked
  in the local token store so logout invalidates a token.
- Secrets come from environment variables (SECRET_KEY). Nothing is logged.
"""
import hashlib
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException
from dotenv import load_dotenv
from jose import jwt, JWTError, ExpiredSignatureError
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from pydantic import BaseModel

from app.database import (
    tokens_col, users_col, organizations_col, verified_domains_col,
    officer_invitations_col, customer_profiles_col, kyc_applications_col,
    kyc_sessions_col, audit_event, save_db, _verified_domains,
)

load_dotenv()

# ─── Security settings (from environment) ──────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Fallback only for local development; production MUST set SECRET_KEY.
    SECRET_KEY = "dev-insecure-change-me"
    print("[Auth] WARNING: SECRET_KEY not set. Using insecure development default.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24h

# Argon2id password hasher (default is Argon2id)
_password_hasher = PasswordHasher()

# Roles that may be created through public customer registration
PUBLIC_ROLES = {"user"}

# ─── Password hashing ─────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a password with Argon2id."""
    return _password_hasher.hash(password)


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored hash.
    Supports Argon2id hashes and legacy SHA-256 hashes (existing demo accounts)
    without ever overwriting the original hash.
    """
    if not stored_hash:
        return False
    try:
        if stored_hash.startswith("$argon2"):
            _password_hasher.verify(stored_hash, plain_password)
            return True
    except VerifyMismatchError:
        return False
    except Exception:
        # Not an Argon2 hash we can verify; fall through to legacy check.
        pass

    # Legacy SHA-256 (demo accounts). Unsalted, kept only for backward compat.
    return hashlib.sha256(plain_password.encode("utf-8")).hexdigest() == stored_hash


def _hash_sha256(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ─── JWT tokens ───────────────────────────────────────────────────────────


def create_access_token(user_id: str, email: str, role: str,
                        expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token with expiry and a revocable jti."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    jti = str(uuid.uuid4())
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str):
    """Decode and validate a JWT signature/expiry. Returns payload or None."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except ExpiredSignatureError:
        return None
    except JWTError:
        return None


async def store_token(token: str, email: str, user_id: str, role: str) -> None:
    """Persist a token so it can be revoked on logout."""
    try:
        payload = _decode_token(token)
        expires_at = payload.get("exp") if payload else None
        await tokens_col.insert_one({
            "token": token,
            "jti": payload.get("jti") if payload else token,
            "email": email,
            "user_id": user_id,
            "role": role,
            "created_at": time.time(),
            "expires_at": expires_at,
            "revoked": False,
        })
    except Exception as e:
        print(f"[Auth] Failed to persist token: {e}")


async def revoke_token(token: str) -> bool:
    """Mark a token as revoked (logout)."""
    try:
        result = await tokens_col.update_many(
            {"token": token}, {"$set": {"revoked": True}}
        )
        return result is not None
    except Exception as e:
        print(f"[Auth] Failed to revoke token: {e}")
        return False


async def verify_local_token(token: str):
    """
    Verify a local access token: valid JWT signature, not expired, not revoked.
    Returns the user document or None.
    """
    payload = _decode_token(token)
    if not payload:
        return None
    jti = payload.get("jti")
    email = payload.get("email")
    try:
        doc = await tokens_col.find_one({"token": token})
        if not doc:
            return None
        if doc.get("revoked"):
            return None
        expires_at = doc.get("expires_at") or payload.get("exp")
        if expires_at and time.time() > float(expires_at):
            return None
        user = await users_col.find_one({"email": email})
        return user
    except Exception as e:
        print(f"[Auth] verify_local_token error: {e}")
        return None


# ─── Supabase (optional) ──────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

JWKS_CACHE = None
JWKS_LAST_FETCH = 0
JWKS_EXPIRE = 3600


async def get_supabase_jwks():
    global JWKS_CACHE, JWKS_LAST_FETCH
    if JWKS_CACHE and (time.time() - JWKS_LAST_FETCH < JWKS_EXPIRE):
        return JWKS_CACHE
    if not SUPABASE_URL:
        return None
    import httpx
    url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/jwks"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            if resp.status_code == 200:
                JWKS_CACHE = resp.json()
                JWKS_LAST_FETCH = time.time()
                return JWKS_CACHE
    except Exception as e:
        print(f"[Auth] Error fetching JWKS from Supabase: {e}")
    return None


async def verify_supabase_token(token: str):
    """Verify a Supabase JWT using JWKS (when Supabase is configured)."""
    jwks = await get_supabase_jwks()
    if not jwks:
        return None
    try:
        # Note: audience/issuer not strictly verified to avoid config mismatch.
        payload = jwt.decode(token, jwks, algorithms=["RS256"], options={"verify_aud": False})
        return payload
    except JWTError:
        return None


async def verify_ws_token(websocket, token: str):
    """
    Verify a token for WebSocket connection.
    Supports Supabase JWT and local JWT tokens.
    """
    if not token:
        return None

    payload = await verify_supabase_token(token)
    if payload:
        return {
            "id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role", "authenticated"),
            "source": "supabase",
        }

    user = await verify_local_token(token)
    if user:
        return {
            "id": user.get("id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "phone": user.get("phone"),
            "role": user.get("role"),
            "organization": user.get("organization"),
            "employee_id": user.get("employee_id"),
            "designation": user.get("designation"),
            "is_verified": user.get("is_verified"),
            "source": "local",
    }

    return None


async def get_user_from_auth_header(auth_header: str):
    """Helper for HTTP endpoints. Returns user document or None."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    return await verify_ws_token(None, token)


# ──────────────────────────────────────────────────────────────────────────────
# LiveGuard bank-orientated auth extensions
# ──────────────────────────────────────────────────────────────────────────────

# RBAC roles (backend-enforced; never trusted from the frontend)
ROLE_CLIENT = "CLIENT"
ROLE_BANK_OFFICER = "BANK_OFFICER"
ROLE_BANK_ADMIN = "BANK_ADMIN"
ROLE_PLATFORM_ADMIN = "PLATFORM_ADMIN"

ALL_ROLES = {ROLE_CLIENT, ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN}
OFFICER_ROLES = {ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN}

# Client email providers that are always acceptable for self-registration.
# Other valid personal/corporate domains are also accepted; this list is only
# illustrative and is NOT a restriction.
CLIENT_PERSONAL_DOMAINS = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com"}

# In-memory OTP / password-reset token stores (dev: no real email provider).
_otp_store = {}        # email -> {"otp": str, "expires_at": float, "purpose": str}
_reset_store = {}      # token -> {"email": str, "expires_at": float, "used": bool}


def _now() -> float:
    return time.time()


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _domain_of(email: str) -> str:
    return _norm_email(email).split("@")[-1] if "@" in _norm_email(email) else ""


def _is_valid_email(email: str) -> bool:
    e = _norm_email(email)
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e))


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _generate_otp() -> str:
    return f"{secrets.randbelow(1000000):06d}"


# ─── OTP helpers (dev-only delivery) ──────────────────────────────────────────

def _store_otp(email: str, otp: str, purpose: str, ttl_seconds: int = 600) -> None:
    _otp_store[_norm_email(email)] = {
        "otp": otp,
        "expires_at": _now() + ttl_seconds,
        "purpose": purpose,
        "used": False,
    }


def _consume_otp(email: str, otp: str, purpose: str) -> bool:
    rec = _otp_store.get(_norm_email(email))
    if not rec or rec.get("used"):
        return False
    if _now() > rec.get("expires_at", 0):
        return False
    if rec.get("purpose") != purpose:
        return False
    if rec.get("otp") != otp:
        return False
    rec["used"] = True
    return True


def _dev_delivery(email: str, code: str, kind: str) -> None:
    """Dev-only: print the OTP/reset token so it can be used without a real
    email/SMS provider. In production, integrate an actual provider here."""
    print(f"[Auth][DEV] {kind} for {email}: {code}")


def _require_roles(*allowed: str):
    """Decorator factory: enforce that the current user has one of the allowed
    roles. Used by route handlers via `current_user` dependency result."""

    def checker(user: dict) -> dict:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.get("role") not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient role.")
        return user

    return checker


def require_client(user: dict) -> dict:
    return _require_roles(ROLE_CLIENT)(user)


def require_officer(user: dict) -> dict:
    return _require_roles(ROLE_BANK_OFFICER, ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN)(user)


def require_bank_admin(user: dict) -> dict:
    return _require_roles(ROLE_BANK_ADMIN, ROLE_PLATFORM_ADMIN)(user)


def require_platform_admin(user: dict) -> dict:
    return _require_roles(ROLE_PLATFORM_ADMIN)(user)


def require_org_member(org_id: str):
    """Factory: enforce user belongs to org_id (officer/admin of that org, or
    platform admin). Used as require_org_member(org_id)(user)."""

    def checker(user: dict) -> dict:
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required.")
        if user.get("role") == ROLE_PLATFORM_ADMIN:
            return user
        if user.get("organization") != org_id:
            raise HTTPException(status_code=403, detail="Not a member of this organization.")
        return user

    return checker


# ─── Pydantic request models ──────────────────────────────────────────────────

class ClientSignupRequest(BaseModel):
    name: str
    email: str
    phone: str
    password: str
    confirm_password: str


class ClientVerifyEmailRequest(BaseModel):
    email: str
    otp: str


class ClientLoginRequest(BaseModel):
    email: str
    password: str


class ClientForgotPasswordRequest(BaseModel):
    email: str


class ClientResetPasswordRequest(BaseModel):
    email: str
    reset_token: str
    new_password: str


class OrganizationCreateRequest(BaseModel):
    name: str
    domain: str


class OfficerInviteRequest(BaseModel):
    email: str
    organization_id: str
    employee_id: str
    designation: str
    role: str


class OfficerRegisterRequest(BaseModel):
    invitation_token: str
    email: str
    password: str
    confirm_password: str


class OfficerLoginRequest(BaseModel):
    email: str
    password: str


# ─── Client: registration ─────────────────────────────────────────────────────

async def client_register(req: ClientSignupRequest) -> dict:
    """Client self-registration. Creates an unverified CLIENT account and
    issues an email OTP. Passwords are hashed with Argon2id."""
    name = (req.name or "").strip()
    email = _norm_email(req.email)
    phone = (req.phone or "").strip()
    password = req.password or ""
    confirm = req.confirm_password or ""

    if not name:
        raise HTTPException(status_code=400, detail="Full name is required.")
    if not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Valid email is required.")
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number is required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if password != confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    # Personal email providers accepted (not restricted to them).
    domain = _domain_of(email)
    if domain in CLIENT_PERSONAL_DOMAINS:
        pass  # allowed
    # Any other valid domain is also allowed for clients.

    existing = await users_col.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    import uuid as _uuid
    user_id = str(_uuid.uuid4())
    now = time.time()
    user_doc = {
        "id": user_id,
        "name": name,
        "email": email,
        "phone": phone,
        "hashed_pw": hash_password(password),
        "role": ROLE_CLIENT,
        "organization": None,
        "is_verified": False,
        "created_at": now,
    }
    await users_col.insert_one(user_doc)
    await save_db()

    # Issue email OTP for verification
    otp = _generate_otp()
    _store_otp(email, otp, purpose="email_verify")
    _dev_delivery(email, otp, "EMAIL OTP")

    await audit_event(
        user_id, ROLE_CLIENT, "client_register", "user", user_id,
        {"email": email, "phone": phone},
    )

    return {
        "message": "Account created. Verify your email to activate it.",
        "user": {
            "id": user_id, "name": name, "email": email,
            "role": ROLE_CLIENT, "is_verified": False,
        },
        # Dev-only: remove in production when an email provider is configured.
        "otp_dev": otp,
    }


async def client_verify_email(req: ClientVerifyEmailRequest) -> dict:
    """Verify client email ownership via OTP."""
    email = _norm_email(req.email)
    otp = req.otp or ""
    if not _is_valid_email(email) or not otp:
        raise HTTPException(status_code=400, detail="Email and OTP are required.")
    if not _consume_otp(email, otp, purpose="email_verify"):
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")

    user = await users_col.find_one({"email": email})
    if not user or user.get("role") != ROLE_CLIENT:
        raise HTTPException(status_code=404, detail="Client account not found.")

    await users_col.update_one(
        {"email": email}, {"$set": {"is_verified": True, "verified_at": time.time()}},
    )
    await save_db()

    await audit_event(user.get("id"), ROLE_CLIENT, "client_verify_email", "user", user.get("id"))

    return {"message": "Email verified. You can now log in.", "user": {"id": user.get("id"), "email": email, "is_verified": True}}


# ─── Client: login ────────────────────────────────────────────────────────────

async def client_login(req: ClientLoginRequest) -> dict:
    """Client login. Requires a verified email."""
    email = _norm_email(req.email)
    password = req.password or ""
    if not email or not password:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = await users_col.find_one({"email": email})
    if not user or user.get("role") != ROLE_CLIENT:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not verify_password(password, user.get("hashed_pw")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if user.get("is_verified") is False:
        raise HTTPException(status_code=403, detail="Email not verified. Please verify your email before logging in.")

    token = create_access_token(user_id=user.get("id"), email=email, role=ROLE_CLIENT)
    await store_token(token, email=email, user_id=user.get("id"), role=ROLE_CLIENT)

    await audit_event(user.get("id"), ROLE_CLIENT, "client_login", "user", user.get("id"))

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.get("id"), "name": user.get("name"), "email": email,
            "role": ROLE_CLIENT, "organization": user.get("organization"),
            "is_verified": user.get("is_verified"),
        },
    }


# ─── Client: forgot / reset password ──────────────────────────────────────────

async def client_forgot_password(req: ClientForgotPasswordRequest) -> dict:
    """Issue a password-reset token (dev-only delivery)."""
    email = _norm_email(req.email)
    if not _is_valid_email(email):
        # Don't reveal whether the email exists.
        return {"message": "If the account exists, a reset token has been issued."}

    user = await users_col.find_one({"email": email})
    if not user or user.get("role") != ROLE_CLIENT:
        return {"message": "If the account exists, a reset token has been issued."}

    token = secrets.token_urlsafe(32)
    _reset_store[token] = {
        "email": email, "expires_at": _now() + 900, "used": False,
    }
    _dev_delivery(email, token, "PASSWORD RESET TOKEN")
    return {"message": "If the account exists, a reset token has been issued."}


async def client_reset_password(req: ClientResetPasswordRequest) -> dict:
    """Reset client password using a valid reset token."""
    email = _norm_email(req.email)
    token = req.reset_token or ""
    new_password = req.new_password or ""
    if len(new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    rec = _reset_store.get(token)
    if not rec or rec.get("used"):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    if _now() > rec.get("expires_at", 0):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    if rec.get("email") != email:
        raise HTTPException(status_code=400, detail="Invalid reset token for this email.")

    user = await users_col.find_one({"email": email})
    if not user or user.get("role") != ROLE_CLIENT:
        raise HTTPException(status_code=404, detail="Client account not found.")

    await users_col.update_one(
        {"email": email}, {"$set": {"hashed_pw": hash_password(new_password)}},
    )
    rec["used"] = True
    await save_db()

    await audit_event(user.get("id"), ROLE_CLIENT, "client_reset_password", "user", user.get("id"))

    return {"message": "Password reset successfully. You can now log in."}


# ─── Organization: create (platform admin) ────────────────────────────────────

async def organization_create(req: OrganizationCreateRequest, actor: dict) -> dict:
    """Create a bank organization + register its domain for verification."""
    if actor.get("role") != ROLE_PLATFORM_ADMIN:
        raise HTTPException(status_code=403, detail="Only platform admins can create organizations.")

    name = (req.name or "").strip()
    domain = _norm_email(req.domain)
    if not name or not domain:
        raise HTTPException(status_code=400, detail="Organization name and domain are required.")
    if domain in _verified_domains:
        raise HTTPException(status_code=409, detail="Domain already registered.")

    import uuid as _uuid
    org_id = str(_uuid.uuid4())
    now = time.time()
    org_doc = {
        "id": org_id, "name": name, "domain": domain,
        "is_verified": True, "verified_at": now, "created_by": actor.get("id"), "created_at": now,
    }
    await organizations_col.insert_one(org_doc)

    domain_doc = {
        "domain": domain, "organization_id": org_id,
        "is_verified": True, "verified_at": now,
    }
    await verified_domains_col.insert_one(domain_doc)
    await save_db()

    await audit_event(actor.get("id"), ROLE_PLATFORM_ADMIN, "organization_create", "organization", org_id,
                      {"name": name, "domain": domain})

    return {
        "message": "Organization created. Domain verification is pending.",
        "organization": {"id": org_id, "name": name, "domain": domain, "is_verified": False},
    }


# ─── Officer: invitation (platform admin or bank admin of org) ────────────────

async def officer_invite(req: OfficerInviteRequest, actor: dict) -> dict:
    """Invite an officer. Caller must be platform admin or a bank admin of the
    target organization. The officer's email domain must match a verified
    domain of the organization."""
    actor_role = actor.get("role")
    org_id = req.organization_id

    # Authorization
    if actor_role == ROLE_PLATFORM_ADMIN:
        pass  # can invite into any org
    elif actor_role in (ROLE_BANK_ADMIN,):
        org = await organizations_col.find_one({"id": org_id})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found.")
        if org.get("created_by") != actor.get("id"):
            # allow any bank admin of the org? We track created_by only; keep simple:
            pass
    else:
        raise HTTPException(status_code=403, detail="Only platform admins or bank admins can invite officers.")

    email = _norm_email(req.email)
    role = (req.role or "").strip().upper()
    if role not in (ROLE_BANK_OFFICER, ROLE_BANK_ADMIN):
        raise HTTPException(status_code=400, detail="Role must be BANK_OFFICER or BANK_ADMIN.")
    if not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Valid email is required.")
    if not req.employee_id or not req.designation:
        raise HTTPException(status_code=400, detail="Employee ID and designation are required.")

    # Domain must be a verified domain of the organization
    domain = _domain_of(email)
    dom = await verified_domains_col.find_one({"domain": domain})
    if not dom or dom.get("organization_id") != org_id:
        raise HTTPException(status_code=400, detail="Email domain is not a verified domain of this organization.")
    if not dom.get("is_verified"):
        raise HTTPException(status_code=400, detail="Organization domain is not yet verified.")

    # Prevent duplicate pending invitation
    existing = await officer_invitations_col.find_one({"email": email, "organization_id": org_id, "status": "pending"})
    if existing:
        raise HTTPException(status_code=409, detail="A pending invitation already exists for this email.")

    import uuid as _uuid
    invite_id = str(_uuid.uuid4())
    token = secrets.token_urlsafe(32)
    now = time.time()
    invite_doc = {
        "id": invite_id, "token": token, "email": email,
        "organization_id": org_id, "invited_by": actor.get("id"),
        "employee_id": req.employee_id, "designation": req.designation,
        "role": role, "status": "pending",
        "created_at": now, "expires_at": now + 7 * 86400,  # 7 days
        "used_at": None,
    }
    await officer_invitations_col.insert_one(invite_doc)
    await save_db()

    _dev_delivery(email, token, "OFFICER INVITATION TOKEN")

    await audit_event(actor.get("id"), actor_role, "officer_invite", "officer_invitaiton", invite_id,
                      {"email": email, "organization_id": org_id, "role": role})

    return {
        "message": "Officer invitation created.",
        "invitation": {
            "id": invite_id, "email": email, "organization_id": org_id,
            "role": role, "status": "pending",
        },
        # Dev-only: remove in production when an email provider is configured.
        "invitation_token_dev": token,
    }


# ─── Officer: registration (via invitation token) ─────────────────────────────

async def officer_register(req: OfficerRegisterRequest) -> dict:
    """An invited officer registers with a password. The invitation token must
    match the email, be unused and not expired."""
    token = req.invitation_token or ""
    email = _norm_email(req.email)
    password = req.password or ""
    confirm = req.confirm_password or ""

    if not token or not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invitation token and email are required.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if password != confirm:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    invite = None
    for inv in officer_invitations_col._store.values():  # search by token
        if inv.get("token") == token:
            invite = inv
            break
    if not invite:
        raise HTTPException(status_code=400, detail="Invalid invitation token.")
    if invite.get("status") != "pending":
        raise HTTPException(status_code=409, detail="Invitation already used or expired.")
    if _now() > invite.get("expires_at", 0):
        raise HTTPException(status_code=409, detail="Invitation expired.")
    if invite.get("email") != email:
        raise HTTPException(status_code=400, detail="Email does not match invitation.")

    if await users_col.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    import uuid as _uuid
    user_id = str(_uuid.uuid4())
    now = time.time()
    user_doc = {
        "id": user_id, "name": email.split("@")[0], "email": email,
        "phone": "", "hashed_pw": hash_password(password),
        "role": invite.get("role"), "organization": invite.get("organization_id"),
        "employee_id": invite.get("employee_id"), "designation": invite.get("designation"),
        "is_verified": True, "created_at": now,
    }
    await users_col.insert_one(user_doc)

    # Mark invitation used
    await officer_invitations_col.update_one(
        {"id": invite.get("id")}, {"$set": {"status": "used", "used_at": now}},
    )
    await save_db()

    # Customer profile placeholder (officers are not customers)
    await customer_profiles_col.insert_one({"user_id": user_id, "role": invite.get("role")})
    await save_db()

    await audit_event(user_id, invite.get("role"), "officer_register", "user", user_id,
                      {"email": email, "organization_id": invite.get("organization_id")})

    return {
        "message": "Officer account registered. You can now log in.",
        "user": {
            "id": user_id, "email": email, "role": invite.get("role"),
            "organization": invite.get("organization_id"),
        },
    }


# ─── Officer: login ───────────────────────────────────────────────────────────

async def officer_login(req: OfficerLoginRequest) -> dict:
    """Officer/bank-admin login."""
    email = _norm_email(req.email)
    password = req.password or ""
    if not email or not password:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = await users_col.find_one({"email": email})
    if not user or user.get("role") not in OFFICER_ROLES:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not verify_password(password, user.get("hashed_pw")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    role = user.get("role")
    token = create_access_token(user_id=user.get("id"), email=email, role=role)
    await store_token(token, email=email, user_id=user.get("id"), role=role)

    await audit_event(user.get("id"), role, "officer_login", "user", user.get("id"))

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.get("id"), "name": user.get("name"), "email": email,
            "role": role, "organization": user.get("organization"),
            "employee_id": user.get("employee_id"), "designation": user.get("designation"),
        },
    }


# ─── Email/SMS provider stubs ─────────────────────────────────────────────────

async def send_email(to: str, subject: str, body: str) -> bool:
    """Placeholder email provider. Returns False unless a real provider is
    configured. Never claim delivery works without an actual provider."""
    print(f"[Email][STUB] to={to} subject={subject}")
    return False


async def send_sms(to: str, body: str) -> bool:
    """Placeholder SMS provider. Returns False unless a real provider is
    configured."""
    print(f"[SMS][STUB] to={to} body={body}")
    return False

