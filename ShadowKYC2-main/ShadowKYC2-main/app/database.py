import time
import json
import os
import asyncio
import threading
import secrets
import uuid
from pathlib import Path

_save_lock = threading.Lock()
# Resolve the database file relative to this module (project root) so the app
# works regardless of the current working directory.
DB_FILE = str(Path(__file__).resolve().parent.parent / "db.json")

# ─── In-memory stores (MUST define before load) ───────────────────────────────
_users           = {}   # email -> user doc
_tokens          = {}   # token -> email
_notif_list      = []   # flat list of notifications
_ticket_list     = []   # flat list of tickets
_history_list    = []   # list of session history docs
_app_status_list = []   # list of app status docs
_doc_approvals   = {}   # session_id -> approval doc

def _save_db():
    """Atomically persist all in-memory stores to DB_FILE.
    Writes to a temporary file first, then atomically replaces DB_FILE so a
    crash mid-write can never leave a corrupt/corrupted database behind.
    """
    if len(_history_list) > 200:
        del _history_list[:-200]
    if len(_notif_list) > 200:
        del _notif_list[:-200]
    if len(_ticket_list) > 200:
        del _ticket_list[:-200]
    with _save_lock:
        tmp_file = DB_FILE + ".tmp"
        try:
            data = {
                "users": _users,
                "tokens": _tokens,
                "notifs": _notif_list,
                "tickets": _ticket_list,
                "doc_approvals": _doc_approvals,
                "history": _history_list,
                "app_status": _app_status_list,
            }
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            os.replace(tmp_file, DB_FILE)
        except Exception as e:
            print(f"[DB] Error saving: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass

def save_async():
    """Trigger a save. Kept for backwards compatibility; now performs an
    atomic synchronous save so writes are never silently lost."""
    _save_db()

def _load_db():
    if not os.path.exists(DB_FILE):
        return
    try:
        with open(DB_FILE, "r") as f:
            data = json.load(f)
        _users.update(data.get("users", {}))
        _tokens.update(data.get("tokens", {}))
        _notif_list.extend(data.get("notifs", []))
        _ticket_list.extend(data.get("tickets", []))
        _doc_approvals.update(data.get("doc_approvals", {}))
        raw_h = data.get("history", [])
        if len(raw_h) > 200:
            raw_h = raw_h[-200:]
        _history_list.extend(raw_h)
        _app_status_list.extend(data.get("app_status", []))
        print(f"[DB] Loaded persistence from {DB_FILE}")
    except Exception as e:
        # Never crash on a corrupt database file. Back it up and start fresh
        # so the server still comes up and existing valid data is preserved.
        corrupt = DB_FILE + ".corrupt"
        try:
            if not os.path.exists(corrupt):
                os.replace(DB_FILE, corrupt)
        except Exception:
            pass
        print(f"[DB] Error loading persistence ({e}); database backed up to {corrupt} and starting empty.")

# Call load on import
_load_db()


# ─── Fake collection API (matches motor async interface) ───────────────────────

class FakeCol:
    """A dead-simple async-compatible collection backed by a Python structure."""

    def __init__(self, store_ref, key_field: str = None):
        # store_ref is either a dict or list
        self._store = store_ref
        self._key = key_field

    async def find_one(self, query: dict, projection: dict = None):
        docs = await self._to_list()
        for doc in docs:
            if self._matches(doc, query):
                return doc
        return None

    async def insert_one(self, doc: dict):
        doc = {k: v for k, v in doc.items()}  # shallow copy
        if isinstance(self._store, list):
            self._store.append(doc)
        elif isinstance(self._store, dict) and self._key:
            key_val = doc.get(self._key)
            if key_val is not None:
                self._store[key_val] = doc
        save_async()

    async def update_one(self, query: dict, update: dict, upsert: bool = False):
        docs = await self._to_list()
        for doc in docs:
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])
                save_async()
                return
        if upsert:
            new_doc = {**{k: v for k, v in query.items() if not k.startswith("$")}}
            if "$set" in update:
                new_doc.update(update["$set"])
            await self.insert_one(new_doc)

    async def update_many(self, query: dict, update: dict):
        docs = await self._to_list()
        for doc in docs:
            if self._matches(doc, query):
                if "$set" in update:
                    doc.update(update["$set"])

    def find(self, query: dict = None, projection: dict = None):
        return _FakeCursor(self._store, query or {})

    async def create_index(self, *args, **kwargs):
        pass  # no-op

    async def _to_list(self):
        if isinstance(self._store, list):
            return list(self._store)
        elif isinstance(self._store, dict):
            return list(self._store.values())
        return []

    @staticmethod
    def _matches(doc: dict, query: dict) -> bool:
        for k, v in query.items():
            if doc.get(k) != v:
                return False
        return True


class _FakeCursor:
    def __init__(self, store, query: dict):
        self._store = store
        self._query = query
        self._sort_key = None
        self._sort_dir = 1

    def sort(self, key: str, direction: int = -1):
        self._sort_key = key
        self._sort_dir = direction
        return self

    async def to_list(self, length: int = 1000):
        if isinstance(self._store, list):
            docs = list(self._store)
        elif isinstance(self._store, dict):
            docs = list(self._store.values())
        else:
            docs = []

        results = [d for d in docs if FakeCol._matches(d, self._query)]

        if self._sort_key:
            results.sort(
                key=lambda d: d.get(self._sort_key) or 0,
                reverse=(self._sort_dir == -1)
            )

        return results[:length]


# ─── Public collection references (same names as old motor collections) ────────
users_col           = FakeCol(_users, key_field="email")
tokens_col          = FakeCol(_tokens, key_field="token")
notifications_col   = FakeCol(_notif_list)
tickets_col         = FakeCol(_ticket_list)
session_history_col = FakeCol(_history_list)
app_status_col      = FakeCol(_app_status_list)
doc_approvals_col   = FakeCol(_doc_approvals, key_field="session_id")

# ─── LiveGuard bank-orientated collections ────────────────────────────────────
_organizations       = {}   # id -> org doc
_verified_domains    = {}   # domain -> domain doc
_officer_invitations = {}   # id -> invitation doc
_customer_profiles   = {}   # user_id -> profile doc
_kyc_applications    = {}   # id -> application doc
_kyc_sessions        = {}   # id -> session doc (holds hashed 6-digit code)
_session_codes       = {}   # code_hash -> session id (for fast lookup)
_documents           = {}   # id -> document doc
_verification_results = {} # id -> verification result doc
_kyc_decisions       = {}   # id -> decision doc
_customer_queries    = {}   # id -> query doc
_audit_events        = []   # flat list of audit events

# ─── Extended persistence helpers ──────────────────────────────────────────────

def _save_db_extended():
    """Persist all stores (existing + LiveGuard) atomically to DB_FILE."""
    with _save_lock:
        tmp_file = DB_FILE + ".tmp"
        try:
            data = {
                # existing stores
                "users": _users,
                "tokens": _tokens,
                "notifs": _notif_list,
                "tickets": _ticket_list,
                "doc_approvals": _doc_approvals,
                "history": _history_list,
                "app_status": _app_status_list,
                # LiveGuard stores
                "organizations": _organizations,
                "verified_domains": _verified_domains,
                "officer_invitations": _officer_invitations,
                "customer_profiles": _customer_profiles,
                "kyc_applications": _kyc_applications,
                "kyc_sessions": _kyc_sessions,
                "session_codes": _session_codes,
                "documents": _documents,
                "verification_results": _verification_results,
                "kyc_decisions": _kyc_decisions,
                "customer_queries": _customer_queries,
                "audit_events": _audit_events,
            }
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            os.replace(tmp_file, DB_FILE)
        except Exception as e:
            print(f"[DB] Error saving extended: {e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass


async def save_db():
    """Save all stores (existing + LiveGuard) atomically. Async so callers can await."""
    _save_db_extended()


def _load_db_extended():
    if not os.path.exists(DB_FILE):
        return
    try:
        with open(DB_FILE, "r") as f:
            data = json.load(f)
        # existing stores
        _users.update(data.get("users", {}))
        _tokens.update(data.get("tokens", {}))
        _notif_list.extend(data.get("notifs", []))
        _ticket_list.extend(data.get("tickets", []))
        _doc_approvals.update(data.get("doc_approvals", {}))
        raw_h = data.get("history", [])
        if len(raw_h) > 200:
            raw_h = raw_h[-200:]
        _history_list.extend(raw_h)
        _app_status_list.extend(data.get("app_status", []))
        # LiveGuard stores
        _organizations.update(data.get("organizations", {}))
        _verified_domains.update(data.get("verified_domains", {}))
        _officer_invitations.update(data.get("officer_invitations", {}))
        _customer_profiles.update(data.get("customer_profiles", {}))
        _kyc_applications.update(data.get("kyc_applications", {}))
        _kyc_sessions.update(data.get("kyc_sessions", {}))
        _session_codes.update(data.get("session_codes", {}))
        _documents.update(data.get("documents", {}))
        _verification_results.update(data.get("verification_results", {}))
        _kyc_decisions.update(data.get("kyc_decisions", {}))
        _customer_queries.update(data.get("customer_queries", {}))
        _audit_events.extend(data.get("audit_events", []))
        print(f"[DB] Loaded extended persistence from {DB_FILE}")
    except Exception as e:
        corrupt = DB_FILE + ".corrupt"
        try:
            if not os.path.exists(corrupt):
                os.replace(DB_FILE, corrupt)
        except Exception:
            pass
        print(f"[DB] Error loading extended persistence ({e}); backed up to {corrupt} and starting empty.")


# Replace the original save/load with extended versions
_save_db = _save_db_extended
_load_db = _load_db_extended

# Re-run load so extended stores are populated on import
_load_db()


# ─── Public LiveGuard collection references ────────────────────────────────────
organizations_col       = FakeCol(_organizations, key_field="id")
verified_domains_col    = FakeCol(_verified_domains, key_field="domain")
officer_invitations_col = FakeCol(_officer_invitations, key_field="id")
customer_profiles_col   = FakeCol(_customer_profiles, key_field="user_id")
kyc_applications_col    = FakeCol(_kyc_applications, key_field="id")
kyc_sessions_col        = FakeCol(_kyc_sessions, key_field="id")
session_codes_col       = FakeCol(_session_codes, key_field="code_hash")
documents_col           = FakeCol(_documents, key_field="id")
verification_results_col = FakeCol(_verification_results, key_field="id")
kyc_decisions_col       = FakeCol(_kyc_decisions, key_field="id")
customer_queries_col    = FakeCol(_customer_queries, key_field="id")
audit_events_col        = FakeCol(_audit_events)


# ─── Audit helper ──────────────────────────────────────────────────────────────

async def audit_event(actor_id: str, actor_role: str, action: str,
                      entity_type: str, entity_id: str, details: dict = None):
    """Append an audit event. Never fails the request if logging fails."""
    event = {
        "id": secrets.token_hex(8),
        "actor_id": actor_id,
        "actor_role": actor_role,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "details": details or {},
        "created_at": time.time(),
    }
    try:
        _audit_events.append(event)
        await save_db()
    except Exception as e:
        print(f"[DB] audit_event save error: {e}")
    return event


# ─── Lifecycle (kept so main.py startup/shutdown hooks still work) ─────────────
async def connect_db():
    print("[DB] In-memory database ready (no MongoDB needed).")

async def close_db():
    print("[DB] Saving state to disk before shutdown...")
    _save_db()
    print("[DB] In-memory database cleared.")
