"""
Day 4 — Case-record database layer, behind one interface (mirrors storage_backend.py).

A "case" is one uploaded image plus everything the triage queue needs:

    case_id, image_ref, image_filename,
    predicted_class, predicted_label, confidence, class_probs (json),
    risk_tier, priority_score,
    status ("pending" | "reviewed" | "dismissed"),
    reviewer_note, override_class,
    created_at, reviewed_at

SQLiteDBBackend is the default — no account needed, and SQLite is a fine
stand-in for "a database in the cloud" for a project of this size; the
interface is what would let you point this at Firestore or DynamoDB in a
real deployment without touching triage/queue_manager.py or the dashboard.
"""
import abc
import datetime
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import LOCAL_DB_PATH  # noqa: E402


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class DBBackend(abc.ABC):
    @abc.abstractmethod
    def create_case(self, case: dict) -> str:
        """Inserts a case dict (must include image_ref, predicted_class,
        confidence, risk_tier, priority_score). Returns the new case_id."""

    @abc.abstractmethod
    def get_case(self, case_id: str) -> Optional[dict]:
        ...

    @abc.abstractmethod
    def update_case(self, case_id: str, updates: dict) -> None:
        ...

    @abc.abstractmethod
    def list_cases(self, status: Optional[str] = None) -> list:
        """Returns all cases (optionally filtered by status), unsorted —
        triage/queue_manager.py owns sort order, not the DB layer."""


class SQLiteDBBackend(DBBackend):
    def __init__(self, db_path: Path = LOCAL_DB_PATH):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    image_ref TEXT,
                    image_filename TEXT,
                    predicted_class TEXT,
                    predicted_label TEXT,
                    confidence REAL,
                    class_probs TEXT,
                    risk_tier TEXT,
                    priority_score REAL,
                    status TEXT DEFAULT 'pending',
                    reviewer_note TEXT,
                    override_class TEXT,
                    created_at TEXT,
                    reviewed_at TEXT
                )
                """
            )

    def create_case(self, case: dict) -> str:
        case_id = case.get("case_id") or uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases
                    (case_id, image_ref, image_filename, predicted_class, predicted_label,
                     confidence, class_probs, risk_tier, priority_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    case_id,
                    case["image_ref"],
                    case.get("image_filename", ""),
                    case["predicted_class"],
                    case.get("predicted_label", case["predicted_class"]),
                    case["confidence"],
                    json.dumps(case.get("class_probs", {})),
                    case["risk_tier"],
                    case["priority_score"],
                    now_iso(),
                ),
            )
        return case_id

    def get_case(self, case_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def update_case(self, case_id: str, updates: dict) -> None:
        if not updates:
            return
        columns = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [case_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE cases SET {columns} WHERE case_id = ?", values)

    def list_cases(self, status: Optional[str] = None) -> list:
        with self._connect() as conn:
            if status:
                rows = conn.execute("SELECT * FROM cases WHERE status = ?", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM cases").fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row) -> dict:
        d = dict(row)
        d["class_probs"] = json.loads(d["class_probs"]) if d.get("class_probs") else {}
        return d


class FirestoreDBBackend(DBBackend):
    """
    Real cloud backend. Requires:
        pip install firebase-admin --break-system-packages
        FIREBASE_CREDENTIALS_JSON env var -> path to a service account key
    Uses the same Firebase app initialized by FirebaseStorageBackend if both
    are active; initializes its own if used standalone.
    """

    COLLECTION = "triage_cases"

    def __init__(self):
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = credentials.Certificate(os.environ["FIREBASE_CREDENTIALS_JSON"])
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def create_case(self, case: dict) -> str:
        case_id = case.get("case_id") or uuid.uuid4().hex
        doc = {**case, "case_id": case_id, "status": "pending", "created_at": now_iso()}
        self.db.collection(self.COLLECTION).document(case_id).set(doc)
        return case_id

    def get_case(self, case_id: str) -> Optional[dict]:
        doc = self.db.collection(self.COLLECTION).document(case_id).get()
        return doc.to_dict() if doc.exists else None

    def update_case(self, case_id: str, updates: dict) -> None:
        self.db.collection(self.COLLECTION).document(case_id).update(updates)

    def list_cases(self, status: Optional[str] = None) -> list:
        from firebase_admin import firestore

        query = self.db.collection(self.COLLECTION)
        if status:
            # Keyword `filter=FieldFilter(...)`; the positional .where("f","==",v)
            # form is deprecated in google-cloud-firestore and warns/breaks.
            query = query.where(filter=firestore.FieldFilter("status", "==", status))
        return [doc.to_dict() for doc in query.stream()]


class DynamoDBBackend(DBBackend):
    """
    Real cloud backend. Requires:
        pip install boto3 --break-system-packages
        DYNAMODB_TABLE_NAME env var -> table with partition key "case_id" (String)
        AWS credentials via the standard boto3 chain
    """

    def __init__(self):
        import boto3

        self.table_name = os.environ["DYNAMODB_TABLE_NAME"]
        self.table = boto3.resource("dynamodb").Table(self.table_name)

    def create_case(self, case: dict) -> str:
        case_id = case.get("case_id") or uuid.uuid4().hex
        item = {**case, "case_id": case_id, "status": "pending", "created_at": now_iso()}
        item["class_probs"] = json.dumps(item.get("class_probs", {}))
        self.table.put_item(Item=item)
        return case_id

    def get_case(self, case_id: str) -> Optional[dict]:
        resp = self.table.get_item(Key={"case_id": case_id})
        item = resp.get("Item")
        if item and "class_probs" in item:
            item["class_probs"] = json.loads(item["class_probs"])
        return item

    def update_case(self, case_id: str, updates: dict) -> None:
        expr = "SET " + ", ".join(f"#{k} = :{k}" for k in updates)
        names = {f"#{k}": k for k in updates}
        values = {f":{k}": v for k, v in updates.items()}
        self.table.update_item(
            Key={"case_id": case_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def list_cases(self, status: Optional[str] = None) -> list:
        items = self.table.scan().get("Items", [])
        for item in items:
            if "class_probs" in item and isinstance(item["class_probs"], str):
                item["class_probs"] = json.loads(item["class_probs"])
        if status:
            items = [i for i in items if i.get("status") == status]
        return items


def get_db_backend() -> DBBackend:
    backend = os.environ.get("DB_BACKEND", "local").lower()
    if backend == "local":
        return SQLiteDBBackend()
    if backend == "firestore":
        return FirestoreDBBackend()
    if backend == "dynamodb":
        return DynamoDBBackend()
    raise ValueError(f"Unknown DB_BACKEND: {backend!r}. Use local, firestore, or dynamodb.")
