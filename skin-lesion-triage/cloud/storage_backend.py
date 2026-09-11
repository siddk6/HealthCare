"""
Day 4 — Cloud storage layer, behind one interface.

Why an abstract interface instead of just calling Firebase directly: the
brief lists Firebase Storage OR AWS S3 as options, and in practice you don't
want the dashboard/pipeline code to care which one a given deployment uses.
`get_storage_backend()` reads an env var and returns whichever concrete
class implements StorageBackend — swapping providers is a config change,
not a code change.

LocalStorageBackend is the default and needs no account/credentials at all —
it's what makes this project runnable and demoable immediately. Point
STORAGE_BACKEND=firebase or STORAGE_BACKEND=s3 at real credentials to go
fully cloud for a real deployment.
"""
import abc
import os
import shutil
import uuid
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import LOCAL_STORAGE_DIR  # noqa: E402


class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def upload(self, local_file_path: str, dest_name: str = None) -> str:
        """Uploads a file, returns a stable reference (URL or path) to it."""

    @abc.abstractmethod
    def download(self, ref: str, local_dest_path: str) -> str:
        """Downloads by reference back to a local path, returns that path."""

    @abc.abstractmethod
    def get_url(self, ref: str) -> str:
        """Returns something the dashboard can render/link to."""


class LocalStorageBackend(StorageBackend):
    """Default backend — a folder on disk standing in for a cloud bucket.
    Zero setup, so `streamlit run dashboard/app.py` works out of the box."""

    def __init__(self, root: Path = LOCAL_STORAGE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def upload(self, local_file_path: str, dest_name: str = None) -> str:
        dest_name = dest_name or f"{uuid.uuid4().hex}{Path(local_file_path).suffix}"
        dest_path = self.root / dest_name
        shutil.copy(local_file_path, dest_path)
        return str(dest_path)  # the "ref" is just the path for this backend

    def download(self, ref: str, local_dest_path: str) -> str:
        shutil.copy(ref, local_dest_path)
        return local_dest_path

    def get_url(self, ref: str) -> str:
        return ref  # Streamlit can open a local path directly


class FirebaseStorageBackend(StorageBackend):
    """
    Real cloud backend. Requires:
        pip install firebase-admin --break-system-packages
        FIREBASE_CREDENTIALS_JSON env var -> path to a service account key
        FIREBASE_STORAGE_BUCKET env var   -> e.g. "your-project.appspot.com"
    """

    def __init__(self):
        import firebase_admin
        from firebase_admin import credentials, storage

        if not firebase_admin._apps:
            cred = credentials.Certificate(os.environ["FIREBASE_CREDENTIALS_JSON"])
            firebase_admin.initialize_app(
                cred, {"storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"]}
            )
        self.bucket = storage.bucket()

    def upload(self, local_file_path: str, dest_name: str = None) -> str:
        dest_name = dest_name or f"lesion_images/{uuid.uuid4().hex}{Path(local_file_path).suffix}"
        blob = self.bucket.blob(dest_name)
        blob.upload_from_filename(local_file_path)
        return dest_name  # store the blob path as the ref; sign URLs on demand

    def download(self, ref: str, local_dest_path: str) -> str:
        blob = self.bucket.blob(ref)
        blob.download_to_filename(local_dest_path)
        return local_dest_path

    def get_url(self, ref: str) -> str:
        blob = self.bucket.blob(ref)
        return blob.generate_signed_url(expiration=3600)  # 1-hour signed URL


class S3StorageBackend(StorageBackend):
    """
    Real cloud backend. Requires:
        pip install boto3 --break-system-packages
        AWS credentials available via the standard boto3 chain (env vars,
        ~/.aws/credentials, or an instance role)
        S3_BUCKET_NAME env var
    """

    def __init__(self):
        import boto3

        self.bucket_name = os.environ["S3_BUCKET_NAME"]
        self.client = boto3.client("s3")

    def upload(self, local_file_path: str, dest_name: str = None) -> str:
        dest_name = dest_name or f"lesion_images/{uuid.uuid4().hex}{Path(local_file_path).suffix}"
        self.client.upload_file(local_file_path, self.bucket_name, dest_name)
        return dest_name

    def download(self, ref: str, local_dest_path: str) -> str:
        self.client.download_file(self.bucket_name, ref, local_dest_path)
        return local_dest_path

    def get_url(self, ref: str) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket_name, "Key": ref}, ExpiresIn=3600
        )


def get_storage_backend() -> StorageBackend:
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        return LocalStorageBackend()
    if backend == "firebase":
        return FirebaseStorageBackend()
    if backend == "s3":
        return S3StorageBackend()
    raise ValueError(f"Unknown STORAGE_BACKEND: {backend!r}. Use local, firebase, or s3.")
