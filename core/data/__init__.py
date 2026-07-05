"""WIM-Z spec data store (WIMZ_Data_Architecture_Spec.md).

Runs in parallel with the legacy core/store.py during the transition —
producers dual-write. This package owns /data/wimz.db and /data/media.
"""
from core.data.ids import uuid7, uuid7_at
from core.data.wimz_store import WimzStore, get_wimz_store, DATA_ROOT, MEDIA_DIR

__all__ = ['uuid7', 'uuid7_at', 'WimzStore', 'get_wimz_store', 'DATA_ROOT', 'MEDIA_DIR']
