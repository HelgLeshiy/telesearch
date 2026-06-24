"""Presigned local upload endpoint (token-authorized raw PUT).

Used by the local blob backend's presign flow: the client PUTs file bytes here
with the signed token returned by ``/sources/presign``. The token authorizes
writing to one specific blob key, so no session header is required (this is the
presigned-URL pattern). S3 backends presign directly to the bucket instead.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..blobs import LocalBlobStore, get_blob_store
from ..config import ServerSettings, get_server_settings

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.put("/{token}")
async def put_upload(
    token: str,
    request: Request,
    settings: ServerSettings = Depends(get_server_settings),
) -> dict:
    store = get_blob_store(settings)
    if not isinstance(store, LocalBlobStore):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "presigned PUT not handled here")
    try:
        blob_key, filename = store.verify_upload_token(token)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    data = await request.body()
    if settings.max_upload_bytes and len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "upload exceeds limit")
    written = store.save_bytes(blob_key, filename, data)
    return {"ok": True, "bytes": written}
