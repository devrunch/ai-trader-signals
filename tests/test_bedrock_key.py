import base64
import datetime
import hashlib
import hmac
from urllib.parse import parse_qsl, quote

import botocore.auth

from app.config import generate_bedrock_key

ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
REGION = "ap-south-1"
FIXED_NOW = datetime.datetime(2026, 9, 13, 9, 19, 15, tzinfo=datetime.timezone.utc)


def _decode(token: str) -> str:
    assert token.startswith("bedrock-api-key-")
    return base64.b64decode(token.removeprefix("bedrock-api-key-")).decode()


def _expected_signature(signed_params: dict[str, str]) -> str:
    """SigV4 query signing done by hand, per AWS's spec, independent of botocore."""
    canonical_query = "&".join(
        f"{quote(k, safe='-_.~')}={quote(v, safe='-_.~')}" for k, v in sorted(signed_params.items())
    )
    canonical_request = "\n".join([
        "POST", "/", canonical_query, "host:bedrock.amazonaws.com", "", "host",
        hashlib.sha256(b"").hexdigest(),
    ])
    datestamp = FIXED_NOW.strftime("%Y%m%d")
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256", FIXED_NOW.strftime("%Y%m%dT%H%M%SZ"),
        f"{datestamp}/{REGION}/bedrock/aws4_request",
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    key = sign(("AWS4" + SECRET_KEY).encode(), datestamp)
    for part in (REGION, "bedrock", "aws4_request"):
        key = sign(key, part)
    return hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()


def test_key_signature_does_not_cover_the_version_parameter(monkeypatch):
    monkeypatch.setattr(botocore.auth, "get_current_datetime", lambda: FIXED_NOW.replace(tzinfo=None))

    url = _decode(generate_bedrock_key(ACCESS_KEY, SECRET_KEY, REGION))
    host, _, query = url.partition("/?")
    params = dict(parse_qsl(query, keep_blank_values=True))

    assert host == "bedrock.amazonaws.com"
    assert url.endswith("&Version=1")
    signature = params.pop("X-Amz-Signature")
    params.pop("Version")
    assert params["Action"] == "CallWithBearerToken"
    assert params["X-Amz-Credential"] == f"{ACCESS_KEY}/20260913/{REGION}/bedrock/aws4_request"
    assert params["X-Amz-Expires"] == "43200"
    assert signature == _expected_signature(params)
