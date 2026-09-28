"""The phoenixd funding-source backend (https://github.com/ACINQ/phoenixd).

A fourth way to fund this mint besides lnd, cln and spark: ACINQ's
headless Phoenix, driven over its HTTP API (checked against phoenixd
v0.9.1, src/commonMain/kotlin/fr/acinq/phoenixd/Api.kt and
json/JsonSerializers.kt). Channels and inbound liquidity are ACINQ's
LSP's business, not the operator's. The node.py contract maps onto five
endpoints:

- create_invoice      -> POST /createinvoice
- pay_invoice         -> POST /payinvoice (answers once the payment ends)
- is_invoice_settled / invoice_preimage
                      -> GET /payments/incoming/{paymentHash}
- is_payment_complete / payment_preimage
                      -> GET /payments/outgoingbyhash/{paymentHash}
- fetch_node_info     -> GET /getinfo (plus a password-scope probe)
- sign_message        -> signed locally (LUD-25 digest, configured key)

Authentication is HTTP Basic with an empty username. phoenix.conf holds
two passwords, and this backend needs the full-access `http-password`:
/payinvoice refuses `http-password-limited-access`. That password can
spend the node's whole balance, so the url must be loopback (phoenixd's
own default is http://127.0.0.1:9740) unless the operator both opts in
(FUNDINGSOURCE_PHOENIXD_ALLOW_REMOTE) and uses https - see checked_url.
It is never logged, never taken from the url itself, and requests ignore
proxy and netrc settings from the environment (trust_env=False), so it
only ever travels to the configured phoenixd.

Deviations from the lnd/cln backends worth knowing:

**create_invoice returns a None preimage**, like spark: phoenixd picks a
random preimage per invoice itself, so the router reads the payment hash
off the returned invoice (cross-checked here against phoenixd's own
paymentHash, amount and description hash).

**Whole sats on the receive side.** /createinvoice takes amountSat, so a
fractional-sat amount is rejected with a clear error rather than rounded,
and config.py refuses to start with sendable bounds that are not whole
sats. Melts are paid at the invoice's own msat amount: /payinvoice reads
it off the invoice.

**Settled means the money arrived.** phoenixd's isPaid only says the
payment completed. Its receivedSat is what was credited, after any
liquidity fee phoenixd took out of the payment on the fly, so an invoice
counts as settled only once isPaid AND receivedSat covers the amount the
invoice asked for (read off the invoice) - a short-paid invoice never
mints a note worth more than arrived. phoenixd also reports an incoming
payment's preimage from the moment the invoice exists; invoice_preimage
withholds it until settlement, the same rule lnd's LookupInvoice echo
needs.

**The melt fee is phoenixd's fixed rule, not a cap we pass.** /payinvoice
takes no fee limit: every payment goes through ACINQ's trampoline, which
charges 4 sat + 0.4% of the amount (see TRAMPOLINE_FEE_BASE_MSAT). So
pay_invoice rejects a melt whose fee under that rule exceeds the router's
budget BEFORE anything is sent (phoenixd then has no record of it and the
note restores cleanly), and warns if the fee phoenixd reports afterwards
is over budget anyway (the rule changed under us - update the constants).
config.py refuses a mint fee that does not cover the rule, since the
router sizes every melt's budget from the mint fee.

**Absence is "never sent".** lightning-kmp, phoenixd's Lightning engine,
stores an outgoing payment as pending before its HTLC leaves the node, as
lnd and cln do, so phoenixd having no record of a payment hash means
nothing was ever sent for it: is_payment_complete answers False. A
recorded payment without completedAt is still in flight - notably one a
payee holds open with a hodl invoice - and raises, never False (see
node.is_payment_complete). phoenixd itself also refuses to pay a hash it
already paid or is paying, but the router never asks it to: a melt's
invoice is never paid twice (see router.get_withdraw_callback).

**Unknown lookups answer 404.** A payment lookup that finds nothing makes
phoenixd respond 204 No Content, which its own StatusPages plugin turns
into 404 "Not found" (Api.kt: "for backward compatibility"). Both are
read as "no such payment"; a 404 with any other body (notably "Unknown
endpoint (check api doc)" - a wrong url, or a phoenixd without the route)
is an error, never an answer.

**LUD-25 signs with a configured key.** phoenixd has no signmessage, so
the operator provides a dedicated secp256k1 key
(FUNDINGSOURCE_PHOENIXD_SIGNING_KEY) and notes are signed locally over the
spec digest, exactly like spark's seed-derived key. Wallets pin the
mintPubkey it yields and refuse a mint whose key changed, so it must never
change - back it up with the database, not just phoenixd's seed.

**Every invoice carries externalId "lnurlcash:<mintPubkey>"**, so a
phoenixd shared with other applications (which prefix their own ids) can
still tell this mint's receipts apart
(GET /payments/incoming?externalId=...&all=true).
"""

import ipaddress
import json
import logging
import re
from hashlib import sha256
from typing import Any, Callable
from urllib.parse import urlparse

import bolt11
import httpx
from coincurve import PrivateKey

from .node import LightningBackendConfig, NodeInfo, PaymentFailed, PaymentResult
from .signing import lightning_signed_message_digest

# ACINQ's trampoline fee on every outgoing payment - phoenixd v0.9.1's
# conf/Lsp.kt: TrampolineFees(feeBase = 4.sat, feeProportional = 4_000), a
# single level (no retries at a higher fee), computed by lightning-kmp as
# feeBase + amount * feeProportional / 1_000_000, rounded down
TRAMPOLINE_FEE_BASE_MSAT = 4_000
TRAMPOLINE_FEE_PPM = 4_000

EXTERNAL_ID_PREFIX = "lnurlcash:"

# phoenixd refuses a longer invoice description (Api.kt, createinvoice)
_MAX_DESCRIPTION_LENGTH = 128
# every phoenixd answer is a small JSON document - anything bigger is not one
_MAX_RESPONSE_BYTES = 1 << 20
_TIMEOUT_SECONDS = 15.0
# /payinvoice only answers once the payment succeeded or failed; a healthy
# trampoline payment takes seconds. Running out is no verdict either way:
# the router then confirms via is_payment_complete (see node.PaymentFailed)
_PAY_TIMEOUT_SECONDS = 90.0

_HEX32_RE = re.compile(r"[0-9a-fA-F]{64}")

# lightning-kmp channel states (ChannelState.stateName) of a channel that
# no longer counts as one
_GONE_CHANNEL_STATES = {"Closing", "Closed", "Aborted"}


def trampoline_fee_msat(amount_msat: int) -> int:
    """What phoenixd pays ACINQ's trampoline to deliver `amount_msat` - its
    whole routing fee (see TRAMPOLINE_FEE_BASE_MSAT)."""
    return TRAMPOLINE_FEE_BASE_MSAT + amount_msat * TRAMPOLINE_FEE_PPM // 1_000_000


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def checked_url(url: str, allow_remote: bool) -> str:
    """`url` without its trailing slash, once it is safe to send phoenixd's
    full-access password to: http(s), no credentials/query/fragment of its
    own, and a loopback host - unless `allow_remote` is set AND the url is
    https, since that password can spend the node's whole balance and plain
    http hands it to anyone on the path. Raises ValueError otherwise, never
    echoing the url itself (it may hold a secret someone put there)."""
    value = url.strip()
    parsed = urlparse(value)
    try:
        parsed.port  # a malformed port only raises once read
    except ValueError as exc:
        raise ValueError("FUNDINGSOURCE_PHOENIXD_URL has a malformed port.") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("FUNDINGSOURCE_PHOENIXD_URL must be an http:// or https:// URL, e.g. http://127.0.0.1:9740.")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError(
            "FUNDINGSOURCE_PHOENIXD_URL must carry no credentials, query or fragment - "
            "the password belongs in FUNDINGSOURCE_PHOENIXD_PASSWORD."
        )
    if not _is_loopback(parsed.hostname) and not (allow_remote and parsed.scheme == "https"):
        raise ValueError(
            f"FUNDINGSOURCE_PHOENIXD_URL names {parsed.hostname}, not a loopback address, so phoenixd's "
            "full-access password would cross the network: keep phoenixd on 127.0.0.1, or serve it over "
            "https and set FUNDINGSOURCE_PHOENIXD_ALLOW_REMOTE=true."
        )
    return value.rstrip("/")


def parse_signing_key(secret_hex: str) -> PrivateKey:
    """FUNDINGSOURCE_PHOENIXD_SIGNING_KEY as a key - 32 bytes of hex, and a
    valid secp256k1 secret (non-zero, below the group order)."""
    secret = secret_hex.strip()
    if not _HEX32_RE.fullmatch(secret):
        raise ValueError("FUNDINGSOURCE_PHOENIXD_SIGNING_KEY must be 32 bytes of hex.")
    try:
        return PrivateKey(bytes.fromhex(secret))
    except ValueError as exc:
        raise ValueError("FUNDINGSOURCE_PHOENIXD_SIGNING_KEY is not a valid secp256k1 secret key.") from exc


def _signing_key(config: LightningBackendConfig) -> PrivateKey:
    if not config.phoenixd_signing_key:
        raise ValueError("Signing key is required.")
    return parse_signing_key(config.phoenixd_signing_key.get_secret_value())


def signing_pubkey_hex(config: LightningBackendConfig) -> str:
    """This mint's LUD-25 mintPubkey under phoenixd - the configured
    signing key's compressed pubkey, what mint_pubkey and the mint-address
    discovery endpoint advertise. Purely local, no phoenixd call."""
    return _signing_key(config).public_key.format(compressed=True).hex()


async def dispatch(operation: str, config: LightningBackendConfig, leading: tuple, trailing: tuple) -> Any:
    """node._dispatch's phoenixd entry point - same shape as spark.dispatch:
    fn(*leading, config, *trailing)."""
    impls: dict[str, Callable[..., Any]] = {
        "create_invoice": _create_invoice_phoenixd,
        "pay_invoice": _pay_invoice_phoenixd,
        "is_payment_complete": _is_payment_complete_phoenixd,
        "invoice_preimage": _invoice_preimage_phoenixd,
        "payment_preimage": _payment_preimage_phoenixd,
        "sign_message": _sign_message_phoenixd,
        "is_invoice_settled": _is_invoice_settled_phoenixd,
        "fetch_node_info": _fetch_node_info_phoenixd,
    }
    impl = impls.get(operation)
    if impl is None:
        raise ValueError(f"{operation} is not supported for backend 'phoenixd'.")
    return await impl(*leading, config, *trailing)


async def _call(
    config: LightningBackendConfig,
    method: str,
    path: str,
    form: dict[str, str] | None = None,
    timeout: float = _TIMEOUT_SECONDS,
) -> tuple[int, bytes]:
    """One request to phoenixd's API, as (status code, body) - the body read
    up to _MAX_RESPONSE_BYTES and no further."""
    if not config.phoenixd_url or not config.phoenixd_password:
        raise ValueError("Url and password are required.")
    url = checked_url(config.phoenixd_url, config.phoenixd_allow_remote) + path
    # Basic with an EMPTY username - what phoenixd expects
    auth = httpx.BasicAuth("", config.phoenixd_password.get_secret_value())
    body = bytearray()
    async with httpx.AsyncClient(verify=config.verify, timeout=timeout, trust_env=False) as client:
        async with client.stream(method, url, data=form, auth=auth) as res:
            status = res.status_code
            async for chunk in res.aiter_bytes():
                body += chunk
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise ValueError(f"phoenixd's answer to {path} is implausibly large.")
    return status, bytes(body)


def _detail(body: bytes) -> str:
    return body[:200].decode(errors="replace").strip()


def _require_ok(status: int, body: bytes, what: str) -> None:
    if 200 <= status < 300:
        return
    if status == 401:
        raise ValueError(f"phoenixd refused the password for {what} (401) - use phoenix.conf's http-password.")
    raise ValueError(f"phoenixd {what} failed ({status}): {_detail(body)}")


def _json_object(body: bytes, what: str) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ValueError(f"phoenixd answered {what} with malformed JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"phoenixd answered {what} with unexpected JSON.")
    return data


def _is_unknown_payment(status: int, body: bytes) -> bool:
    """phoenixd's "no such payment" - 204, as rewritten to 404 "Not found"
    (see the module docstring). Any other 404 is not this answer."""
    return status == 204 or (status == 404 and body.strip() == b"Not found")


def _checked_hash(payment_hash: str) -> str:
    """`payment_hash` as the lowercase hex phoenixd's paths and answers use -
    validated, since it becomes part of a request path."""
    normalized = payment_hash.lower()
    if not _HEX32_RE.fullmatch(normalized):
        raise ValueError("A payment hash is 32 bytes of hex.")
    return normalized


def _checked_preimage(preimage_hex: Any, payment_hash: str) -> bytes:
    if not isinstance(preimage_hex, str) or not _HEX32_RE.fullmatch(preimage_hex):
        raise ValueError("phoenixd returned a malformed preimage.")
    preimage = bytes.fromhex(preimage_hex)
    if sha256(preimage).hexdigest() != payment_hash:
        raise ValueError("phoenixd returned a preimage that does not match the payment hash.")
    return preimage


def _external_id(config: LightningBackendConfig) -> str:
    return f"{EXTERNAL_ID_PREFIX}{signing_pubkey_hex(config)}"


async def _create_invoice_phoenixd(
    amount_msat: int,
    config: LightningBackendConfig,
    memo: str = "lnurlcash mint",
    description_for_hash: str | None = None,
) -> tuple[str, None]:
    # /createinvoice takes whole sats - rejected rather than rounded, which
    # would invoice a different amount than the one asked for
    if amount_msat % 1000:
        raise ValueError("The phoenixd backend can only mint whole-sat amounts.")
    form = {"amountSat": str(amount_msat // 1000), "externalId": _external_id(config)}
    description_hash: str | None = None
    if description_for_hash is None:
        if len(memo) > _MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"phoenixd takes an invoice description of at most {_MAX_DESCRIPTION_LENGTH} characters.")
        form["description"] = memo
    else:
        # hex, which is how phoenixd parses it - raw bytes would commit the
        # invoice to a hash nobody's request has
        description_hash = sha256(description_for_hash.encode()).hexdigest()
        form["descriptionHash"] = description_hash
    status, body = await _call(config, "POST", "/createinvoice", form)
    _require_ok(status, body, "createinvoice")
    created = _json_object(body, "createinvoice")
    invoice, payment_hash = created.get("serialized"), created.get("paymentHash")
    if not isinstance(invoice, str) or not invoice or not isinstance(payment_hash, str):
        raise ValueError("phoenixd did not return an invoice and its payment hash.")
    # the invoice is what the payer pays and what the router keys the note
    # by - it must be exactly the one asked for
    decoded = bolt11.decode(invoice)
    if (
        decoded.payment_hash != _checked_hash(payment_hash)
        or decoded.amount_msat != amount_msat
        or decoded.description_hash != description_hash
    ):
        raise ValueError("phoenixd returned an invoice other than the one requested.")
    return invoice, None


async def _pay_invoice_phoenixd(invoice: str, config: LightningBackendConfig, fee_limit_msat: int) -> PaymentResult:
    try:
        decoded = bolt11.decode(invoice)
    except Exception as exc:
        raise PaymentFailed(f"The phoenixd backend could not decode the invoice: {exc}") from exc
    if decoded.amount_msat is None or not decoded.has_payment_hash:
        # /payinvoice would need an amountSat for an amountless invoice, and
        # nothing here says how much - the router only melts into invoices
        # for exactly the notes' value anyway
        raise PaymentFailed("The phoenixd backend only pays invoices that carry an amount and a payment hash.")
    # phoenixd takes no fee limit, so the budget is enforced up front against
    # its fixed fee rule - before anything is sent (see the module docstring)
    fee_msat = trampoline_fee_msat(decoded.amount_msat)
    if fee_msat > fee_limit_msat:
        raise PaymentFailed(
            f"phoenixd charges {fee_msat} msat to pay this invoice, over this melt's {fee_limit_msat} msat budget."
        )
    status, body = await _call(config, "POST", "/payinvoice", {"invoice": invoice}, timeout=_PAY_TIMEOUT_SECONDS)
    if status == 401:
        raise PaymentFailed(
            "phoenixd refused the password for /payinvoice (401) - it needs phoenix.conf's full-access "
            "http-password, not http-password-limited-access."
        )
    if 400 <= status < 500:
        # refused while parsing the request, before any payment was attempted
        raise PaymentFailed(f"phoenixd refused the payment ({status}): {_detail(body)}")
    _require_ok(status, body, "payinvoice")
    result = _json_object(body, "payinvoice")
    preimage_hex = result.get("paymentPreimage")
    if preimage_hex is None:
        reason = result.get("reason")
        if isinstance(reason, str):
            # phoenixd's PaymentFailed answer - lightning-kmp gave up on it
            raise PaymentFailed(f"phoenixd could not pay the invoice: {reason}")
        raise ValueError("phoenixd reported neither a preimage nor a failure reason.")
    preimage = _checked_preimage(preimage_hex, decoded.payment_hash)
    routing_fee_sat = result.get("routingFeeSat")
    paid_fee_msat = routing_fee_sat * 1000 if isinstance(routing_fee_sat, int) else None
    if paid_fee_msat is not None and paid_fee_msat > fee_limit_msat:
        # paid, so nothing to undo - but the fee rule this backend checks
        # melts against no longer matches phoenixd's, and must be updated
        logging.warning(
            "phoenixd paid %s with a %d msat routing fee, over its %d msat budget - "
            "check phoenixd.TRAMPOLINE_FEE_* against phoenixd's current fee",
            decoded.payment_hash,
            paid_fee_msat,
            fee_limit_msat,
        )
    return PaymentResult(preimage, paid_fee_msat)


async def _outgoing_payment(payment_hash: str, config: LightningBackendConfig) -> dict[str, Any] | None:
    """phoenixd's best attempt at paying `payment_hash` (succeeded, else
    pending, else failed), or None if it never tried."""
    status, body = await _call(config, "GET", f"/payments/outgoingbyhash/{payment_hash}")
    if _is_unknown_payment(status, body):
        return None
    _require_ok(status, body, "payments/outgoingbyhash")
    payment = _json_object(body, "payments/outgoingbyhash")
    if payment.get("paymentHash") != payment_hash:
        raise ValueError("phoenixd answered for a different outgoing payment.")
    return payment


async def _is_payment_complete_phoenixd(payment_hash: str, config: LightningBackendConfig) -> bool:
    """True once phoenixd reports the payment paid, False once it reports it
    failed (completedAt set, not paid) or has no record of it at all - it
    records a payment before its HTLC leaves (see the module docstring).
    Anything else raises: a payment without completedAt is still in
    flight, and a still-pending payment must never read as "not paid"."""
    payment = await _outgoing_payment(_checked_hash(payment_hash), config)
    if payment is None:
        return False
    is_paid = payment.get("isPaid")
    if not isinstance(is_paid, bool):
        raise ValueError("phoenixd did not say whether the payment is paid.")
    if is_paid:
        return True
    if payment.get("completedAt") is None:
        raise ValueError("phoenixd reports the payment still pending - not a terminal outcome.")
    return False


async def _payment_preimage_phoenixd(payment_hash: str, config: LightningBackendConfig) -> bytes | None:
    payment_hash = _checked_hash(payment_hash)
    payment = await _outgoing_payment(payment_hash, config)
    if payment is None or payment.get("isPaid") is not True:
        return None
    return _checked_preimage(payment.get("preimage"), payment_hash)


async def _incoming_payment(payment_hash: str, config: LightningBackendConfig) -> dict[str, Any] | None:
    """phoenixd's record of the invoice `payment_hash`, or None if it has
    none (not an invoice this phoenixd issued)."""
    status, body = await _call(config, "GET", f"/payments/incoming/{payment_hash}")
    if _is_unknown_payment(status, body):
        return None
    _require_ok(status, body, "payments/incoming")
    payment = _json_object(body, "payments/incoming")
    if payment.get("paymentHash") != payment_hash:
        raise ValueError("phoenixd answered for a different incoming payment.")
    return payment


def _settled(payment: dict[str, Any], payment_hash: str) -> bool:
    """isPaid AND receivedSat covering the invoice's own amount - never
    isPaid alone (see the module docstring)."""
    invoice, received_sat = payment.get("invoice"), payment.get("receivedSat")
    if payment.get("isPaid") is not True or not isinstance(invoice, str) or not isinstance(received_sat, int):
        return False
    decoded = bolt11.decode(invoice)
    if decoded.payment_hash != payment_hash:
        raise ValueError("phoenixd reported an invoice for a different payment hash.")
    if decoded.amount_msat is None:
        return False
    if received_sat * 1000 < decoded.amount_msat:
        # paid, but short - most likely a liquidity fee phoenixd took out of
        # the payment itself. Nothing else compares these two numbers, so
        # this warning is how an operator finds out
        logging.warning(
            "phoenixd: invoice %s is paid, but only %d sat of the %d msat it asked for arrived - not settled",
            payment_hash,
            received_sat,
            decoded.amount_msat,
        )
        return False
    return True


async def _is_invoice_settled_phoenixd(payment_hash: str, config: LightningBackendConfig) -> bool:
    payment_hash = _checked_hash(payment_hash)
    payment = await _incoming_payment(payment_hash, config)
    return payment is not None and _settled(payment, payment_hash)


async def _invoice_preimage_phoenixd(payment_hash: str, config: LightningBackendConfig) -> bytes | None:
    payment_hash = _checked_hash(payment_hash)
    payment = await _incoming_payment(payment_hash, config)
    # phoenixd reports the preimage of an unpaid invoice too - handed out
    # only once the invoice has actually settled
    if payment is None or not _settled(payment, payment_hash):
        return None
    return _checked_preimage(payment.get("preimage"), payment_hash)


async def _sign_message_phoenixd(message: str, config: LightningBackendConfig) -> tuple[bytes, int]:
    # the exact LUD-25 digest and wire format lnd/cln produce via their
    # signmessage RPC, signed locally with the configured key (RFC6979
    # deterministic nonces via libsecp256k1) - phoenixd has no signmessage
    recoverable = _signing_key(config).sign_recoverable(lightning_signed_message_digest(message), hasher=None)
    return recoverable[:64], recoverable[64]


async def _fetch_node_info_phoenixd(config: LightningBackendConfig) -> NodeInfo:
    status, body = await _call(config, "GET", "/getinfo")
    _require_ok(status, body, "getinfo")
    info = _json_object(body, "getinfo")
    node_id = info.get("nodeId")
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("phoenixd did not report its node id.")
    # getinfo answers the limited-access password too, but melting needs the
    # full-access one. An empty /payinvoice form tells them apart without
    # paying anything: refused for its missing invoice (400) past the
    # full-access check, 401 before it - so this health probe catches the
    # misconfiguration instead of every melt failing on it
    status, _ = await _call(config, "POST", "/payinvoice", {})
    if status == 401:
        raise ValueError(
            "phoenixd accepts this password for reading only: FUNDINGSOURCE_PHOENIXD_PASSWORD must be "
            "phoenix.conf's full-access http-password, not http-password-limited-access."
        )
    channels = [channel for channel in info.get("channels") or [] if isinstance(channel, dict)]
    states = [channel.get("state") for channel in channels]
    # phoenixd's channels are private ones with ACINQ's LSP, its only peer:
    # nothing is announced (no connect string, no public capacity - see
    # NodeInfo.capacity), and the LSP counts as a connected peer while a
    # channel with it is Normal
    return NodeInfo(
        alias="phoenixd",
        uri=node_id,
        color=None,
        num_channels=sum(1 for state in states if state not in _GONE_CHANNEL_STATES),
        num_peers=1 if "Normal" in states else 0,
        capacity=0,
    )
