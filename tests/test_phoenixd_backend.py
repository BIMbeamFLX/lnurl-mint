"""The phoenixd backend (lnurl_mint/phoenixd.py) against a fake phoenixd:
an httpx.MockTransport answering the way phoenixd v0.9.1's Api.kt and
JsonSerializers.kt do - form-encoded requests, Basic auth with an empty
username and either of its two passwords, JSON with null fields left
out, and a payment lookup that finds nothing answered 404 "Not found"
(phoenixd's rewrite of its own 204). phoenixd has no regtest, so this is
where the backend's reading of every answer is pinned; the last tests
drive the real router on top of it - mint, settle, melt, verify,
reconcile."""

import asyncio
import json
import logging
import time
from base64 import b64decode
from hashlib import sha256
from os import urandom
from typing import Any, NamedTuple
from urllib.parse import parse_qs
from uuid import uuid4

import bolt11
import httpx
import pytest
from bolt11.models.tags import TagChar, Tags
from bolt11.types import Bolt11
from coincurve import PrivateKey
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import lnurl_mint.db as db_module
import lnurl_mint.node as node_module
import lnurl_mint.phoenixd as phoenixd_module
import lnurl_mint.router as router_module
import lnurl_mint.server as server_module
from lnurl_mint import bech32m
from lnurl_mint.config import Settings, settings
from lnurl_mint.db import NoteStore
from lnurl_mint.node import LightningBackendConfig, PaymentFailed, PaymentResult
from lnurl_mint.phoenixd import signing_pubkey_hex, trampoline_fee_msat
from lnurl_mint.server import app
from lnurl_mint.signing import mint_pubkey, sign_note, verify_note
from tests.conftest import bearer_id, fake_invoice

FULL = "full-access-password"
LIMITED = "limited-access-password"
SIGNING_KEY = "4c" * 32
PHOENIXD_URL = "http://127.0.0.1:9740"
CONFIG = LightningBackendConfig(
    backend="phoenixd", phoenixd_url=PHOENIXD_URL, phoenixd_password=FULL, phoenixd_signing_key=SIGNING_KEY
)
LIMITED_CONFIG = CONFIG.model_copy(update={"phoenixd_password": SecretStr(LIMITED)})

_RealAsyncClient = httpx.AsyncClient


def _run(coro):
    return asyncio.run(coro)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _invoice(
    amount_msat: int | None, payment_hash: str, description: str = "test", description_hash: str | None = None
):
    """A signed BOLT11 invoice, like conftest.fake_invoice but with the
    description or description hash phoenixd was asked for."""
    tags = Tags()
    tags.add(TagChar.payment_hash, payment_hash)
    tags.add(TagChar.payment_secret, urandom(32).hex())
    if description_hash is None:
        tags.add(TagChar.description, description)
    else:
        tags.add(TagChar.description_hash, description_hash)
    return bolt11.encode(
        Bolt11(currency="bc", amount_msat=amount_msat, date=int(time.time()), tags=tags), private_key=urandom(32).hex()
    )


class Call(NamedTuple):
    method: str
    path: str
    form: dict[str, str]
    username: str | None
    password: str | None
    content_type: str | None


class FakePhoenixd:
    """phoenixd's HTTP API, as far as the backend touches it."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.client_options: list[dict[str, Any]] = []
        # payment hash -> the JSON phoenixd would answer for it
        self.incoming: dict[str, dict[str, Any]] = {}
        self.outgoing: dict[str, dict[str, Any]] = {}
        # invoices "the network" settles when paid: payment hash -> preimage
        self.payable: dict[str, bytes] = {}
        # payment hashes whose payee holds the HTLC open (a hodl invoice)
        self.held: set[str] = set()
        # canned answers by path, ahead of everything else
        self.overrides: dict[str, httpx.Response] = {}
        # transport errors by path, raised before any answer (e.g. ConnectError)
        self.transport_errors: dict[str, type[httpx.TransportError]] = {}
        # what a lookup that finds nothing answers - phoenixd's 204, as its
        # StatusPages plugin rewrites it
        self.not_found = (404, "Not found")
        self.node_id = PrivateKey().public_key.format(compressed=True).hex()
        self.channels: list[dict[str, Any]] = [
            {"state": "Normal", "channelId": "ab" * 32, "balanceSat": 90_000, "capacitySat": 2_000_000}
        ]
        # getinfo's version: BuildVersions.phoenixdVersion, "<version>-<commit>"
        self.version = "0.9.1-598c80d"
        # GET /getbalance: sats the node can spend, and those it can only
        # spend on its own liquidity fees
        self.balance_sat = 90_000
        self.fee_credit_sat = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        username = password = None
        authorization = request.headers.get("authorization", "")
        if authorization.startswith("Basic "):
            username, _, password = b64decode(authorization[6:]).decode().partition(":")
        form = {key: values[0] for key, values in parse_qs(request.content.decode(), keep_blank_values=True).items()}
        path = request.url.path
        if path in self.transport_errors:
            raise self.transport_errors[path]("phoenixd is not listening", request=request)
        self.calls.append(Call(request.method, path, form, username, password, request.headers.get("content-type")))
        if password not in (FULL, LIMITED):
            return httpx.Response(
                401, text="Invalid authentication (use basic auth with the http password set in phoenix.conf)"
            )
        if path in self.overrides:
            return self.overrides[path]
        if request.method == "GET" and path == "/getinfo":
            info = {"nodeId": self.node_id, "channels": self.channels, "chain": "mainnet", "version": self.version}
            return httpx.Response(200, json=info)
        if request.method == "GET" and path == "/getbalance":
            return httpx.Response(200, json={"balanceSat": self.balance_sat, "feeCreditSat": self.fee_credit_sat})
        if request.method == "POST" and path == "/createinvoice":
            return self._create_invoice(form)
        if request.method == "POST" and path == "/payinvoice":
            if password != FULL:
                return httpx.Response(
                    401, text="Invalid authentication (use basic auth with the http password set in phoenix.conf)"
                )
            return self._pay_invoice(request, form)
        if request.method == "GET" and path.startswith("/payments/incoming/"):
            return self._lookup(self.incoming, path.rsplit("/", 1)[1])
        if request.method == "GET" and path.startswith("/payments/outgoingbyhash/"):
            return self._lookup(self.outgoing, path.rsplit("/", 1)[1])
        return httpx.Response(404, text="Unknown endpoint (check api doc)")

    def _lookup(self, records: dict[str, dict[str, Any]], payment_hash: str) -> httpx.Response:
        record = records.get(payment_hash)
        if record is None:
            status, text = self.not_found
            return httpx.Response(status, text=text)
        return httpx.Response(200, json=record)

    def _create_invoice(self, form: dict[str, str]) -> httpx.Response:
        description, description_hash = form.get("description"), form.get("descriptionHash")
        if (description is None) == (description_hash is None):
            return httpx.Response(400, text="Must provide either a description or descriptionHash")
        if description is not None and len(description) > 128:
            return httpx.Response(400, text="Request parameter description is too long (max 128 characters)")
        if description_hash is not None and len(description_hash) != 64:
            return httpx.Response(400, text="Request parameter descriptionHash couldn't be parsed/converted to hex32")
        amount_sat = int(form["amountSat"])
        preimage = urandom(32)
        payment_hash = sha256(preimage).hexdigest()
        invoice = _invoice(amount_sat * 1000, payment_hash, description or "test", description_hash)
        record: dict[str, Any] = {
            "type": "incoming_payment",
            "subType": "lightning",
            "paymentHash": payment_hash,
            # reported before the invoice is paid, too
            "preimage": preimage.hex(),
            "invoice": invoice,
            "isPaid": False,
            "isExpired": False,
            "requestedSat": amount_sat,
            "receivedSat": 0,
            "fees": 0,
            "createdAt": _now_ms(),
        }
        if "externalId" in form:
            record["externalId"] = form["externalId"]
        if description is not None:
            record["description"] = description
        self.incoming[payment_hash] = record
        return httpx.Response(200, json={"amountSat": amount_sat, "paymentHash": payment_hash, "serialized": invoice})

    def pay_incoming(self, payment_hash: str, received_sat: int | None = None) -> None:
        """A payer pays the invoice - `received_sat` is what reaches the node,
        after any liquidity fee phoenixd takes out of the payment itself."""
        record = self.incoming[payment_hash]
        received = record["requestedSat"] if received_sat is None else received_sat
        record.update(isPaid=True, receivedSat=received, completedAt=_now_ms())

    def _pay_invoice(self, request: httpx.Request, form: dict[str, str]) -> httpx.Response:
        if "invoice" not in form:
            return httpx.Response(400, text="Request parameter invoice is missing")
        decoded = bolt11.decode(form["invoice"])
        payment_hash = decoded.payment_hash
        fee_msat = trampoline_fee_msat(decoded.amount_msat)
        # stored as pending before any HTLC leaves, as lightning-kmp does
        record: dict[str, Any] = {
            "type": "outgoing_payment",
            "subType": "lightning",
            "paymentId": str(uuid4()),
            "paymentHash": payment_hash,
            "isPaid": False,
            "sent": (decoded.amount_msat + fee_msat) // 1000,
            "fees": fee_msat,
            "invoice": form["invoice"],
            "createdAt": _now_ms(),
        }
        self.outgoing[payment_hash] = record
        if payment_hash in self.held:
            # /payinvoice answers only once the payment ends - the client gives up first
            raise httpx.ReadTimeout("phoenixd did not answer in time", request=request)
        record["completedAt"] = _now_ms()
        preimage = self.payable.get(payment_hash)
        if preimage is None:
            return httpx.Response(200, json={"paymentHash": payment_hash, "reason": "recipient node is unreachable"})
        record.update(isPaid=True, preimage=preimage.hex())
        return httpx.Response(
            200,
            json={
                "recipientAmountSat": decoded.amount_msat // 1000,
                "routingFeeSat": fee_msat // 1000,
                "paymentId": record["paymentId"],
                "paymentHash": payment_hash,
                "paymentPreimage": preimage.hex(),
            },
        )

    def finish_held(self, payment_hash: str, preimage: bytes | None) -> None:
        """The held payment ends: settled with `preimage`, or failed back."""
        self.held.discard(payment_hash)
        record = self.outgoing[payment_hash]
        record["completedAt"] = _now_ms()
        if preimage is not None:
            record.update(isPaid=True, preimage=preimage.hex())

    def paid_invoices(self) -> list[str]:
        return [call.form["invoice"] for call in self.calls if call.path == "/payinvoice" and "invoice" in call.form]


@pytest.fixture
def phoenixd(monkeypatch: pytest.MonkeyPatch, tmp_path) -> FakePhoenixd:
    fake = FakePhoenixd()

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        fake.client_options.append(dict(kwargs))
        kwargs.pop("verify", None)
        return _RealAsyncClient(*args, transport=httpx.MockTransport(fake.handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    # the backend's own bookkeeping, fresh per test: its logged-once state,
    # and the store its fee credit baseline lives in (the suite's shared
    # database keeps serving the router)
    monkeypatch.setattr(phoenixd_module, "_degraded", {})
    monkeypatch.setattr(phoenixd_module, "_logged_versions", set())
    monkeypatch.setattr(db_module, "notes", NoteStore(str(tmp_path / "phoenixd-meta.db")))
    return fake


# --- create_invoice ----------------------------------------------------------


def test_create_invoice_posts_whole_sats_with_a_prefixed_external_id(phoenixd: FakePhoenixd):
    pr, preimage = _run(node_module.create_invoice(21_000, CONFIG, "a memo"))
    # phoenixd picks the preimage - the router reads the hash off the invoice
    assert preimage is None
    call = phoenixd.calls[-1]
    assert (call.method, call.path) == ("POST", "/createinvoice")
    assert call.content_type == "application/x-www-form-urlencoded"
    # Basic auth with an EMPTY username
    assert (call.username, call.password) == ("", FULL)
    assert call.form == {
        "amountSat": "21",
        "description": "a memo",
        "externalId": f"lnurlcash:{signing_pubkey_hex(CONFIG)}",
        # an hour, not phoenixd's default day
        "expirySeconds": "3600",
    }
    decoded = bolt11.decode(pr)
    assert decoded.amount_msat == 21_000
    assert decoded.payment_hash in phoenixd.incoming


def test_create_invoice_sends_the_description_hash_as_hex(phoenixd: FakePhoenixd):
    # a NIP-57 zap invoice commits to sha256(zap request) - as the 64 hex
    # characters phoenixd parses, never the raw 32 bytes
    zap_request = json.dumps({"kind": 9734, "content": "gm", "tags": [["p", "ab" * 32]]})
    pr, _ = _run(node_module.create_invoice(21_000, CONFIG, description_for_hash=zap_request))
    expected = sha256(zap_request.encode()).hexdigest()
    form = phoenixd.calls[-1].form
    assert form["descriptionHash"] == expected
    assert "description" not in form
    assert bolt11.decode(pr).description_hash == expected


def test_create_invoice_rejects_fractional_sats_before_asking_phoenixd(phoenixd: FakePhoenixd):
    with pytest.raises(ValueError, match="whole-sat"):
        _run(node_module.create_invoice(21_500, CONFIG))
    assert phoenixd.calls == []


def test_create_invoice_rejects_an_overlong_description(phoenixd: FakePhoenixd):
    with pytest.raises(ValueError, match="128"):
        _run(node_module.create_invoice(21_000, CONFIG, "x" * 129))
    assert phoenixd.calls == []


@pytest.mark.parametrize(
    "answer",
    [
        # a different amount than asked for
        {"amountSat": 42, "paymentHash": "ab" * 32, "serialized": _invoice(42_000, "ab" * 32)},
        # a paymentHash the invoice does not carry
        {"amountSat": 21, "paymentHash": "cd" * 32, "serialized": _invoice(21_000, "ab" * 32)},
        # no invoice at all
        {"amountSat": 21, "paymentHash": "ab" * 32},
    ],
)
def test_create_invoice_refuses_anything_but_the_invoice_requested(phoenixd: FakePhoenixd, answer: dict):
    phoenixd.overrides["/createinvoice"] = httpx.Response(200, json=answer)
    with pytest.raises(ValueError):
        _run(node_module.create_invoice(21_000, CONFIG))


def test_create_invoice_refuses_an_invoice_without_the_requested_description_hash(phoenixd: FakePhoenixd):
    answer = {"amountSat": 21, "paymentHash": "ab" * 32, "serialized": _invoice(21_000, "ab" * 32, "zap")}
    phoenixd.overrides["/createinvoice"] = httpx.Response(200, json=answer)
    with pytest.raises(ValueError, match="other than"):
        _run(node_module.create_invoice(21_000, CONFIG, description_for_hash="{}"))


# --- incoming: is_invoice_settled / invoice_preimage ------------------------


@pytest.mark.parametrize("not_found", [(404, "Not found"), (204, "")])
def test_an_invoice_phoenixd_does_not_know_is_unpaid(phoenixd: FakePhoenixd, not_found: tuple[int, str]):
    phoenixd.not_found = not_found
    assert _run(node_module.is_invoice_settled("ab" * 32, CONFIG)) is False
    assert _run(node_module.invoice_preimage("ab" * 32, CONFIG)) is None


def test_an_unpaid_invoice_keeps_its_preimage_back_until_it_settles(phoenixd: FakePhoenixd):
    pr, _ = _run(node_module.create_invoice(21_000, CONFIG))
    payment_hash = bolt11.decode(pr).payment_hash
    # phoenixd itself reports the preimage of an unpaid invoice ...
    assert phoenixd.incoming[payment_hash]["preimage"]
    # ... which must not be passed on before the invoice settles
    assert _run(node_module.is_invoice_settled(payment_hash, CONFIG)) is False
    assert _run(node_module.invoice_preimage(payment_hash, CONFIG)) is None

    phoenixd.pay_incoming(payment_hash)
    assert _run(node_module.is_invoice_settled(payment_hash, CONFIG)) is True
    preimage = _run(node_module.invoice_preimage(payment_hash, CONFIG))
    assert preimage is not None and sha256(preimage).hexdigest() == payment_hash


def test_a_short_paid_invoice_is_not_settled(phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture):
    # isPaid alone is not settlement: receivedSat is what arrived, net of
    # any liquidity fee phoenixd took out of the payment on the fly
    pr, _ = _run(node_module.create_invoice(21_000, CONFIG))
    payment_hash = bolt11.decode(pr).payment_hash
    phoenixd.pay_incoming(payment_hash, received_sat=20)
    with caplog.at_level(logging.WARNING):
        assert _run(node_module.is_invoice_settled(payment_hash, CONFIG)) is False
    assert _run(node_module.invoice_preimage(payment_hash, CONFIG)) is None
    assert any("only 20 sat" in record.message for record in caplog.records)


def test_an_overpaid_invoice_is_settled(phoenixd: FakePhoenixd):
    # BOLT 4 lets a payer send up to twice the amount asked for
    pr, _ = _run(node_module.create_invoice(21_000, CONFIG))
    payment_hash = bolt11.decode(pr).payment_hash
    phoenixd.pay_incoming(payment_hash, received_sat=42)
    assert _run(node_module.is_invoice_settled(payment_hash, CONFIG)) is True


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(404, text="Unknown endpoint (check api doc)"),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"paymentHash": "cd" * 32, "isPaid": True}),
    ],
)
def test_an_incoming_lookup_without_a_clear_answer_raises(phoenixd: FakePhoenixd, answer: httpx.Response):
    phoenixd.overrides[f"/payments/incoming/{'ab' * 32}"] = answer
    with pytest.raises(ValueError):
        _run(node_module.is_invoice_settled("ab" * 32, CONFIG))


# --- pay_invoice --------------------------------------------------------------


def test_pay_invoice_returns_the_preimage_and_the_fee_phoenixd_reports(phoenixd: FakePhoenixd):
    preimage = urandom(32)
    payment_hash = sha256(preimage).hexdigest()
    phoenixd.payable[payment_hash] = preimage
    pr = fake_invoice(1_000_000, payment_hash)
    result = _run(node_module.pay_invoice(pr, CONFIG, fee_limit_msat=10_000))
    # 4 sat + 0.4% of 1000 sat = 8 sat, reported as routingFeeSat
    assert result == PaymentResult(preimage, 8_000)
    call = phoenixd.calls[-1]
    assert (call.method, call.path, call.password) == ("POST", "/payinvoice", FULL)
    # never an amountSat: phoenixd pays the invoice's own amount
    assert call.form == {"invoice": pr}
    assert phoenixd.client_options[-1]["timeout"] == phoenixd_module._PAY_TIMEOUT_SECONDS


def test_pay_invoice_pays_a_fractional_sat_invoice_at_its_own_amount(phoenixd: FakePhoenixd):
    preimage = urandom(32)
    phoenixd.payable[sha256(preimage).hexdigest()] = preimage
    pr = fake_invoice(10_500, sha256(preimage).hexdigest())
    assert _run(node_module.pay_invoice(pr, CONFIG, fee_limit_msat=5_000)).preimage == preimage
    assert phoenixd.calls[-1].form == {"invoice": pr}


def test_a_failed_payment_is_a_clean_payment_failure(phoenixd: FakePhoenixd):
    pr = fake_invoice(21_000, "ab" * 32)  # nobody settles it
    with pytest.raises(PaymentFailed, match="recipient node is unreachable"):
        _run(node_module.pay_invoice(pr, CONFIG, fee_limit_msat=5_000))
    # and the confirmation the router makes next sees it failed for good
    assert _run(node_module.is_payment_complete("ab" * 32, CONFIG)) is False


def test_a_melt_over_its_fee_budget_is_refused_before_anything_is_sent(phoenixd: FakePhoenixd):
    # phoenixd takes no fee limit - its fixed fee is checked instead: 8000
    # msat for 1000 sat, one msat over this budget
    pr = fake_invoice(1_000_000, "ab" * 32)
    with pytest.raises(PaymentFailed, match="budget"):
        _run(node_module.pay_invoice(pr, CONFIG, fee_limit_msat=7_999))
    assert phoenixd.calls == []
    # never sent, provably - the router's confirmation restores the note
    assert _run(node_module.is_payment_complete("ab" * 32, CONFIG)) is False


def test_a_fee_over_budget_after_the_fact_is_logged(phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture):
    preimage = urandom(32)
    payment_hash = sha256(preimage).hexdigest()
    answer = {
        "recipientAmountSat": 21,
        "routingFeeSat": 9,
        "paymentId": str(uuid4()),
        "paymentHash": payment_hash,
        "paymentPreimage": preimage.hex(),
    }
    phoenixd.overrides["/payinvoice"] = httpx.Response(200, json=answer)
    with caplog.at_level(logging.WARNING):
        result = _run(node_module.pay_invoice(fake_invoice(21_000, payment_hash), CONFIG, fee_limit_msat=5_000))
    # paid - nothing to undo - but loud: phoenixd's fee rule changed
    assert result == PaymentResult(preimage, 9_000)
    assert any("over its 5000 msat budget" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(400, text="Request parameter invoice couldn't be parsed/converted to bolt11invoice"),
        httpx.Response(401, text="Invalid authentication (use basic auth with the http password set in phoenix.conf)"),
    ],
)
def test_phoenixds_own_refusals_are_clean_payment_failures(phoenixd: FakePhoenixd, answer: httpx.Response):
    # phoenixd answers 400/401 before it hands anything to its peer - nothing
    # can be queued, so the router may restore at once after confirming
    phoenixd.overrides["/payinvoice"] = answer
    with pytest.raises(PaymentFailed):
        _run(node_module.pay_invoice(fake_invoice(21_000, "ab" * 32), CONFIG, fee_limit_msat=5_000))


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(403, text="Forbidden"),  # a reverse proxy's
        httpx.Response(404, text="Unknown endpoint (check api doc)"),
        httpx.Response(500, text="Internal error"),
        httpx.Response(502, text="Bad Gateway"),
        httpx.Response(504, text="Gateway Timeout"),
    ],
)
def test_any_other_status_is_ambiguous_not_a_clean_failure(phoenixd: FakePhoenixd, answer: httpx.Response):
    # none of these proves phoenixd never queued the request - they must not
    # skip the router's grace period the way a PaymentFailed does
    phoenixd.overrides["/payinvoice"] = answer
    with pytest.raises(ValueError) as failed:
        _run(node_module.pay_invoice(fake_invoice(21_000, "ab" * 32), CONFIG, fee_limit_msat=5_000))
    assert not isinstance(failed.value, PaymentFailed)


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_a_request_that_never_reached_phoenixd_is_a_clean_failure(
    phoenixd: FakePhoenixd, error: type[httpx.TransportError]
):
    phoenixd.transport_errors["/payinvoice"] = error
    with pytest.raises(PaymentFailed, match="could not be reached"):
        _run(node_module.pay_invoice(fake_invoice(21_000, "ab" * 32), CONFIG, fee_limit_msat=5_000))


def test_the_limited_access_password_cannot_melt(phoenixd: FakePhoenixd):
    with pytest.raises(PaymentFailed, match="full-access"):
        _run(node_module.pay_invoice(fake_invoice(21_000), LIMITED_CONFIG, fee_limit_msat=5_000))
    assert phoenixd.outgoing == {}


def test_an_amountless_invoice_is_refused(phoenixd: FakePhoenixd):
    with pytest.raises(PaymentFailed, match="amount"):
        _run(node_module.pay_invoice(_invoice(None, "ab" * 32), CONFIG, fee_limit_msat=5_000))
    assert phoenixd.calls == []


def test_a_preimage_that_does_not_open_the_invoice_is_ambiguous_not_paid(phoenixd: FakePhoenixd):
    answer = {"routingFeeSat": 4, "paymentId": str(uuid4()), "paymentHash": "ab" * 32, "paymentPreimage": "cd" * 32}
    phoenixd.overrides["/payinvoice"] = httpx.Response(200, json=answer)
    with pytest.raises(ValueError, match="does not match"):
        _run(node_module.pay_invoice(fake_invoice(21_000, "ab" * 32), CONFIG, fee_limit_msat=5_000))


def test_a_payment_phoenixd_never_answers_for_is_ambiguous_not_failed(phoenixd: FakePhoenixd):
    # a payee holding a hodl invoice: /payinvoice waits, the client gives up
    phoenixd.held.add("ab" * 32)
    with pytest.raises(httpx.ReadTimeout):
        _run(node_module.pay_invoice(fake_invoice(21_000, "ab" * 32), CONFIG, fee_limit_msat=5_000))
    with pytest.raises(ValueError, match="pending"):
        _run(node_module.is_payment_complete("ab" * 32, CONFIG))


# --- outgoing: is_payment_complete / payment_preimage -------------------------


@pytest.mark.parametrize("not_found", [(404, "Not found"), (204, "")])
def test_a_payment_phoenixd_never_sent_is_not_complete(phoenixd: FakePhoenixd, not_found: tuple[int, str]):
    phoenixd.not_found = not_found
    assert _run(node_module.is_payment_complete("ab" * 32, CONFIG)) is False
    assert _run(node_module.payment_preimage("ab" * 32, CONFIG)) is None


def test_is_payment_complete_by_status(phoenixd: FakePhoenixd):
    preimage = urandom(32)
    payment_hash = sha256(preimage).hexdigest()
    record: dict[str, Any] = {
        "type": "outgoing_payment",
        "subType": "lightning",
        "paymentId": str(uuid4()),
        "paymentHash": payment_hash,
        "isPaid": False,
        "sent": 25,
        "fees": 4_084,
        "createdAt": _now_ms(),
    }
    phoenixd.outgoing[payment_hash] = record
    # in flight (no completedAt): never "not paid"
    with pytest.raises(ValueError, match="pending"):
        _run(node_module.is_payment_complete(payment_hash, CONFIG))
    assert _run(node_module.payment_preimage(payment_hash, CONFIG)) is None

    record["completedAt"] = _now_ms()  # failed
    assert _run(node_module.is_payment_complete(payment_hash, CONFIG)) is False
    assert _run(node_module.payment_preimage(payment_hash, CONFIG)) is None
    # a preimage on a payment not reported paid proves nothing - never served
    record["preimage"] = preimage.hex()
    assert _run(node_module.payment_preimage(payment_hash, CONFIG)) is None

    record.update(isPaid=True)  # succeeded
    assert _run(node_module.is_payment_complete(payment_hash, CONFIG)) is True
    assert _run(node_module.payment_preimage(payment_hash, CONFIG)) == preimage


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(404, text="Unknown endpoint (check api doc)"),
        httpx.Response(401, text="Invalid authentication (use basic auth with the http password set in phoenix.conf)"),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        # an answer about some other payment
        httpx.Response(200, json={"paymentHash": "cd" * 32, "isPaid": False, "completedAt": 1}),
        # no isPaid at all
        httpx.Response(200, json={"paymentHash": "ab" * 32, "completedAt": 1}),
    ],
)
def test_is_payment_complete_raises_without_a_clear_answer(phoenixd: FakePhoenixd, answer: httpx.Response):
    phoenixd.overrides[f"/payments/outgoingbyhash/{'ab' * 32}"] = answer
    with pytest.raises(ValueError):
        _run(node_module.is_payment_complete("ab" * 32, CONFIG))


# --- signing ------------------------------------------------------------------


def test_notes_are_signed_locally_with_the_configured_key(phoenixd: FakePhoenixd):
    message = "LNURLcash:5000:" + "ab" * 32
    r_s, recovery_id = _run(node_module.sign_message(message, CONFIG))
    pubkey = PrivateKey(bytes.fromhex(SIGNING_KEY)).public_key.format(compressed=True).hex()
    assert signing_pubkey_hex(CONFIG) == pubkey
    assert verify_note(pubkey, "ab" * 32, 5000, (r_s + bytes([recovery_id])).hex())
    # RFC6979: the same message always signs the same way
    assert _run(node_module.sign_message(message, CONFIG)) == (r_s, recovery_id)
    # phoenixd has no signmessage - nothing was asked of it
    assert phoenixd.calls == []


def test_mint_pubkey_is_the_signing_key_not_the_node_id(phoenixd: FakePhoenixd):
    assert _run(mint_pubkey(CONFIG)) == signing_pubkey_hex(CONFIG)
    assert signing_pubkey_hex(CONFIG) != phoenixd.node_id
    assert phoenixd.calls == []


def test_sign_note_end_to_end_against_verify_note(phoenixd: FakePhoenixd):
    h = urandom(32).hex()
    signature = _run(sign_note(h, 5000, CONFIG))
    assert signature is not None
    assert verify_note(signing_pubkey_hex(CONFIG), h, 5000, signature)


def test_without_a_signing_key_notes_go_unsigned_rather_than_failing(phoenixd: FakePhoenixd):
    assert _run(sign_note("ab" * 32, 5000, CONFIG.model_copy(update={"phoenixd_signing_key": None}))) is None


# --- fetch_node_info ----------------------------------------------------------


def test_fetch_node_info_reports_the_node_and_no_public_capacity(phoenixd: FakePhoenixd):
    info = _run(node_module.fetch_node_info(CONFIG))
    assert info.uri == phoenixd.node_id
    assert info.uris == [] and info.color is None
    # one private channel with the LSP, which is connected - and nothing public
    assert (info.num_channels, info.num_peers, info.capacity) == (1, 1, 0)

    phoenixd.channels = [{"state": "Offline"}, {"state": "Closed"}]
    info = _run(node_module.fetch_node_info(CONFIG))
    assert (info.num_channels, info.num_peers) == (1, 0)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.levelname == "WARNING"]


def test_the_health_check_warns_about_the_limited_access_password(
    phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture
):
    # getinfo answers the limited-access password too; the probe does not.
    # phoenixd is still reachable, so the check itself passes - it warns once
    with caplog.at_level(logging.INFO):
        assert _run(node_module.fetch_node_info(LIMITED_CONFIG)).uri == phoenixd.node_id
        _run(node_module.fetch_node_info(LIMITED_CONFIG))
    reading_only = [message for message in _warnings(caplog) if "reading only" in message]
    assert len(reading_only) == 1  # once, not on every health check
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _run(node_module.fetch_node_info(CONFIG))
    assert any("melting works again" in record.message for record in caplog.records)
    # an empty /payinvoice form - nothing was, or could have been, paid - sent
    # as a form, so phoenixd parses it and refuses it for the missing invoice
    probes = [call for call in phoenixd.calls if call.path == "/payinvoice"]
    assert [(call.form, call.content_type) for call in probes] == [({}, "application/x-www-form-urlencoded")] * 3
    assert phoenixd.outgoing == {}


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(415, text="Unsupported Media Type"),
        httpx.Response(404, text="Unknown endpoint (check api doc)"),
        httpx.Response(200, json={}),
    ],
)
def test_only_phoenixds_missing_invoice_refusal_proves_full_access(
    phoenixd: FakePhoenixd, answer: httpx.Response, caplog: pytest.LogCaptureFixture
):
    phoenixd.overrides["/payinvoice"] = answer
    with caplog.at_level(logging.WARNING):
        _run(node_module.fetch_node_info(CONFIG))
    assert any("not 400" in message for message in _warnings(caplog))


def test_fee_credit_stops_minting_but_not_the_health_check(phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture):
    # fee credit reads as a full payment but can never pay a melt: while
    # phoenixd holds any (no baseline recorded), no invoice is made - but
    # phoenixd stays healthy, so pending melts keep being reconciled
    phoenixd.fee_credit_sat = 21
    with pytest.raises(ValueError, match="no baseline is recorded"):
        _run(node_module.create_invoice(21_000, CONFIG))
    assert not any(call.path == "/createinvoice" for call in phoenixd.calls)
    with caplog.at_level(logging.INFO):
        assert _run(node_module.fetch_node_info(CONFIG)).uri == phoenixd.node_id
    assert any("21 sat of fee credit" in message for message in _warnings(caplog))

    phoenixd.fee_credit_sat = 0
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")
    with caplog.at_level(logging.INFO):
        _run(node_module.fetch_node_info(CONFIG))
    assert any("minting works again" in record.message for record in caplog.records)


class _Clock:
    """Stands in for phoenixd.time: monotonic() moves only when told to."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return time.time()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(phoenixd_module, "time", fake)
    monkeypatch.setattr(phoenixd_module, "_zero_reading_since", None)
    return fake


def test_fee_credit_is_measured_against_the_recorded_baseline(phoenixd: FakePhoenixd, clock: _Clock):
    # a liquidity purchase leaves some credit behind for good - that much
    # must not stop minting, only credit gained since
    phoenixd.fee_credit_sat = 5
    assert _run(phoenixd_module.record_fee_credit_baseline(CONFIG)) == (None, 5)
    assert db_module.notes.fee_credit_baseline() == 5
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")

    phoenixd.fee_credit_sat = 6  # a payment landed in fee credit
    with pytest.raises(ValueError, match="above its 5 sat baseline"):
        _run(node_module.create_invoice(21_000, CONFIG))

    # credit that shrinks (a liquidity purchase) lowers the baseline at once,
    # so growth from there on counts, even below the old baseline
    phoenixd.fee_credit_sat = 3
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")
    assert db_module.notes.fee_credit_baseline() == 3
    phoenixd.fee_credit_sat = 4
    with pytest.raises(ValueError, match="above its 3 sat baseline"):
        _run(node_module.create_invoice(21_000, CONFIG))
    # the health check lowers it too, and never raises it
    phoenixd.fee_credit_sat = 1
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 1
    phoenixd.fee_credit_sat = 9
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 1


def test_a_receipt_right_after_a_purchase_is_not_hidden(phoenixd: FakePhoenixd, clock: _Clock):
    # the review's probe: a purchase drops the credit from 3000 to 1000, and a
    # minute later a 1500 sat receipt makes it 2500 - still below the old
    # baseline, but credit gained after the purchase all the same
    phoenixd.fee_credit_sat = 3_000
    _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    phoenixd.fee_credit_sat = 1_000
    _run(node_module.fetch_node_info(CONFIG))
    clock.now += 60
    phoenixd.fee_credit_sat = 2_500
    with pytest.raises(ValueError, match="rose to 2500 sat, above its 1000 sat baseline"):
        _run(node_module.create_invoice(21_000, CONFIG))


def test_a_zero_reading_lowers_the_baseline_only_after_five_minutes(phoenixd: FakePhoenixd, clock: _Clock):
    phoenixd.fee_credit_sat = 5
    _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    phoenixd.fee_credit_sat = 0
    _run(node_module.fetch_node_info(CONFIG))
    clock.now += 299
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 5
    clock.now += 1  # 300 s after the first 0
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 0
    assert phoenixd_module._BASELINE_SETTLE_SECONDS == 300


def test_a_reading_back_at_the_baseline_restarts_the_zero_window(phoenixd: FakePhoenixd, clock: _Clock):
    phoenixd.fee_credit_sat = 5
    _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    phoenixd.fee_credit_sat = 0
    _run(node_module.fetch_node_info(CONFIG))  # a 0 from t=0
    clock.now += 200
    phoenixd.fee_credit_sat = 5  # back at the baseline: that 0 was a transient
    _run(node_module.fetch_node_info(CONFIG))
    clock.now += 99
    phoenixd.fee_credit_sat = 0  # a new 0 from t=299
    _run(node_module.fetch_node_info(CONFIG))
    clock.now += 201  # 500 s after the first 0, 201 s after this one
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 5
    clock.now += 99  # 300 s after this 0
    _run(node_module.fetch_node_info(CONFIG))
    assert db_module.notes.fee_credit_baseline() == 0


def test_phoenixds_zero_after_a_restart_does_not_pin_the_baseline(phoenixd: FakePhoenixd, clock: _Clock):
    # right after a restart phoenixd reports no fee credit until its LSP's
    # CurrentFeeCredit arrives - that transient 0 must not become the
    # baseline, or minting would stay off once the real credit is back
    phoenixd.fee_credit_sat = 5
    _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    phoenixd.fee_credit_sat = 0
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")
    clock.now += phoenixd_module._BASELINE_SETTLE_SECONDS - 1
    phoenixd.fee_credit_sat = 5  # the LSP's figure arrives
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")
    clock.now += 10
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")
    assert db_module.notes.fee_credit_baseline() == 5


def test_the_baseline_is_re_read_after_lowering(phoenixd: FakePhoenixd, clock: _Clock, monkeypatch: pytest.MonkeyPatch):
    phoenixd.fee_credit_sat = 5
    _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    store = db_module.notes
    lower = store.lower_fee_credit_baseline

    def lowered_further_meanwhile(fee_credit_sat: int) -> None:
        lower(fee_credit_sat)
        lower(2)  # an interleaved call saw a smaller credit still

    monkeypatch.setattr(store, "lower_fee_credit_baseline", lowered_further_meanwhile)
    phoenixd.fee_credit_sat = 4  # lowers to 4 - and meanwhile to 2
    with pytest.raises(ValueError, match="above its 2 sat baseline"):
        _run(node_module.create_invoice(21_000, CONFIG))


def test_a_read_only_password_cannot_mint(phoenixd: FakePhoenixd):
    # every melt would meet the same 401 - so no deposits either, whether the
    # health check found it first or create_invoice probes itself
    with pytest.raises(ValueError, match="Minting is refused while melting cannot work"):
        _run(node_module.create_invoice(21_000, LIMITED_CONFIG))
    assert not any(call.path == "/createinvoice" for call in phoenixd.calls)
    _run(node_module.fetch_node_info(LIMITED_CONFIG))
    with pytest.raises(ValueError, match="reading only"):
        _run(node_module.create_invoice(21_000, LIMITED_CONFIG))
    assert not any(call.path == "/createinvoice" for call in phoenixd.calls)
    # proven full access, cached from the health check, mints
    _run(node_module.fetch_node_info(CONFIG))
    assert _run(node_module.create_invoice(21_000, CONFIG))[0].startswith("lnbc")


def test_the_stored_baseline_is_only_ever_lowered(tmp_path):
    store = NoteStore(str(tmp_path / "baseline.db"))
    assert store.fee_credit_baseline() is None
    store.lower_fee_credit_baseline(3)  # nothing recorded, nothing to lower
    assert store.fee_credit_baseline() is None
    store.record_fee_credit_baseline(5)
    store.lower_fee_credit_baseline(7)
    assert store.fee_credit_baseline() == 5
    store.lower_fee_credit_baseline(2)
    assert store.fee_credit_baseline() == 2
    store.record_fee_credit_baseline(8)  # only the operator's own step raises it
    assert store.fee_credit_baseline() == 8


def test_the_baseline_command_records_the_fee_credit(
    phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setattr(settings, "fundingsource_backend", "phoenixd")
    monkeypatch.setattr(settings, "fundingsource_phoenixd_url", PHOENIXD_URL)
    monkeypatch.setattr(settings, "fundingsource_phoenixd_password", SecretStr(FULL))
    monkeypatch.setattr(settings, "fundingsource_phoenixd_signing_key", SecretStr(SIGNING_KEY))
    phoenixd.fee_credit_sat = 1_234
    phoenixd_module.main(["record-fee-credit-baseline"])
    assert db_module.notes.fee_credit_baseline() == 1_234
    assert "1234 sat as the baseline (was: none)" in capsys.readouterr().out

    # raising it needs --force - the credit gained since may back notes
    phoenixd.fee_credit_sat = 1_300
    with pytest.raises(SystemExit, match="1300 sat, above the recorded baseline of 1234 sat"):
        phoenixd_module.main(["record-fee-credit-baseline"])
    assert db_module.notes.fee_credit_baseline() == 1_234
    phoenixd_module.main(["record-fee-credit-baseline", "--force"])
    assert db_module.notes.fee_credit_baseline() == 1_300
    assert "(was: 1234 sat)" in capsys.readouterr().out
    # lowering it needs nothing
    phoenixd.fee_credit_sat = 7
    phoenixd_module.main(["record-fee-credit-baseline"])
    assert db_module.notes.fee_credit_baseline() == 7

    monkeypatch.setattr(settings, "fundingsource_backend", "lnd")
    with pytest.raises(SystemExit, match="not phoenixd"):
        phoenixd_module.main(["record-fee-credit-baseline"])


def _invoice_issued(amount_msat: int, seconds_ago: int, expiry: int) -> str:
    """A signed invoice dated `seconds_ago`, valid for `expiry` seconds."""
    tags = Tags()
    tags.add(TagChar.payment_hash, urandom(32).hex())
    tags.add(TagChar.payment_secret, urandom(32).hex())
    tags.add(TagChar.description, "test")
    tags.add(TagChar.expire_time, expiry)
    issued = int(time.time()) - seconds_ago
    return bolt11.encode(Bolt11(currency="bc", amount_msat=amount_msat, date=issued, tags=tags), urandom(32).hex())


def _expired_invoice(amount_msat: int) -> str:
    return _invoice_issued(amount_msat, seconds_ago=7200, expiry=3600)


def _mint_row(invoice: str, seconds_ago: int = 0) -> str:
    """Records `invoice` as a mint invoice issued `seconds_ago` - its payment hash."""
    store = db_module.notes
    payment_hash = bolt11.decode(invoice).payment_hash
    store.create_mint(payment_hash, invoice, 21_000, urandom(32).hex())
    with store.conn:
        store.conn.execute(
            "UPDATE mints SET created_at = ? WHERE payment_hash = ?", (int(time.time()) - seconds_ago, payment_hash)
        )
    return payment_hash


def test_the_baseline_command_looks_back_seven_days(phoenixd: FakePhoenixd):
    # phoenixd has only ever issued invoices valid for a day at most, so a
    # week back covers every one that could still be paid - an invoice issued
    # six days ago with a longer life still counts, one from eight days ago
    # lies outside the window
    day = 24 * 60 * 60
    _mint_row(_invoice_issued(21_000, seconds_ago=8 * day, expiry=30 * day), seconds_ago=8 * day)
    assert _run(phoenixd_module.record_fee_credit_baseline(CONFIG)) == (None, 0)
    _mint_row(_invoice_issued(21_000, seconds_ago=6 * day, expiry=30 * day), seconds_ago=6 * day)
    with pytest.raises(ValueError, match="1 mint invoice"):
        _run(phoenixd_module.record_fee_credit_baseline(CONFIG))
    assert phoenixd_module._PAYABLE_LOOKBACK_SECONDS == 7 * day


def test_a_mint_invoice_already_settled_into_a_note_does_not_count(phoenixd: FakePhoenixd):
    # still unexpired, but paid and minted - no later payment can land on it
    payment_hash = _mint_row(fake_invoice(21_000))
    assert db_module.notes.settle_mint(payment_hash) == 21_000
    assert _run(phoenixd_module.record_fee_credit_baseline(CONFIG)) == (None, 0)


def test_no_baseline_is_recorded_while_a_mint_invoice_can_still_be_paid(phoenixd: FakePhoenixd):
    # a payment into fee credit after recording would pass for baseline
    store = db_module.notes
    expired = _expired_invoice(21_000)
    store.create_mint(bolt11.decode(expired).payment_hash, expired, 21_000, urandom(32).hex())
    phoenixd.fee_credit_sat = 3
    assert _run(phoenixd_module.record_fee_credit_baseline(CONFIG)) == (None, 3)  # expired ones don't count

    payable = fake_invoice(21_000)
    store.create_mint(bolt11.decode(payable).payment_hash, payable, 21_000, urandom(32).hex())
    with pytest.raises(ValueError, match="1 mint invoice"):
        _run(phoenixd_module.record_fee_credit_baseline(CONFIG, force=True))
    assert store.fee_credit_baseline() == 3


def test_an_unreadable_mint_invoice_counts_as_still_payable(phoenixd: FakePhoenixd):
    db_module.notes.create_mint(urandom(32).hex(), "not-an-invoice", 21_000, urandom(32).hex())
    with pytest.raises(ValueError, match="1 mint invoice"):
        _run(phoenixd_module.record_fee_credit_baseline(CONFIG))


def test_an_unreadable_fee_credit_refuses_minting_and_only_warns_the_health_check(
    phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture
):
    phoenixd.overrides["/getbalance"] = httpx.Response(200, json={"balanceSat": 90_000})
    with pytest.raises(ValueError, match="did not report its fee credit"):
        _run(node_module.create_invoice(21_000, CONFIG))
    with caplog.at_level(logging.WARNING):
        assert _run(node_module.fetch_node_info(CONFIG)).uri == phoenixd.node_id
    assert any("could not be read" in message for message in _warnings(caplog))


def test_the_version_is_logged_once_and_an_unchecked_one_warns(
    phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(phoenixd_module, "_logged_versions", set())
    with caplog.at_level(logging.INFO):
        _run(node_module.fetch_node_info(CONFIG))
        _run(node_module.fetch_node_info(CONFIG))
    checked = [record for record in caplog.records if "0.9.1-598c80d" in record.message]
    assert [record.levelname for record in checked] == ["INFO"]

    caplog.clear()
    phoenixd.version = "0.10.0-1234567"  # a fee rule nobody checked
    with caplog.at_level(logging.INFO):
        _run(node_module.fetch_node_info(CONFIG))
        _run(node_module.fetch_node_info(CONFIG))
    unchecked = [record for record in caplog.records if "0.10.0-1234567" in record.message]
    assert [record.levelname for record in unchecked] == ["WARNING"]

    caplog.clear()
    phoenixd.version = "0.5.0-7654321"  # before the checked range
    with caplog.at_level(logging.INFO):
        _run(node_module.fetch_node_info(CONFIG))
    assert [r.levelname for r in caplog.records if "0.5.0-7654321" in r.message] == ["WARNING"]


# --- the connection ----------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "allow_remote", "expected"),
    [
        ("http://127.0.0.1:9740", False, "http://127.0.0.1:9740"),
        ("http://localhost:9740/", False, "http://localhost:9740"),
        ("http://[::1]:9740", False, "http://[::1]:9740"),
        ("https://127.0.0.1:9740", False, "https://127.0.0.1:9740"),
        ("https://phoenixd.example", True, "https://phoenixd.example"),
    ],
)
def test_loopback_or_opted_in_https_urls_are_accepted(url: str, allow_remote: bool, expected: str):
    assert phoenixd_module.checked_url(url, allow_remote) == expected


@pytest.mark.parametrize(
    ("url", "allow_remote"),
    [
        ("http://10.0.0.5:9740", False),
        ("http://10.0.0.5:9740", True),  # plain http off loopback, even opted in
        ("https://phoenixd.example", False),  # https, but nobody opted in
        ("http://0.0.0.0:9740", False),
        ("http://:hunter2@127.0.0.1:9740", False),  # the password belongs in its own setting
        ("http://127.0.0.1:9740/?password=hunter2", False),
        ("ftp://127.0.0.1:9740", False),
        ("127.0.0.1:9740", False),
        ("http://127.0.0.1:99999", False),
    ],
)
def test_urls_that_could_leak_the_password_are_refused(url: str, allow_remote: bool):
    with pytest.raises(ValueError) as refused:
        phoenixd_module.checked_url(url, allow_remote)
    assert "hunter2" not in str(refused.value)


def test_a_refused_url_is_never_contacted(phoenixd: FakePhoenixd):
    remote = CONFIG.model_copy(update={"phoenixd_url": "http://10.0.0.5:9740"})
    with pytest.raises(ValueError, match="loopback"):
        _run(node_module.fetch_node_info(remote))
    assert phoenixd.calls == []


def test_requests_ignore_the_environment_and_carry_timeouts(phoenixd: FakePhoenixd):
    _run(node_module.fetch_node_info(CONFIG))
    _run(node_module.is_invoice_settled("ab" * 32, CONFIG))
    for options in phoenixd.client_options:
        # no proxy or netrc picked up from the environment - the password
        # only travels to the configured phoenixd
        assert options["trust_env"] is False
        assert options["timeout"] == phoenixd_module._TIMEOUT_SECONDS


def test_the_password_never_reaches_an_error_or_the_logs(phoenixd: FakePhoenixd, caplog: pytest.LogCaptureFixture):
    phoenixd.overrides["/getinfo"] = httpx.Response(500, text="Internal error")
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as failed:
            _run(node_module.fetch_node_info(CONFIG))
    assert FULL not in str(failed.value)
    assert FULL not in caplog.text


def test_an_oversized_answer_is_refused(phoenixd: FakePhoenixd):
    phoenixd.overrides["/getinfo"] = httpx.Response(200, content=b" " * (phoenixd_module._MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError, match="implausibly large"):
        _run(node_module.fetch_node_info(CONFIG))


def test_dispatch_rejects_unknown_operations():
    with pytest.raises(ValueError, match="not supported for backend 'phoenixd'"):
        _run(node_module._dispatch("some_future_operation", CONFIG, None, None))  # type: ignore[arg-type]


# --- the fee rule --------------------------------------------------------------


def test_the_fee_rule_matches_lightning_kmps_own_example():
    # TrampolineFees.calculateReverseAmount's doc (lightning-kmp 1.13.1):
    # at 4 sat + 0.4%, sending 992_032 msat costs 7_968 msat - 1000 sat in all
    assert trampoline_fee_msat(992_032) == 7_968


@pytest.mark.parametrize(
    "amount_msat", [1_000, 10_000, 10_500, 250_000, 251_000, 1_000_000, 3_999_000, 4_001_000, 10**8, 10**9]
)
def test_the_minimum_mint_fee_covers_every_melt(amount_msat: int, monkeypatch: pytest.MonkeyPatch):
    # the floor config.py enforces lets the router's melt budget pay
    # phoenixd's fee at every note size
    monkeypatch.setattr(settings, "base_fee_msat", phoenixd_module.TRAMPOLINE_FEE_BASE_MSAT)
    monkeypatch.setattr(settings, "fee_percent_ppm", phoenixd_module.TRAMPOLINE_FEE_PPM)
    assert router_module._melt_fee_limit_msat(amount_msat) >= trampoline_fee_msat(amount_msat)


def test_below_that_floor_mid_sized_melts_could_not_be_paid(monkeypatch: pytest.MonkeyPatch):
    # why config.py insists: between ~250 and ~4000 sat the router's own
    # floor (0.5%, 5000 msat) is below phoenixd's fee
    monkeypatch.setattr(settings, "base_fee_msat", 1_000)
    monkeypatch.setattr(settings, "fee_percent_ppm", 0)
    assert router_module._melt_fee_limit_msat(1_000_000) < trampoline_fee_msat(1_000_000)


# --- configuration ----------------------------------------------------------------


def _phoenixd_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "fundingsource_backend": "phoenixd",
        "fundingsource_phoenixd_url": PHOENIXD_URL,
        "fundingsource_phoenixd_password": FULL,
        "fundingsource_phoenixd_signing_key": SIGNING_KEY,
        "base_fee_msat": 4_000,
        "fee_percent_ppm": 4_000,
    }
    values.update(overrides)
    return Settings(**values)


def test_a_complete_phoenixd_config_reaches_the_backend():
    config = _phoenixd_settings().funding_source()
    assert config.backend == "phoenixd"
    assert config.phoenixd_url == PHOENIXD_URL
    assert config.phoenixd_password is not None and config.phoenixd_password.get_secret_value() == FULL
    assert config.phoenixd_signing_key is not None and config.phoenixd_signing_key.get_secret_value() == SIGNING_KEY
    assert config.phoenixd_allow_remote is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fundingsource_phoenixd_url": None}, "FUNDINGSOURCE_PHOENIXD_URL is required"),
        ({"fundingsource_phoenixd_url": "http://10.0.0.5:9740"}, "not a loopback address"),
        ({"fundingsource_phoenixd_url": "https://phoenixd.example"}, "not a loopback address"),
        ({"fundingsource_phoenixd_password": None}, "FUNDINGSOURCE_PHOENIXD_PASSWORD is required"),
        ({"fundingsource_phoenixd_password": ""}, "FUNDINGSOURCE_PHOENIXD_PASSWORD is required"),
        ({"fundingsource_phoenixd_signing_key": None}, "FUNDINGSOURCE_PHOENIXD_SIGNING_KEY is required"),
        ({"fundingsource_phoenixd_signing_key": "ab" * 31}, "32 bytes of hex"),
        ({"fundingsource_phoenixd_signing_key": "00" * 32}, "not a valid secp256k1 secret key"),
        ({"fundingsource_phoenixd_signing_key": "ff" * 32}, "not a valid secp256k1 secret key"),
        ({"min_sendable_msat": 10_500}, "MIN_SENDABLE_MSAT must be a whole number of sats"),
        ({"max_sendable_msat": 1_000_000_500}, "MAX_SENDABLE_MSAT must be a whole number of sats"),
        ({"min_mint_msat": 10_500}, "MIN_MINT_MSAT must be a whole number of sats"),
        ({"base_fee_msat": 3_999}, "mint fee must cover"),
        ({"fee_percent_ppm": 3_999}, "mint fee must cover"),
    ],
)
def test_an_unsafe_phoenixd_config_refuses_to_start(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        _phoenixd_settings(**overrides)


def test_a_remote_phoenixd_needs_https_and_an_explicit_opt_in():
    opted_in = _phoenixd_settings(
        fundingsource_phoenixd_url="https://phoenixd.example", fundingsource_phoenixd_allow_remote=True
    )
    assert opted_in.funding_source().phoenixd_allow_remote is True
    with pytest.raises(ValidationError, match="not a loopback address"):
        _phoenixd_settings(
            fundingsource_phoenixd_url="http://phoenixd.example", fundingsource_phoenixd_allow_remote=True
        )


def test_the_phoenixd_rules_leave_other_backends_alone():
    Settings(fundingsource_backend="lnd", min_sendable_msat=10_500, base_fee_msat=0, fee_percent_ppm=0)


def test_a_refused_config_does_not_quote_its_secrets():
    # the error a failed start prints must not carry the password or the
    # signing key - pydantic quotes its raw input unless told not to
    with pytest.raises(ValidationError) as refused:
        _phoenixd_settings(base_fee_msat=1)
    text = str(refused.value)
    assert "mint fee must cover" in text
    assert FULL not in text and SIGNING_KEY not in text and SIGNING_KEY[-8:] not in text
    assert "input_value" not in text


def test_zaps_are_offered_on_phoenixd(monkeypatch: pytest.MonkeyPatch):
    # phoenixd binds an invoice to a description hash, as lnd and cln do
    monkeypatch.setattr(settings, "nostr_key", SecretStr(urandom(32).hex()))
    monkeypatch.setattr(settings, "fundingsource_backend", "phoenixd")
    assert router_module._zaps_offered() is True


# --- the router, end to end ---------------------------------------------------------


@pytest.fixture
def mint(phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The real router and backend on top of the fake phoenixd - no FakeNode."""
    monkeypatch.setattr(settings, "fundingsource_backend", "phoenixd")
    monkeypatch.setattr(settings, "fundingsource_phoenixd_url", PHOENIXD_URL)
    monkeypatch.setattr(settings, "fundingsource_phoenixd_password", SecretStr(FULL))
    monkeypatch.setattr(settings, "fundingsource_phoenixd_signing_key", SecretStr(SIGNING_KEY))
    monkeypatch.setattr(node_module, "_node_info_cache", None)
    monkeypatch.setattr(router_module, "_CONFIRMATION_RETRY_DELAYS_SECONDS", ())
    return TestClient(app)


def _mint_note(client: TestClient, phoenixd: FakePhoenixd, amount_msat: int) -> tuple[str, str, str]:
    """(k1, h, mint payment hash) of a bearer note minted and paid for."""
    secret = urandom(32)
    h = sha256(secret).hexdigest()
    minted = client.get("/p/cb", params={"amount": amount_msat, "comment": h}).json()
    payment_hash = bolt11.decode(minted["pr"]).payment_hash
    phoenixd.pay_incoming(payment_hash)
    return secret.hex(), h, payment_hash


def test_mint_settle_melt_and_verify_through_the_router(
    mint: TestClient, phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "verify_enabled", True)
    secret = urandom(32)
    h = sha256(secret).hexdigest()
    minted = mint.get("/p/cb", params={"amount": 21_000, "comment": h}).json()
    mint_hash = bolt11.decode(minted["pr"]).payment_hash
    assert phoenixd.incoming[mint_hash]["externalId"] == f"lnurlcash:{signing_pubkey_hex(CONFIG)}"

    # unpaid: no note, and verify keeps back the preimage phoenixd already has
    assert mint.get("/w", params={"k1": secret.hex()}).json()["status"] == "ERROR"
    assert "preimage" not in mint.get(f"/verify/{mint_hash}").json()

    phoenixd.pay_incoming(mint_hash)
    note = mint.get("/w", params={"k1": secret.hex()}).json()
    assert note["maxWithdrawable"] == 21_000
    assert note["mintPubkey"] == signing_pubkey_hex(CONFIG)
    certificate = bech32m.decode_cs1(note["c"])
    assert certificate is not None
    assert verify_note(note["mintPubkey"], bearer_id(h), 21_000, certificate[1].hex())

    melt_preimage = urandom(32)
    melt_hash = sha256(melt_preimage).hexdigest()
    phoenixd.payable[melt_hash] = melt_preimage
    pr = fake_invoice(21_000, melt_hash)
    assert mint.get("/w/cb", params={"k1": secret.hex(), "pr": pr}).json()["status"] == "OK"
    # paid exactly once, and the note is gone
    assert phoenixd.paid_invoices() == [pr]
    assert mint.get("/w", params={"k1": secret.hex()}).json()["reason"] == "Note already spent."
    melt_verify = mint.get(f"/verify/{melt_hash}").json()
    assert (melt_verify["settled"], melt_verify["preimage"]) == (True, melt_preimage.hex())


def test_a_melt_phoenixd_cannot_afford_restores_the_note_unpaid(mint: TestClient, phoenixd: FakePhoenixd):
    # this suite runs fee-free (conftest), so a 1000 sat note's melt budget
    # (5000 msat) is under phoenixd's 8000 msat fee - the configuration
    # config.py refuses at startup, caught here per melt as well
    k1, _, _ = _mint_note(mint, phoenixd, 1_000_000)
    pr = fake_invoice(1_000_000, "ab" * 32)
    assert mint.get("/w/cb", params={"k1": k1, "pr": pr}).json()["status"] == "OK"
    assert phoenixd.paid_invoices() == []
    assert mint.get("/w", params={"k1": k1}).json()["maxWithdrawable"] == 1_000_000


@pytest.mark.parametrize("settles", [True, False])
def test_a_melt_held_open_stays_pending_until_phoenixd_knows(
    mint: TestClient, phoenixd: FakePhoenixd, settles: bool, monkeypatch: pytest.MonkeyPatch
):
    k1, _, _ = _mint_note(mint, phoenixd, 21_000)
    melt_preimage = urandom(32)
    melt_hash = sha256(melt_preimage).hexdigest()
    phoenixd.held.add(melt_hash)
    pr = fake_invoice(21_000, melt_hash)
    assert mint.get("/w/cb", params={"k1": k1, "pr": pr}).json()["status"] == "OK"
    # /payinvoice timed out and phoenixd reports the payment in flight:
    # the note is neither burned nor restored
    assert mint.get("/w", params={"k1": k1}).json()["reason"] == "pending"
    _run(router_module.reconcile_pending_melts(settings.funding_source()))
    assert mint.get("/w", params={"k1": k1}).json()["reason"] == "pending"

    phoenixd.finish_held(melt_hash, melt_preimage if settles else None)
    # its /payinvoice went unanswered, so a failure restores the note only
    # past the grace period - long past, for an HTLC a payee held this long
    monkeypatch.setattr(router_module, "_UNCONFIRMED_RESTORE_GRACE_SECONDS", 0)
    _run(router_module.reconcile_pending_melts(settings.funding_source()))
    after = mint.get("/w", params={"k1": k1}).json()
    if settles:
        assert after["reason"] == "Note already spent."
    else:
        assert after["maxWithdrawable"] == 21_000
    assert phoenixd.paid_invoices() == [pr]


def test_a_proxys_gateway_timeout_waits_out_the_grace_period(
    mint: TestClient, phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch
):
    # a 504 from a reverse proxy says nothing about whether phoenixd queued
    # the payment, even though phoenixd has no record of it yet: the note
    # stays pending, and only a restore after the grace period frees it
    k1, _, _ = _mint_note(mint, phoenixd, 21_000)
    phoenixd.overrides["/payinvoice"] = httpx.Response(504, text="Gateway Timeout")
    assert mint.get("/w/cb", params={"k1": k1, "pr": fake_invoice(21_000)}).json()["status"] == "OK"
    assert mint.get("/w", params={"k1": k1}).json()["reason"] == "pending"
    _run(router_module.reconcile_pending_melts(settings.funding_source()))
    assert mint.get("/w", params={"k1": k1}).json()["reason"] == "pending"

    monkeypatch.setattr(router_module, "_UNCONFIRMED_RESTORE_GRACE_SECONDS", 0)
    _run(router_module.reconcile_pending_melts(settings.funding_source()))
    assert mint.get("/w", params={"k1": k1}).json()["maxWithdrawable"] == 21_000


def test_phoenixds_own_refusal_restores_the_note_at_once(mint: TestClient, phoenixd: FakePhoenixd):
    k1, _, _ = _mint_note(mint, phoenixd, 21_000)
    phoenixd.overrides["/payinvoice"] = httpx.Response(400, text="Request parameter invoice is missing")
    assert mint.get("/w/cb", params={"k1": k1, "pr": fake_invoice(21_000)}).json()["status"] == "OK"
    assert mint.get("/w", params={"k1": k1}).json()["maxWithdrawable"] == 21_000


def test_mint_address_discovery_advertises_the_signing_key(mint: TestClient, phoenixd: FakePhoenixd):
    data = mint.get(f"/.well-known/lnurlw/{settings.username}").json()
    assert data["mintPubkey"] == signing_pubkey_hex(CONFIG)
    assert data["nodeUri"] == phoenixd.node_id


def test_a_fractional_sat_amount_is_the_wallets_error_not_the_mints(mint: TestClient, phoenixd: FakePhoenixd):
    refused = mint.get("/p/cb", params={"amount": 21_500, "comment": urandom(32).hex()}).json()
    assert refused == {"status": "ERROR", "reason": "Amount must be a whole number of sats."}
    assert phoenixd.calls == []


def test_the_mint_refuses_to_start_once_its_signing_key_changed(
    mint: TestClient, phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    # a database of its own, and no reconcile touching the suite's shared one
    monkeypatch.setattr(server_module, "notes", NoteStore(str(tmp_path / "pinned.db")))

    async def no_reconcile(funding_source: LightningBackendConfig) -> None:
        return None

    monkeypatch.setattr(server_module, "_reconcile_pending_melts_safely", no_reconcile)

    with TestClient(app):  # the first start remembers this key's mintPubkey
        pass
    with TestClient(app):  # the same key starts again
        pass
    monkeypatch.setattr(settings, "fundingsource_phoenixd_signing_key", SecretStr("5d" * 32))
    with pytest.raises(RuntimeError, match="restore the original signing key"):
        with TestClient(app):
            pass
    assert server_module.notes.pin_mint_pubkey("anything else") == signing_pubkey_hex(CONFIG)


def test_melts_are_still_reconciled_while_minting_is_blocked(
    mint: TestClient, phoenixd: FakePhoenixd, monkeypatch: pytest.MonkeyPatch, tmp_path
):
    # fee credit above its baseline stops minting only: phoenixd stays
    # healthy, so boot and every monitor tick keep reconciling pending melts
    # (which is how an unanswered melt's note is ever restored), and the
    # mint-address discovery keeps advertising mintPubkey
    phoenixd.fee_credit_sat = 21
    monkeypatch.setattr(server_module, "notes", NoteStore(str(tmp_path / "boot.db")))
    monkeypatch.setattr(settings, "funding_source_health_check_interval_seconds", 0.01)
    reconciled: list[str | None] = []

    async def record_reconcile(funding_source: LightningBackendConfig) -> None:
        reconciled.append(funding_source.backend)

    monkeypatch.setattr(server_module, "_reconcile_pending_melts_safely", record_reconcile)
    with TestClient(app):
        time.sleep(0.2)  # boot, then a few monitor ticks
    assert len(reconciled) >= 2 and set(reconciled) == {"phoenixd"}

    refused = mint.get("/p/cb", params={"amount": 21_000, "comment": urandom(32).hex()}).json()
    assert refused["status"] == "ERROR"
    assert not any(call.path == "/createinvoice" for call in phoenixd.calls)
    assert mint.get(f"/.well-known/lnurlw/{settings.username}").json()["mintPubkey"] == signing_pubkey_hex(CONFIG)
