"""Regression tests for the unanswered-melt double payout (CWE-367
time-of-check/time-of-use), found while adding the phoenixd backend.

A melt's payment request can reach the funding source and still go
unanswered: our client times out, or the connection drops, while the
request waits in the node's own queue - phoenixd hands /payinvoice to its
peer through an in-memory command queue (lightning-kmp Peer.payInvoice),
and cln's xpay may still be routing. Until the node acts on it, it
truthfully reports no such payment (phoenixd 404, lnd 404, cln an empty
listpays), which is_payment_complete rightly reads as "not paid" - so
_melt_pay's immediate confirmation used to restore the note. The holder
could then melt the same value into a second invoice while the first
payment still went out late: two payouts for one note.

The fix (router._restorable): after a failure the funding source did not
itself answer (anything but PaymentFailed), "not paid" restores a note
only once _UNCONFIRMED_RESTORE_GRACE_SECONDS have passed since the melt's
attempt began - a time NoteStore.record_melt stores with the melt, so a
restart inside the window does not reopen it. Until then the note stays
pending, and reconcile_pending_melts finalizes it if the payment turns up,
or restores it once the grace has passed and it still has not - loudly,
naming the payment hash, since the grace only makes a late payment
unlikely: a node stalled for longer while still answering lookups can pay
after the restore.

LateNode below models that node: pay_invoice parks the request in `queued`
and raises as a client timeout would, and is_payment_complete answers from
what the node has processed so far.
"""

import asyncio
from hashlib import sha256
from os import urandom

import bolt11
import httpx
import pytest
from fastapi.testclient import TestClient

import lnurl_mint.router as router_module
from lnurl_mint.config import settings
from lnurl_mint.db import NoteStore, notes
from lnurl_mint.node import PaymentFailed
from lnurl_mint.server import app
from tests.conftest import fake_invoice, fresh_secret, k1_id

VALUE = 100_000


class LateNode:
    """A funding source that may take a payment request without answering
    it, and act on it later."""

    def __init__(self) -> None:
        self.settled: set[str] = set()  # settled MINT invoices (for minting notes)
        self.last_preimage = b""
        self.preimages: dict[str, bytes] = {}
        self.pay_mode = "queued"  # "queued" | "refused"
        self.queued: list[str] = []  # requests received, not yet acted on
        self.in_flight: list[str] = []  # acted on, HTLC out, outcome open
        self.paid_out: list[str] = []  # reality ledger: invoices funds actually left for

    async def create_invoice(self, amount_msat, config, memo=""):
        preimage = urandom(32)
        self.last_preimage = preimage
        ph = sha256(preimage).hexdigest()
        self.preimages[ph] = preimage
        return fake_invoice(amount_msat, ph), preimage

    async def is_invoice_settled(self, ph, config):
        return ph in self.settled

    async def invoice_preimage(self, ph, config):
        return self.preimages.get(ph) if ph in self.settled else None

    async def pay_invoice(self, invoice, config, fee_limit_msat):
        if self.pay_mode == "refused":
            # the node's own verdict: it acted on the request and sent nothing
            raise PaymentFailed("Could not find a route to pay this invoice.")
        # the request reached the node, but no answer reaches us
        self.queued.append(invoice)
        raise httpx.ReadTimeout("the funding source did not answer in time")

    def act_on_queue(self, outcome: str) -> None:
        """The node finally gets to the queued requests: "paid" settles them,
        "in_flight" sends the HTLC and leaves it held open."""
        if outcome == "paid":
            self.paid_out.extend(self.queued)
        else:
            self.in_flight.extend(self.queued)
        self.queued.clear()

    def settle_in_flight(self) -> None:
        """Every held HTLC is finally settled by its payee."""
        self.paid_out.extend(self.in_flight)
        self.in_flight.clear()

    def _hash_of(self, invoice: str) -> str:
        return bolt11.decode(invoice).payment_hash

    async def is_payment_complete(self, payment_hash, config):
        if any(self._hash_of(pr) == payment_hash for pr in self.in_flight):
            raise ValueError("payment still pending - not a terminal outcome")
        # a queued request is not a payment yet: truthfully "no such payment"
        return any(self._hash_of(pr) == payment_hash for pr in self.paid_out)

    async def payment_preimage(self, payment_hash, config):
        return None


@pytest.fixture
def late(monkeypatch: pytest.MonkeyPatch) -> LateNode:
    node = LateNode()
    monkeypatch.setattr(settings, "fundingsource_backend", "lnd")
    monkeypatch.setattr(router_module, "create_invoice", node.create_invoice)
    monkeypatch.setattr(router_module, "is_invoice_settled", node.is_invoice_settled)
    monkeypatch.setattr(router_module, "invoice_preimage", node.invoice_preimage)
    monkeypatch.setattr(router_module, "pay_invoice", node.pay_invoice)
    monkeypatch.setattr(router_module, "is_payment_complete", node.is_payment_complete)
    monkeypatch.setattr(router_module, "payment_preimage", node.payment_preimage)
    # no real backoff in tests - see conftest.py's node fixture
    monkeypatch.setattr(router_module, "_CONFIRMATION_RETRY_DELAYS_SECONDS", ())
    return node


@pytest.fixture
def late_client(late: LateNode) -> TestClient:
    return TestClient(app)  # no lifespan, no monitor task - reconcile runs by hand


def _mint(client: TestClient, node: LateNode) -> str:
    secret, comment = fresh_secret()
    assert client.get(f"/p/cb?amount={VALUE}&comment={comment}").json().get("pr")
    node.settled.add(sha256(node.last_preimage).hexdigest())
    return secret


def _outstanding(k1: str) -> int | None:
    return notes.note_amount(k1_id(k1))


def _reconcile() -> None:
    asyncio.run(router_module.reconcile_pending_melts(settings.funding_source()))


def _age_attempt(payment_hash: str, seconds: int) -> None:
    """Moves a melt's recorded attempt `seconds` into the past."""
    with notes.conn:
        notes.conn.execute(
            "UPDATE melts SET attempted_at = attempted_at - ? WHERE payment_hash = ?", (seconds, payment_hash)
        )


def _still_pending(client: TestClient, k1: str) -> bool:
    _, h = fresh_secret()
    return client.get(f"/w/cb?k1={k1}&p1={h}").json() == {"status": "ERROR", "reason": "pending"}


def test_an_unanswered_melt_that_pays_late_is_paid_exactly_once(late_client: TestClient, late: LateNode):
    """The original interleaving: the request goes unanswered, the node has
    no record yet ("not paid"), and the payment goes out afterwards. The
    note must stay pending the whole time, never restored for a second
    melt, and end up burned by the late payment."""
    k1 = _mint(late_client, late)
    pr = fake_invoice(VALUE)
    assert late_client.get(f"/w/cb?k1={k1}&pr={pr}").json() == {"status": "OK"}

    # "not paid" so far - but the request was never answered: kept pending
    assert late.queued == [pr]
    assert _outstanding(k1) == VALUE and _still_pending(late_client, k1)
    # no second melt of the same value can start
    second = late_client.get(f"/w/cb?k1={k1}&pr={fake_invoice(VALUE)}").json()
    assert second == {"status": "ERROR", "reason": "pending"}
    # nor does a monitor tick inside the grace period restore it
    _reconcile()
    assert _still_pending(late_client, k1)

    # the node gets to the queued request; the next tick sees it paid
    late.act_on_queue("paid")
    _reconcile()
    assert _outstanding(k1) is None
    assert late.paid_out == [pr]


def test_a_restart_inside_the_grace_period_keeps_the_note_pending(
    late_client: TestClient, late: LateNode, monkeypatch: pytest.MonkeyPatch
):
    """The attempt time lives in the database, not in the process: a fresh
    NoteStore over the same file - what a restarted mint opens - still
    knows the melt is inside its grace period."""
    k1 = _mint(late_client, late)
    pr = fake_invoice(VALUE)
    assert late_client.get(f"/w/cb?k1={k1}&pr={pr}").json() == {"status": "OK"}

    restarted = NoteStore(settings.database_path)
    assert restarted.melt_attempted_at(bolt11.decode(pr).payment_hash) is not None
    monkeypatch.setattr(router_module, "notes", restarted)
    monkeypatch.setattr(router_module, "_in_flight_melts", {})  # nothing survives a restart
    _reconcile()
    assert _still_pending(late_client, k1)

    # and the restarted mint still finalizes it once the payment goes out
    late.act_on_queue("paid")
    _reconcile()
    assert _outstanding(k1) is None
    assert len(late.paid_out) == 1


def test_once_the_grace_period_has_passed_an_unpaid_melt_is_restored_loudly(
    late_client: TestClient, late: LateNode, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    k1 = _mint(late_client, late)
    pr = fake_invoice(VALUE)
    payment_hash = bolt11.decode(pr).payment_hash
    assert late_client.get(f"/w/cb?k1={k1}&pr={pr}").json() == {"status": "OK"}
    # the request is lost for good (e.g. the node restarted and dropped its queue)
    late.queued.clear()
    # the grace period is an hour - not a second less
    _age_attempt(payment_hash, router_module._UNCONFIRMED_RESTORE_GRACE_SECONDS - 5)
    _reconcile()
    assert _still_pending(late_client, k1)

    error_log: list[str] = []
    monkeypatch.setattr(router_module, "log_internal_error", lambda context, exc: error_log.append(f"{context}: {exc}"))
    _age_attempt(payment_hash, 5)
    with caplog.at_level("WARNING"):
        _reconcile()
    assert _outstanding(k1) == VALUE
    # the rare case an operator must hear about: stdout and error.log, by hash
    assert any(payment_hash in record.message for record in caplog.records if record.levelname == "WARNING")
    assert any(payment_hash in line for line in error_log)
    assert not _still_pending(late_client, k1)  # spendable again - that rotate just went through
    assert late.paid_out == []


def test_a_payment_still_in_flight_is_never_restored_even_after_the_grace_period(
    late_client: TestClient, late: LateNode
):
    k1 = _mint(late_client, late)
    pr = fake_invoice(VALUE)
    assert late_client.get(f"/w/cb?k1={k1}&pr={pr}").json() == {"status": "OK"}
    late.act_on_queue("in_flight")  # a payee holding the HTLC open

    _age_attempt(bolt11.decode(pr).payment_hash, router_module._UNCONFIRMED_RESTORE_GRACE_SECONDS * 10)
    _reconcile()
    assert _still_pending(late_client, k1)

    late.settle_in_flight()
    _reconcile()
    assert _outstanding(k1) is None
    assert late.paid_out == [pr]


def test_a_melt_the_node_refused_is_restored_at_once(late_client: TestClient, late: LateNode):
    """The grace period is for unanswered requests only: a refusal from the
    node itself (PaymentFailed) means it acted on the request, so its "not
    paid" restores the note right away, as before."""
    k1 = _mint(late_client, late)
    late.pay_mode = "refused"
    assert late_client.get(f"/w/cb?k1={k1}&pr={fake_invoice(VALUE)}").json() == {"status": "OK"}
    assert _outstanding(k1) == VALUE
    assert not _still_pending(late_client, k1)


def test_the_grace_period_is_an_hour():
    assert router_module._UNCONFIRMED_RESTORE_GRACE_SECONDS == 60 * 60


def test_a_melt_recorded_before_attempt_times_counts_as_past_the_grace_period():
    # a melts row from before the attempted_at column reads 0 - "long ago"
    ph = sha256(urandom(32)).hexdigest()
    notes.record_melt(ph, fake_invoice(VALUE, ph))
    with notes.conn:
        notes.conn.execute("UPDATE melts SET attempted_at = 0 WHERE payment_hash = ?", (ph,))
    assert notes.melt_attempted_at(ph) == 0
    assert router_module._restorable(ph)
    # and a melt this mint never recorded has nothing to wait for
    assert router_module._restorable(sha256(urandom(32)).hexdigest())
