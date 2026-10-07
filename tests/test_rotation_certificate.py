import logging

import pytest
from coincurve import PrivateKey
from fastapi.testclient import TestClient

from lnurl_mint import bech32m
from lnurl_mint.config import settings
from lnurl_mint.signing import lightning_signed_message_digest, verify_note, verify_rotation
from tests.conftest import bearer_id, fresh_secret, k1_id


def _certifies_rotation(pubkey: str, spent_id: str, h: str, amount_msat: int, cr1: str) -> bool:
    """Whether `cr1` is `pubkey`'s certificate that the note `spent_id` (hex
    Q) was burned into the bearer note named by its hex `h`, worth
    `amount_msat` - checked against the claimed amount, not the one the
    certificate's own HRP carries, so a wrong claim fails."""
    decoded = bech32m.decode_cr1(cr1)
    return decoded is not None and verify_rotation(pubkey, spent_id, bearer_id(h), amount_msat, decoded[1].hex())


def test_cr1_test_vector():
    """A fixed vector other implementations can reproduce: the SERVICE key
    is sha256("LNURLcash rotation certificate test vector"), the burned note
    is pk_0 of 25.md's test vector 1 and the credited one pk_0 of its test
    vector 2. Signed with RFC6979, like a node's signmessage."""
    mint_pubkey = "0305299ebc7d5301da5ff64350c558d2daf9933445e611574474024d10d30f826a"
    spent = "aad3a0e36c083eb0d2d92ec0860977dc46d10c952f31830e6443b1faa1997634"
    note = "01fee34e378bf66de6afa1bfa6e30f5c89551fd92bc1b089dca93c52b7ab61bc"

    digest_1000 = lightning_signed_message_digest(f"LNURLcash:rotate:1000:{spent}:{note}")
    assert digest_1000.hex() == "0d60b3e73e340395bccbd3b87a2210dc6c949df3cd6f0c74d256349b2b989cc1"
    sig_1000 = (
        "3ef03201d14594c51de4d5372f57036b29c0fed94f88c804fc6b69198a40155d4d2969bfef7a9226f1fbcb15246a1be0"
        "c2f8c3cc2d0983e0c3e67e422c7c807001"
    )
    assert verify_rotation(mint_pubkey, spent, note, 1000, sig_1000)
    assert bech32m.encode_cr1(1000, bytes.fromhex(sig_1000)) == (
        "cr10n18mcryqw3gk2v280y65mj74crdv5uplkef7yvsp8udd53nzjqz4w562tfhlhh4y3x78auk9fydgd7pshcc0xz6zvrurp7vljz937gquqp09cjwz"
    )

    digest_21m = lightning_signed_message_digest(f"LNURLcash:rotate:21000000:{spent}:{note}")
    assert digest_21m.hex() == "2ee0ec8c1361fd0e1613d62575dd3e790cd4480ac699af7c852eb9819a29db05"
    sig_21m = (
        "6c329cc404f152e5d59d3b737ea9a347e718b676475679d13db0b9112bdfe4064bae7416c87918ad030423c9dcf49139"
        "94e83eb278c6ca9e955292220cc109ac01"
    )
    assert verify_rotation(mint_pubkey, spent, note, 21000000, sig_21m)
    assert bech32m.encode_cr1(21000000, bytes.fromhex(sig_21m)) == (
        "cr210u1dsefe3qy79fwt4va8deha2drgln33dnkgat8n5fakzu3z27lusryhtn5zmy8jx9dqvzz8jwu7jgnn98g86e833k2n6249y3zpnqsntqpu0dlaz"
    )

    # the amount, the direction and the notes themselves are all signed
    assert not verify_rotation(mint_pubkey, spent, note, 21000000, sig_1000)
    assert not verify_rotation(mint_pubkey, note, spent, 1000, sig_1000)
    assert not verify_rotation(mint_pubkey, spent, spent, 1000, sig_1000)
    # and a rotation certificate is never a note's own certificate, for
    # either of the two notes it names
    assert not verify_note(mint_pubkey, note, 1000, sig_1000)
    assert not verify_note(mint_pubkey, spent, 1000, sig_1000)


def test_rotate_returns_a_valid_rotation_certificate(client: TestClient, mint_note, node):
    k1 = mint_note(5000)
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    assert data["r"].startswith("cr50n1")
    assert _certifies_rotation(node.pubkey, k1_id(k1), h, 5000, data["r"])


def test_rotation_certificates_chain_note_to_note(client: TestClient, mint_note, node):
    # what a consignment carries: one certificate per transfer, each one's
    # credited note the next one's burned note
    k1 = mint_note(5000)
    chain = [k1_id(k1)]
    certificates = []
    for _ in range(3):
        new_k1, h = fresh_secret()
        certificates.append(client.get(f"/w/cb?k1={k1}&p1={h}").json()["r"])
        chain.append(bearer_id(h))
        k1 = new_k1
    for index, cr1 in enumerate(certificates):
        signature = bech32m.decode_cr1(cr1)[1].hex()
        assert verify_rotation(node.pubkey, chain[index], chain[index + 1], 5000, signature)
        # never for a step it doesn't describe: not backwards, not the next
        # one, and not a shortcut past a note in between
        assert not verify_rotation(node.pubkey, chain[index + 1], chain[index], 5000, signature)
        if index + 2 < len(chain):
            assert not verify_rotation(node.pubkey, chain[index + 1], chain[index + 2], 5000, signature)
            assert not verify_rotation(node.pubkey, chain[index], chain[index + 2], 5000, signature)


def test_rotation_certificate_does_not_verify_against_another_note(client: TestClient, mint_note, node):
    k1, other_k1 = mint_note(5000), mint_note(5000)
    _, h = fresh_secret()
    _, other_h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    # a look-alike: a note minted on the side, not the one this note became
    assert not _certifies_rotation(node.pubkey, k1_id(k1), other_h, 5000, data["r"])
    # nor from a note that was never burned for it
    assert not _certifies_rotation(node.pubkey, k1_id(other_k1), h, 5000, data["r"])
    assert not _certifies_rotation(node.pubkey, k1_id(k1), h, 5001, data["r"])


def test_rotation_certificate_does_not_verify_against_wrong_pubkey(client: TestClient, mint_note, node):
    k1 = mint_note(5000)
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    wrong_pubkey = PrivateKey().public_key.format(compressed=True).hex()
    assert not _certifies_rotation(wrong_pubkey, k1_id(k1), h, 5000, data["r"])


def test_rotation_certificate_is_not_a_note_certificate(client: TestClient, mint_note, node):
    # the two certificates of one rotate are never interchangeable: a cr1's
    # signature is not a cs1's over either note, and their prefixes differ
    k1 = mint_note(5000)
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    rotation = bech32m.decode_cr1(data["r"])[1].hex()
    assert not verify_note(node.pubkey, bearer_id(h), 5000, rotation)
    assert not verify_note(node.pubkey, k1_id(k1), 5000, rotation)
    assert bech32m.decode_cs1(data["r"]) is None
    assert bech32m.decode_cr1(data["c"]) is None


def test_split_carries_no_rotation_certificate(client: TestClient, mint_note, node):
    # two notes came out of one: neither is "the" note it became, and a
    # holder of either must not be able to pass it off as such
    k1 = mint_note(5000)
    _, h = fresh_secret()
    _, h2 = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&amount=2000&p1={h}&p2={h2}").json()
    assert data["status"] == "OK"
    assert "r" not in data


def test_merge_carries_no_rotation_certificate(client: TestClient, mint_note, node):
    a, b = mint_note(2000), mint_note(3000)
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={a}&k1={b}&p1={h}").json()
    assert data["status"] == "OK"
    assert "r" not in data


def test_melt_carries_no_rotation_certificate(client: TestClient, node, mint_note):
    from tests.conftest import fake_invoice

    k1 = mint_note(5000)
    data = client.get(f"/w/cb?k1={k1}&pr={fake_invoice(5000)}").json()
    assert data == {"status": "OK"}


def test_informational_request_carries_no_rotation_certificate(client: TestClient, mint_note, node):
    k1 = mint_note(5000)
    assert "r" not in client.get(f"/w?k1={k1}").json()


def test_retried_rotate_replays_the_same_rotation_certificate(client: TestClient, mint_note, node):
    # how whoever made a rotation gets its certificate again later: an
    # exact retry, LUD-25's Retrying a mutation - nothing is stored for it
    k1 = mint_note(5000)
    _, h = fresh_secret()
    first = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    second = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    assert second["r"] == first["r"]
    assert _certifies_rotation(node.pubkey, k1_id(k1), h, 5000, second["r"])


@pytest.mark.parametrize("split", [True, False])
def test_retried_split_and_merge_replay_without_a_rotation_certificate(client: TestClient, mint_note, node, split):
    _, h = fresh_secret()
    _, h2 = fresh_secret()
    if split:
        query = f"/w/cb?k1={mint_note(5000)}&amount=2000&p1={h}&p2={h2}"
    else:
        query = f"/w/cb?k1={mint_note(2000)}&k1={mint_note(3000)}&p1={h}"
    first = client.get(query).json()
    second = client.get(query).json()
    assert second == first
    assert "r" not in second


def test_rotation_certificate_is_unaffected_by_mint_fees(client: TestClient, mint_note, node, monkeypatch):
    # a rotate moves a note's whole value: the amount certified is the
    # amount that was burned
    monkeypatch.setattr(settings, "base_fee_msat", 1000)
    k1 = mint_note(5000)
    value = client.get(f"/w?k1={k1}").json()["maxWithdrawable"]
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    assert _certifies_rotation(node.pubkey, k1_id(k1), h, value, data["r"])


def test_rotation_certificate_absent_without_a_funding_source(client: TestClient, mint_note, monkeypatch):
    k1 = mint_note(5000)
    assert client.get(f"/w?k1={k1}").json()["maxWithdrawable"] == 5000
    monkeypatch.setattr(settings, "fundingsource_backend", None)
    _, h = fresh_secret()
    data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    assert data["status"] == "OK"
    assert "r" not in data


def test_rotation_signing_failure_is_swallowed_and_logged(client: TestClient, mint_note, node, monkeypatch, caplog):
    # the rotate itself already happened: an unreachable node must cost the
    # holder a certificate they can ask for again, never the note
    async def _broken_sign_message(message, config):
        raise ConnectionError("node unreachable")

    k1 = mint_note(5000)
    new_k1, h = fresh_secret()
    assert client.get(f"/w?k1={k1}").json()["maxWithdrawable"] == 5000
    monkeypatch.setattr("lnurl_mint.signing.sign_message", _broken_sign_message)
    with caplog.at_level(logging.WARNING):
        data = client.get(f"/w/cb?k1={k1}&p1={h}").json()
    assert data == {"status": "OK"}
    assert any("sign_rotation" in r.message and "node unreachable" in r.message for r in caplog.records)
    assert client.get(f"/w?k1={new_k1}").json()["maxWithdrawable"] == 5000
