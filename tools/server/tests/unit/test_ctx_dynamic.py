import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.offline = False
    server.n_ctx = 1024          # max tier (cap)
    server.n_slots = 1
    server.n_predict = 64


def test_ctx_dynamic_small_request_starts_small():
    # With ctx-dynamic and a small min tier, a tiny request should succeed and
    # the server should report a small context (the smallest tier), not the cap.
    global server
    server.ctx_dynamic = True
    server.ctx_dynamic_min = 256
    server.n_ctx = 1024
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Hello",
    })
    assert res.status_code == 200
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["default_generation_settings"]["n_ctx"] <= 256


def test_ctx_dynamic_large_request_grows():
    # A prompt that needs more than the smallest tier should trigger a grow and
    # still succeed (no "exceeds context" error, not truncated).
    global server
    server.ctx_dynamic = True
    server.ctx_dynamic_min = 256
    server.n_ctx = 1024
    server.start()
    long_prompt = "word " * 300  # exceeds the 256 tier
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": long_prompt,
    })
    assert res.status_code == 200
    assert res.body["truncated"] is False
