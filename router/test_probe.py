"""
test_probe.py - the root probe's verdicts when no supervisor answers. Run: pytest -q
"""

import asyncio

import httpx2
from config import Settings
from service import Router


def router_with(handler) -> Router:
    settings = Settings(_env_file=None, cloud_api_key="k", local_base_url="http://pi:9990")
    return Router(settings, httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


def raising(exc):
    def handler(request):
        raise exc

    return handler


def test_root_answering_is_healthy():
    r = router_with(lambda req: httpx2.Response(200, json={"data": []}))
    assert asyncio.run(r.probe_root()) == "healthy"


def test_nothing_on_the_wire_is_unreachable_not_healthy():
    r = router_with(raising(httpx2.ConnectTimeout("no route")))
    assert asyncio.run(r.probe_root()) == "unreachable"
    r = router_with(raising(httpx2.ConnectError("refused")))
    assert asyncio.run(r.probe_root()) == "unreachable"


def test_a_busy_root_keeps_its_last_verdict():
    r = router_with(raising(httpx2.ReadTimeout("generating")))
    r.st.cluster_status = "degraded"
    assert asyncio.run(r.probe_root()) == "degraded"
    r.st.cluster_status = "unknown"
    assert asyncio.run(r.probe_root()) == "healthy"
