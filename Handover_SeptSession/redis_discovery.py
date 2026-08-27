"""Discover the Amplify Redis broker advertised over mDNS/DNS-SD.

The hub PC runs ``redis_service_advertiser.py`` (``1B_START_REDIS_ADVERTISER.bat``),
which announces the service ``_amplify-redis._tcp.local.`` with the broker's LAN
IPv4 address + port while Redis answers PING. This lets data clients (the EmotiBit
publisher, the engagement GUI, the AR glasses) find the hub automatically instead
of being told its IP every time.

Usage::

    from redis_discovery import discover_redis_broker
    found = discover_redis_broker(timeout=5.0)   # (host, port) or None
"""

from __future__ import annotations

import threading

# Must match SERVICE_TYPE in redis_service_advertiser.py.
SERVICE_TYPE = "_amplify-redis._tcp.local."


def discover_redis_broker(timeout: float = 5.0):
    """Browse the LAN via mDNS for the advertised Amplify Redis broker.

    Returns ``(host_ip, port)`` for the first advertisement seen within
    ``timeout`` seconds, or ``None`` if nothing is found (advertiser not
    running, different subnet, or mDNS blocked). Never raises — a missing
    ``zeroconf`` package or any browse error degrades to ``None`` so callers
    can fall back to an explicit host/localhost.
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except Exception:
        return None

    result: dict[str, object] = {}
    done = threading.Event()

    def _capture(zc, type_, name) -> None:
        try:
            info = zc.get_service_info(type_, name, timeout=int(timeout * 1000))
        except Exception:
            return
        if not info:
            return
        try:
            addresses = info.parsed_addresses()
        except Exception:
            addresses = []
        ipv4 = next((a for a in addresses if ":" not in a), None)
        if ipv4:
            result["host"] = ipv4
            result["port"] = info.port
            done.set()

    class _Listener:
        def add_service(self, zc, type_, name):
            _capture(zc, type_, name)

        def update_service(self, zc, type_, name):
            _capture(zc, type_, name)

        def remove_service(self, zc, type_, name):
            pass

    zeroconf = Zeroconf()
    try:
        ServiceBrowser(zeroconf, SERVICE_TYPE, _Listener())
        done.wait(timeout)
    except Exception:
        return None
    finally:
        try:
            zeroconf.close()
        except Exception:
            pass

    if "host" in result:
        return str(result["host"]), int(result["port"])  # type: ignore[arg-type]
    return None
