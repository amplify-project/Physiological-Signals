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

import os
import socket
import threading

# Must match SERVICE_TYPE in redis_service_advertiser.py.
SERVICE_TYPE = "_amplify-redis._tcp.local."

# Last-known-good broker, next to this module so every launcher shares it.
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_redis_broker")


def _browse_once(timeout: float):
    """One mDNS browse round. Returns ``(host, port)`` or ``None``. Never raises."""
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


def discover_redis_broker(timeout: float = 6.0, attempts: int = 2):
    """Browse the LAN via mDNS for the advertised Amplify Redis broker.

    Runs up to ``attempts`` browse rounds (returning early on the first hit) to
    ride out dropped multicast packets. Returns ``(host, port)`` or ``None``.
    Never raises — a missing ``zeroconf`` package or any browse error is None.
    """
    per_round = max(1.0, timeout / max(1, attempts))
    for _ in range(max(1, attempts)):
        found = _browse_once(per_round)
        if found:
            return found
    return None


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """Quick TCP connect probe so we never hand back a stale cached broker."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _read_cache():
    try:
        with open(_CACHE_FILE, encoding="utf-8") as fh:
            host, _, port = fh.read().strip().partition(":")
        if host and port:
            return host, int(port)
    except (OSError, ValueError):
        pass
    return None


def _write_cache(host: str, port: int) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
            fh.write(f"{host}:{port}")
    except OSError:
        pass


def resolve_redis_broker(timeout: float = 6.0, use_cache: bool = True):
    """Resolve the broker: mDNS first, then a validated last-known-good cache.

    Returns ``(host, port)`` or ``None``. A successful mDNS discovery is cached
    for next time; if mDNS misses, a cached broker is returned only when it
    still answers a TCP connect (so a moved/dead hub is skipped).
    """
    found = discover_redis_broker(timeout=timeout)
    if found:
        if use_cache:
            _write_cache(*found)
        return found

    if use_cache:
        cached = _read_cache()
        if cached and _reachable(*cached):
            return cached
    return None
