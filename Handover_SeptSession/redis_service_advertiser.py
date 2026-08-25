"""Advertise the local Redis broker over mDNS/DNS-SD.

This sidecar is intentionally independent of Redis and of the data publishers.
It registers the service only while the configured Redis endpoint answers PING,
and refreshes the registration if the computer's network address changes.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import signal
import socket
import sys
import threading
from collections.abc import Callable, Sequence

import redis
from zeroconf import IPVersion, ServiceInfo, Zeroconf


SERVICE_TYPE = "_amplify-redis._tcp.local."
DEFAULT_SERVICE_NAME = "Amplify Redis"
DEFAULT_REDIS_HOST = "127.0.0.1"
DEFAULT_REDIS_PORT = 6379
DEFAULT_CHECK_INTERVAL = 2.0
DEFAULT_REDIS_TIMEOUT = 1.0

LOG = logging.getLogger("redis-advertiser")


def usable_ipv4(value: str) -> str | None:
    """Return a normalized, advertiseable IPv4 address or None."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    if not isinstance(address, ipaddress.IPv4Address):
        return None
    if address.is_loopback or address.is_link_local or address.is_multicast:
        return None
    if address.is_unspecified:
        return None
    return str(address)


def discover_primary_ipv4() -> str | None:
    """Find the IPv4 address selected by the OS for ordinary network traffic.

    Connecting a UDP socket chooses a route but sends no packets. Hostname
    resolution is retained as a fallback for networks without a default route.
    """
    for destination in (("8.8.8.8", 80), ("1.1.1.1", 80)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(destination)
            address = usable_ipv4(sock.getsockname()[0])
            if address:
                return address
        except OSError:
            pass
        finally:
            sock.close()

    try:
        candidates = socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM
        )
    except OSError:
        return None

    for candidate in candidates:
        address = usable_ipv4(candidate[4][0])
        if address:
            return address
    return None


def local_server_name() -> str:
    """Return a safe single-label .local hostname for the DNS-SD record."""
    raw_name = socket.gethostname().strip() or "amplify-redis"
    label = "".join(
        character if character.isalnum() or character == "-" else "-"
        for character in raw_name
    ).strip("-")
    return f"{(label or 'amplify-redis')[:63]}.local."


class RedisHealthChecker:
    """Small Redis PING probe with short, bounded timeouts."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._client = redis.Redis(
            host=host,
            port=port,
            db=0,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        )

    def is_healthy(self) -> bool:
        try:
            return bool(self._client.ping())
        except (redis.RedisError, OSError):
            return False


class MdnsRedisAdvertiser:
    """Own the zeroconf registration and its network-bound socket."""

    def __init__(self, service_name: str, port: int) -> None:
        self._service_name = service_name
        self._port = port
        self._zeroconf: Zeroconf | None = None
        self._service_info: ServiceInfo | None = None
        self.registered_address: str | None = None

    def start(self, address: str) -> None:
        if address == self.registered_address:
            return
        self.stop()

        full_name = f"{self._service_name}.{SERVICE_TYPE}"
        service_info = ServiceInfo(
            SERVICE_TYPE,
            full_name,
            addresses=[socket.inet_aton(address)],
            port=self._port,
            properties={
                b"role": b"engagement-broker",
                b"protocol": b"redis",
                b"version": b"1",
            },
            server=local_server_name(),
        )
        zeroconf = Zeroconf(interfaces=[address], ip_version=IPVersion.V4Only)

        try:
            zeroconf.register_service(service_info, allow_name_change=True)
        except Exception:
            zeroconf.close()
            raise

        self._zeroconf = zeroconf
        self._service_info = service_info
        self.registered_address = address
        LOG.info(
            "Advertising %s at %s:%d as %s",
            SERVICE_TYPE,
            address,
            self._port,
            service_info.name,
        )

    def stop(self) -> None:
        zeroconf = self._zeroconf
        service_info = self._service_info
        old_address = self.registered_address
        self._zeroconf = None
        self._service_info = None
        self.registered_address = None

        if zeroconf is None:
            return
        try:
            if service_info is not None:
                zeroconf.unregister_service(service_info)
        except Exception as error:
            LOG.warning("Could not withdraw the mDNS service cleanly: %s", error)
        try:
            zeroconf.close()
        except Exception as error:
            LOG.warning("Could not close the mDNS socket cleanly: %s", error)
        if old_address:
            LOG.info("Stopped advertising Redis at %s:%d", old_address, self._port)


class AdvertiserController:
    """Apply Redis health and address changes to the mDNS registration."""

    def __init__(
        self,
        health_checker: RedisHealthChecker,
        advertiser: MdnsRedisAdvertiser,
        address_provider: Callable[[], str | None],
    ) -> None:
        self._health_checker = health_checker
        self._advertiser = advertiser
        self._address_provider = address_provider
        self._last_state: str | None = None

    def evaluate(self) -> None:
        if not self._health_checker.is_healthy():
            self._advertiser.stop()
            self._log_state(
                "redis-unavailable",
                "Waiting for Redis to answer PING; nothing is being advertised",
            )
            return

        address = self._address_provider()
        if not address:
            self._advertiser.stop()
            self._log_state(
                "network-unavailable",
                "Redis is healthy, but no usable network IPv4 address was found",
            )
            return

        try:
            self._advertiser.start(address)
        except Exception as error:
            self._advertiser.stop()
            self._log_state(
                f"registration-error:{error}",
                f"Could not advertise on {address}: {error}",
                level=logging.WARNING,
            )
            return
        self._last_state = f"advertising:{address}"

    def close(self) -> None:
        self._advertiser.stop()

    def _log_state(self, state: str, message: str, level: int = logging.INFO) -> None:
        if state != self._last_state:
            LOG.log(level, message)
            self._last_state = state


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def port_number(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def ipv4_argument(value: str) -> str:
    address = usable_ipv4(value)
    if not address:
        raise argparse.ArgumentTypeError("must be a non-loopback IPv4 address")
    return address


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advertise the local Redis service using mDNS/DNS-SD."
    )
    parser.add_argument("--redis-host", default=DEFAULT_REDIS_HOST)
    parser.add_argument("--redis-port", type=port_number, default=DEFAULT_REDIS_PORT)
    parser.add_argument("--service-name", default=DEFAULT_SERVICE_NAME)
    parser.add_argument(
        "--advertise-address",
        type=ipv4_argument,
        help="IPv4 address to announce instead of detecting the primary interface",
    )
    parser.add_argument(
        "--check-interval",
        type=positive_float,
        default=DEFAULT_CHECK_INTERVAL,
        help=f"seconds between health/address checks (default: {DEFAULT_CHECK_INTERVAL:g})",
    )
    parser.add_argument(
        "--redis-timeout",
        type=positive_float,
        default=DEFAULT_REDIS_TIMEOUT,
        help=f"Redis PING timeout in seconds (default: {DEFAULT_REDIS_TIMEOUT:g})",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    service_name = args.service_name.strip()
    if (
        not service_name
        or "." in service_name
        or "\x00" in service_name
        or len(service_name.encode("utf-8")) > 63
    ):
        parser.error(
            "--service-name must be a non-empty DNS-SD instance label of at most 63 bytes"
        )
    args.service_name = service_name
    return args


def install_signal_handlers(stop_event: threading.Event) -> None:
    def request_stop(signum: int, _frame: object) -> None:
        LOG.info("Stop requested (signal %d)", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    address_provider: Callable[[], str | None]
    if args.advertise_address:
        address_provider = lambda: args.advertise_address
    else:
        address_provider = discover_primary_ipv4

    controller = AdvertiserController(
        RedisHealthChecker(args.redis_host, args.redis_port, args.redis_timeout),
        MdnsRedisAdvertiser(args.service_name, args.redis_port),
        address_provider,
    )
    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    LOG.info(
        "Redis service advertiser started; checking %s:%d",
        args.redis_host,
        args.redis_port,
    )
    LOG.info("Glasses should browse for %s", SERVICE_TYPE)

    try:
        while not stop_event.is_set():
            controller.evaluate()
            stop_event.wait(args.check_interval)
    finally:
        controller.close()
        LOG.info("Redis service advertiser stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
