from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from urllib.parse import urljoin

from prometheus_client.parser import text_string_to_metric_families

from kvstat.engines import vllm
from kvstat.errors import ConfigError

log = logging.getLogger(__name__)

# One Prometheus sample: (metric name, labels, value).
Sample = tuple[str, dict[str, str], float]

# Called with (scrape time, samples) after every successful scrape.
ScrapeCallback = Callable[[float, list[Sample]], None]


def parse(text: str) -> list[Sample]:
    """Return the samples of every vllm: metric family and drop the process and runtime ones."""
    out: list[Sample] = []
    for family in text_string_to_metric_families(text):
        if not family.name.startswith(vllm.METRIC_PREFIX):
            continue
        out.extend((s.name, dict(s.labels), float(s.value)) for s in family.samples)
    return out


def fetch(url: str, timeout: float = 5.0) -> str:
    """GET one http(s) URL and return the body as text.

    Raises:
        ConfigError: the URL is not http or https.
    """
    if not url.startswith(("http://", "https://")):
        raise ConfigError(f"not an http(s) URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - scheme checked above
        return str(resp.read().decode("utf-8"))  # urlopen is untyped; pin the boundary


def fetch_server_info(server: str, timeout: float = 5.0) -> dict[str, object]:
    """Collect the vLLM version, model name and KV cache configuration for a capture header.

    Reads /version and /metrics. An entry stays None when its request fails.
    """
    info: dict[str, object] = {"vllm_version": None, "model_name": None, "cache_config": None}
    try:
        info["vllm_version"] = json.loads(fetch(urljoin(server, vllm.VERSION_PATH), timeout)).get(
            "version"
        )
    except (OSError, ValueError) as exc:
        log.warning("server info: %s unavailable: %s", vllm.VERSION_PATH, exc)
    try:
        for name, labels, _ in parse(fetch(urljoin(server, vllm.METRICS_PATH), timeout)):
            if info["model_name"] is None and vllm.MODEL_LABEL in labels:
                info["model_name"] = labels[vllm.MODEL_LABEL]
            if name == vllm.CACHE_CONFIG_METRIC:
                info["cache_config"] = {k: v for k, v in labels.items() if k != "engine"}
    except (OSError, ValueError) as exc:
        log.warning("server info: %s unavailable: %s", vllm.METRICS_PATH, exc)
    return info


class MetricsPoller:
    """Scrapes /metrics on its own thread, once per interval, and calls on_scrape with each result.

    start() launches a daemon thread that scrapes until close() is called. When the endpoint is
    unreachable, the poller logs one warning and keeps trying.

    Args:
        url: Full URL of the metrics endpoint.
        on_scrape: Called with (scrape time, samples) after every successful scrape. kvstat
            record passes CaptureWriter.write_metrics.
        interval: Seconds between scrapes.
        timeout: Seconds to wait for one HTTP response.

    Attributes:
        scrapes: Successful scrapes so far.
        failures: Failed scrapes so far.
    """

    def __init__(
        self, url: str, on_scrape: ScrapeCallback, *, interval: float = 1.0, timeout: float = 5.0
    ) -> None:
        self._url = url
        self._on_scrape = on_scrape
        self._interval = interval
        self._timeout = timeout
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kvstat-metrics", daemon=True)
        self.scrapes = 0
        self.failures = 0

    def start(self) -> MetricsPoller:
        """Start the scrape thread and return self."""
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the scrape thread and wait for it."""
        self._stop.set()
        self._thread.join(timeout=self._timeout + 1)

    def _run(self) -> None:
        """Thread body: scrape, hand off, sleep, repeat until stopped."""
        down = False
        while not self._stop.is_set():
            now = time.time()
            try:
                samples = parse(fetch(self._url, self._timeout))
            except (OSError, ValueError) as exc:
                self.failures += 1
                if not down:
                    log.warning("metrics: %s unreachable: %s", self._url, exc)
                    down = True
            else:
                if down:
                    log.info("metrics: %s reachable again", self._url)
                    down = False
                self.scrapes += 1
                self._on_scrape(now, samples)
            self._stop.wait(self._interval)
