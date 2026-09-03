# kvtop

**htop for vLLM's KV cache.**

kvtop attaches to a running [vLLM](https://github.com/vllm-project/vllm) server and shows what is happening inside its KV cache right now, request by request: which requests hold which blocks, which ones are hitting the prefix cache, which one was just preempted, and why TTFT spiked when it happened. Dashboards show the totals; kvtop shows the requests behind them.

It rebuilds this view from vLLM's KV cache event stream (`--kv-events-config`), the same stream KV-aware routers rely on, and keeps cross-checking it against vLLM's own Prometheus metrics. When the two disagree, the screen flags it, so you always know how much to trust what you see.

## What v1 shows

- **KV occupancy map** — who holds how many blocks, how fragmented the pool is, how much headroom is left.
- **Prefix-cache attribution** — which requests ride cached prefixes and which pay full prefill.
- **Preemption timeline** — evictions and recomputes lined up against queue depth and TTFT spikes.
- **Status bar with error bounds** — state age, reconcile drift, dropped batches.
- **Record & replay** — capture the event stream to a file, replay it in the TUI later.

Not a benchmark, autotuner, router, or Grafana replacement. Single vLLM instance first.

## Development

```bash
uv sync
uv run kvtop --help
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
