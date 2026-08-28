"""
Gunicorn configuration. The one thing this adds beyond gunicorn's
defaults: cleaning up a worker's Prometheus metrics files when that
worker exits (restart, crash, deploy, etc). Without this hook, the
multiprocess metrics directory (PROMETHEUS_MULTIPROC_DIR) accumulates
files from dead workers forever, and Prometheus's view of, e.g.,
readmission_api_requests_total keeps counting requests from processes
that no longer exist.
"""
from prometheus_client import multiprocess


def child_exit(server, worker):
    multiprocess.mark_process_dead(worker.pid)
