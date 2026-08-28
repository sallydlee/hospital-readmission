# Monitoring — Hospital Readmission API

Prometheus + Grafana + Alertmanager wired up against the real `api/`
Flask service.

## What was added / changed in `api/`

| File | Change |
|---|---|
| `app.py` | Instrumented with Prometheus metrics (request count, latency, prediction distribution, model-loaded status) + a `/metrics` endpoint |
| `gunicorn.conf.py` | **New.** Cleans up a worker's metrics when it exits, required for multiprocess mode |
| `requirements.txt` | Added `prometheus-client` |
| `Dockerfile` | Sets `PROMETHEUS_MULTIPROC_DIR`, uses `gunicorn.conf.py`, worker count now matches `Procfile` (2) |
| `Procfile` | Now also loads `gunicorn.conf.py` |

**Why multiprocess mode matters here specifically:** your `Procfile` already
runs `--workers 2`. Two gunicorn workers are two separate OS processes;
without multiprocess mode, each keeps a private copy of every metric, and
whichever worker happens to answer a given Prometheus scrape only reports
its own half of the traffic — counts would be silently wrong. This was
tested locally with 2 real workers before being included here: firing 11
prediction requests across both workers correctly summed to 11 in
`/metrics`, not 5-and-6 reported separately.

## What's in `monitoring/`

```
monitoring/
├── docker-compose.yml
├── prometheus/
│   ├── prometheus.yml       # scrape config
│   └── alert_rules.yml      # the 4 alert rules
├── alertmanager/
│   └── alertmanager.yml     # where firing alerts get routed
└── grafana/
    ├── provisioning/        # auto-configures data sources + dashboard on startup
    └── dashboards/
        └── readmission-dashboard.json
```

## Run it

```bash
cd monitoring
docker compose up --build
```

This builds `../api`'s existing Dockerfile as the `api` service, plus
Prometheus, Alertmanager, and Grafana.

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000 (`admin` / `admin`)

## Checklist against your requirements

**✅ Prometheus scrapes metrics from the Flask API**
`prometheus/prometheus.yml` scrapes `api:8000/metrics` every 5s. Confirm
at http://localhost:9090/targets — `hospital-readmission-api` should show
`UP`.

**✅ Grafana dashboard: API request volume and latency**
Panels 1–2 in the dashboard: request rate by endpoint/status, and p50/p95/p99
latency on `/predict`.

**✅ Grafana dashboard: Prediction distribution**
Panel 4 (high-risk vs low-risk rate over time) and panel 5 (histogram of
the raw predicted probability, so you can see the shape of the model's
output, not just the thresholded split).

**✅ Grafana dashboard: Error rates**
Panel 3: % of requests returning 5xx (and 4xx separately, since those mean
something different — bad client input vs. a real server problem).

**✅ Alerts for API downtime and abnormal error rates**
`prometheus/alert_rules.yml` defines:
- `ReadmissionAPIDown` — fires if Prometheus can't scrape the API for 30s
- `ReadmissionAPIHighErrorRate` — fires if >5% of requests return 5xx, sustained 1 minute
- `ReadmissionAPIModelNotLoaded` — bonus: fires if the model failed to load even though the process is up (a failure mode your `/health` endpoint already detects; this makes it pageable too)
- `ReadmissionAPIHighLatency` — bonus: fires if p95 latency exceeds 1s for 2+ minutes

## Testing that the alerts actually fire

**Downtime:**
```bash
docker compose stop api
# wait ~30s, then check http://localhost:9090/alerts - ReadmissionAPIDown should go from "Pending" to "Firing"
docker compose start api
```

**Abnormal error rate:** the real API only returns 500 on a genuine
scoring failure, which is hard to trigger on demand. Easiest way to test
the alert path itself: temporarily point `MODEL_PATH` at a file that
doesn't exist, which makes every request hit the `model is None` → 503
branch:
```bash
# in monitoring/docker-compose.yml, under the api service's environment:
- MODEL_PATH=model/does-not-exist.joblib
docker compose up --build api
# generate some traffic against /predict, then check http://localhost:9090/alerts
```
(This also happens to trigger `ReadmissionAPIModelNotLoaded` at the same
time, which is a more realistic version of the same underlying failure.)
Revert the `MODEL_PATH` override afterward.

**Where alerts show up:** http://localhost:9090/alerts (Prometheus's own
view), http://localhost:9093 (Alertmanager), or Grafana's own left-nav
Alerting page (reads from the Alertmanager data source that's already
provisioned).

## Wiring up real notifications

Right now `alertmanager/alertmanager.yml` has an empty receiver — alerts
fire and are visible in the UIs above, but nothing pages anyone. Uncomment
and fill in one of the `slack_configs` / `email_configs` / `webhook_configs`
blocks in that file with your real credentials to change that.

## A note unrelated to monitoring, worth flagging

While testing locally, loading `api/model/model.joblib` printed
`InconsistentVersionWarning` — it was pickled with scikit-learn 1.9.0,
but `requirements.txt` pins `scikit-learn==1.8.0`. It still loaded and
predicted successfully in testing, but a version mismatch like this is
worth resolving (either re-pin to 1.9.0 or re-export the model under
1.8.0) since silent behavior differences across sklearn versions are a
real risk for a model already in production.
