FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Matches Procfile's worker count so metrics behave identically in both
# environments (Elastic Beanstalk via Procfile, container via this
# Dockerfile). See gunicorn.conf.py / app.py for why multiprocess mode
# needs both the shared directory and the child_exit cleanup hook.
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
RUN mkdir -p /tmp/prometheus_multiproc

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "-c", "gunicorn.conf.py", "app:application"]
