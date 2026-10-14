FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
COPY src ./src
RUN uv sync --no-dev --no-editable
ENTRYPOINT ["uv", "run", "ragscore"]
