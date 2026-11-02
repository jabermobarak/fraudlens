FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -u 10001 -m analyst
COPY --chown=analyst:analyst . .
USER analyst
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]
