FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# pip robustness
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=10
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/
RUN pip install --timeout 180 --retries 10 --no-cache-dir -r requirements.txt -i https://pypi.org/simple

COPY serve.py /app/serve.py
COPY model/ /app/model/

EXPOSE 8000

CMD ["python","-m","uvicorn","serve:app","--host","0.0.0.0","--port","8000"]
