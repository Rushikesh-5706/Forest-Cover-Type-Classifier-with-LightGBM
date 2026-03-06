FROM python:3.11-slim

LABEL maintainer="rushi5706"
LABEL description="Forest Cover Type Classifier - LightGBM"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data outputs/models outputs/figures outputs/reports

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''", \
     "--notebook-dir=/app/notebook"]
