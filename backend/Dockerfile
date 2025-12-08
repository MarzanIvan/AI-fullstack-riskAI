FROM python:3.13-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install python-multipart
RUN pip install uvicorn[standard]

COPY . .

CMD ["python", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

