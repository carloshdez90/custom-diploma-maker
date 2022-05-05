FROM python:3.9  

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

COPY ./bucket-service-account.json /app/bucket-service-account.json

RUN apt-get update && apt-get install libgl1 -y

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY ./main.py /app/

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]