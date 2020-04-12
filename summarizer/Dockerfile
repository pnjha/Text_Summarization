FROM ubuntu:18.04

WORKDIR /Summarizer
COPY . .

RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install -r requirements.txt

ENTRYPOINT python3 app.py
