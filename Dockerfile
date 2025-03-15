FROM python:3.12

WORKDIR /app

ADD requirements.txt .
ADD tier_analysis.py .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "./tier_analysis.py" ]