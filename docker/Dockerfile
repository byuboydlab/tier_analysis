FROM python:3.13.2

WORKDIR /app

ADD ./docker/requirements.txt .
ADD tier_analysis.py .

RUN pip install -r requirements.txt

CMD [ "python", "./tier_analysis.py" ]