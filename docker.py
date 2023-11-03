FROM 3.10.12-slim


COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["predict.py", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:9696", "predict:app"]