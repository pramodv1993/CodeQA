#
FROM python:3.10.4

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY . /code/

EXPOSE 8001
#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--reload", "--port", "8001"]