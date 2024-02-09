#set the image
FROM python:3.11-slim 

RUN pip install -U pip
RUN pip install pipenv


#due to an error this following line was added \
#However gunicorn is recommended to install while building the pipenv
RUN pip install gunicorn


WORKDIR /app


COPY ["Pipfile","Pipfile.lock", "./"]


RUN pipenv install --system --deploy


EXPOSE 9696

COPY ["predict.py","extra_trees_reg.bin","./"]

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]