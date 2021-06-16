FROM jupyter/datascience-notebook

COPY requirements.txt .
RUN pip install -r requirements.txt
