FROM python 3.9-slim
WORKDIR/ app
COPY ./app
COPY requirements.txt/app/requirements.txt
RUN pip install-- no cache-dir-r requirements.txt
RUN pip install jupyter
COPY model3.ipnyb/app/model3.ipynb
EXPOSE 8888
CMD ['jupyter','notebook', "--ip=0.0.0.0", "--port=8888","--nobrowser", "--allow-root"]
docker build -t my-notebook
docker run p8888:8888 my-notebook
