FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get -y update && \
        apt-get -y install gcc mono-mcs && \
        apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


ENV embed_dim=100 
ENV embed_file_name=glove.6B."$embed_dim"d.txt 
RUN echo 'export embed_dim=$embed_dim' >> /root/.bashrc  #To keep env variable on the system after restarting
RUN echo 'export embed_file_name=$embed_file_name' >> /root/.bashrc #To keep env variable on the system after restarting

# make directory where we will download the pretrained word2vec embeddings
RUN mkdir -p /opt/pretrained_embed
# download and unzip the embeddings file; for referencess please visit https://nlp.stanford.edu/projects/glove/ 
RUN wget -P /opt/pretrained_embed http://nlp.stanford.edu/data/glove.6B.zip
RUN unzip /opt/pretrained_embed/glove.6B.zip -d  /opt/pretrained_embed

COPY ./requirements.txt .
RUN pip install -r requirements.txt 

COPY app ./opt/app
WORKDIR /opt/app

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"


RUN chmod +x train &&\
    chmod +x predict &&\
    chmod +x serve 

RUN chown -R 1000:1000 /opt/app/  && \
    chown -R 1000:1000 /var/log/nginx/  && \
    chown -R 1000:1000 /var/lib/nginx/ && \
    chown -R 1000:1000 /opt/pretrained_embed/

USER 1000