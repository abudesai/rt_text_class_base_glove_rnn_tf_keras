FROM tensorflow/tensorflow:2.11.0-gpu

RUN apt-get -y update && \
        apt-get -y install gcc mono-mcs && \
        apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt .
RUN pip install -r requirements.txt 

ARG embed_dim=50 
ENV embed_dim=${embed_dim} 
ENV embed_file_name=glove.6B."$embed_dim"d.txt 

# download and unzip the glove embeddings file
RUN wget -P /tmp http://nlp.stanford.edu/data/glove.6B.zip
RUN unzip /tmp/glove.6B.zip -d /tmp/


COPY app ./opt/app
WORKDIR /opt/app

# move embeddings into the location where model looks for it
RUN mv /tmp/"$embed_file_name" /opt/app/Utils/pretrained_embed/
RUN echo "export embed_dim=${embed_dim}" >> /root/.bashrc #To keep env variable on the system after restarting
RUN echo "export embed_file_name=${embed_file_name}" >> /root/.bashrc #To keep env variable on the system after restarting


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"


RUN chmod +x train &&\
    chmod +x predict &&\
    chmod +x serve 


# USER 1001