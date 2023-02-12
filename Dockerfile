# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git  \
        git-lfs

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy

COPY ./model /model
ADD download.py .
RUN python3 download.py

#RUN mkdir openjourney
#RUN cd /openjourney
#RUN git lfs clone https://huggingface.co/prompthero/openjourney
##RUN cd ..
# We add the banana boilerplate here

ADD server.py .
EXPOSE 8000

# Add your huggingface auth key here
ENV HF_AUTH_TOKEN="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"


# Add your custom app code, init() and inference()
ADD upload.py .
ADD ai-studio-credential.json .
ADD app.py .

CMD python3 -u server.py