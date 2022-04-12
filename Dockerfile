FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


WORKDIR /stage/

# Copy the files to /stage
COPY setup.py ./
COPY nlp_gym/ ./nlp_gym
COPY demo_scripts/ ./demo_scripts
RUN pip install -e .