FROM nvcr.io/nvidia/pytorch:23.07-py3
ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

ARG USER=docker
ARG UID=1000
ARG GID=1000
RUN apt-get update -y --fix-missing && apt install htop -y

RUN useradd --system --create-home --home-dir /home/${USER} \
    --shell /bin/bash --gid root -G sudo --uid ${UID} ${USER}
RUN echo ${USER}:docker | chpasswd
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="${PATH}:/home/${USER}/.local/bin"

# Repo requirements
RUN python -m pip install -U pip
COPY repo_requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# Add zsh ricing
USER root
RUN apt install -y zsh wget && chsh -s $(which zsh)
USER ${USER}
# Default powerline10k theme, no plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)"
