FROM tensorflow/tensorflow:latest-gpu
ADD . /developer
LABEL maintainer="haverothgabriel@gmail.com"
# VOLUME "/test" "/home/gabs/Software_Projetos/Codigos_Mestrado/deep_egomotion_radar_heatmap/" 
ARG USERNAME=gabs
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME  \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME 
    # && apt-get install python3-scipy -y sudo 

USER $USERNAME

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir transformations \
    && pip install --no-cache-dir scipy==1.5.2 \
    && pip install --no-cache-dir pandas \
    && pip install --no-cache-dir matplotlib \
    && pip install --no-cache-dir scikit-learn