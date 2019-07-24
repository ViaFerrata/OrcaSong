FROM python:3.6

ENV INSTALL_DIR /orcasong
ADD . $INSTALL_DIR
RUN cd $INSTALL_DIR && make install
WORKDIR /orcasong
ENTRYPOINT /bin/bash
