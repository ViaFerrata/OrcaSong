FROM python:3.6

ENV INSTALL_DIR /orcasong

RUN pip install --upgrade --no-cache-dir pip setuptools wheel
COPY . $INSTALL_DIR
RUN cd $INSTALL_DIR && make install
RUN cd / && rm -rf $INSTALL_DIR
