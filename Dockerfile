FROM python:3.6

ENV INSTALL_DIR /orcasong

COPY . $INSTALL_DIR
RUN pip install --upgrade --no-cache-dir pip setuptools wheel
RUN cd $INSTALL_DIR && make install
RUN cd / && rm -rf $INSTALL_DIR
