FROM python

RUN pip3 install lightgbm matplotlib

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

COPY . .
