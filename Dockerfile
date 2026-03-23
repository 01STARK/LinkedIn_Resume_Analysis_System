FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

EXPOSE 8080

CMD ["streamlit", "run", "app1.py", "--server.port=8080", "--server.address=0.0.0.0"]