FROM ollama/ollama AS ollama_build

FROM python:3.10-slim
COPY --from=ollama_build /bin/ollama /bin/ollama

WORKDIR /usr/src/app
# COPY . .

RUN apt update
RUN apt-get update
RUN apt-get install gnupg curl -y
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

RUN pip install --no-cache-dir gradio
RUN pip install --upgrade google-cloud-aiplatform
RUN pip install pandas tqdm google-auth openai azure-identity ollama dotenv

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
# ENV GOOGLE_APPLICATION_CREDENTIALS='./application_default_credentials.json'

CMD ["bash"]