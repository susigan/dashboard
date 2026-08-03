FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Criar pasta de cache do Streamlit
RUN mkdir -p /root/.streamlit

# Config Streamlit
RUN echo "[server]\nheadless = true\nport = 8501\nenableXsrfProtection = false\n[client]\nshowErrorDetails = true" > /root/.streamlit/config.toml

# Expor porta
EXPOSE 8501

# Comando para rodar
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
