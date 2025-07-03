# Base image with Python 3.11
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
  curl \
  git \
  nodejs \
  npm \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  ffmpeg \
  && apt-get clean

# ---------- Backend setup ----------
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend files (removed duplicates)
COPY backend /app/backend
COPY backend/sd /app/sd
COPY backend/data /app/data
COPY screens/ip.json /app/backend/screens/ip.json

# Make sure Python can find our modules
ENV PYTHONPATH="/app"

# ---------- Frontend setup ----------
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps
RUN npm install -g expo @expo/ngrok@^4.1.0

# Copy remaining files (excluding what's in .dockerignore)
COPY . .

# Expose necessary ports
EXPOSE 8000    
EXPOSE 19000  
EXPOSE 19001   
EXPOSE 19002  
EXPOSE 19006

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["bash", "start.sh"]