# Murf AI Voice Agent Challenge – 10 Day Project

A clean and professional README documenting my journey in building an AI-powered voice agent for the **Murf AI Voice Agent Challenge 2025**.

---

## Overview

This project explores real-time conversational AI using:

* Murf Falcon (fast TTS)
* LiveKit (audio streaming and signaling)
* Python backend
* Next.js/React frontend

The goal is to build a complete browser-based voice agent capable of listening, understanding, and responding with natural-sounding speech.

---

## Features

* Real-time microphone streaming
* Low-latency AI responses
* Murf Falcon text-to-speech integration
* Duplex voice flow
* Modular backend architecture

---

## Tech Stack

### Frontend

* Next.js / React
* TypeScript
* LiveKit Web SDK

### Backend

* Python
* FastAPI / WebSockets
* Murf Falcon TTS
* Optional STT providers (Whisper / Deepgram)

---

## Project Structure

```
root
├── backend
│   ├── src
│   ├── requirements.txt
│   └── .env.example
├── frontend
│   ├── src
│   ├── public
│   └── .env.example
└── README.md
```

---

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Backend Setup

```bash
cd backend
uv sync
cp .env.example .env.local
```

Add required keys:

* LIVEKIT_URL
* LIVEKIT_API_KEY
* LIVEKIT_API_SECRET
* MURF_API_KEY
* STT provider keys (if used)

Run backend:

```bash
uv run python src/agent.py dev
```

### 3. Frontend Setup

```bash
cd frontend
pnpm install
cp .env.example .env.local
pnpm dev
```

Visit the app at:

```
http://localhost:3000
```

---

## Day-wise Progress

### Day 1 – Environment Setup & First Voice Interaction

* Backend and frontend configured
* LiveKit Cloud connected
* Murf Falcon integrated
* First successful test conversation completed

*(Additional days will be documented as the project continues.)*

---

## Environment Variables

Do not commit real environment files.
Ignored by default:

```
*.env
*.env.local
.env*
```

Required keys include:

* Murf Falcon API key
* LiveKit API credentials
* STT provider keys (optional)

---

## Contributing

Suggestions and improvements are welcome.

---

## License

Released under the MIT License.

---

## Credits

* Murf AI for TTS
* LiveKit for real-time infrastructure
* Open-source community for tools and support
