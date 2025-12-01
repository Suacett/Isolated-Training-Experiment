# Project: Proxmox AI Stock Predictor (Closed-Loop Paper Trading)

## 1. Infrastructure & Hardware (CRITICAL)
- **Environment:** Docker (via `docker-compose`).
- **Host:** Proxmox Server (128GB RAM, Ryzen 9 3900X).
- **GPU Acceleration:**
  - The system possesses either an NVIDIA T400 or RTX 4080 Super.
  - **NEVER** hardcode device IDs (e.g., `cuda:0`).
  - **ALWAYS** use dynamic detection: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
  - **Boot Check:** The Backend service must log the detected GPU name on startup.

## 2. Data Architecture
- **Database:** PostgreSQL with TimescaleDB extension (running in Docker).
- **Data Source:** Alpaca Markets (Free Tier).
  - **Secrets:** API Keys must be loaded from `Config` (via `config_loader.py`). NEVER commit keys.
- **Stock List:** Refer to `stock_list.txt` or `stockList.csv` in the legacy files for the initial watchlist.

## 3. Refactoring Strategy
- **Intrinsic Value:** Port logic from `suacett/intrinsic-value-monitor/1-produce_data.ipynb` into a stateless Python class `IntrinsicCalculator`.
- **LSTM Predictor:** Port logic from `suacett/lstm_ai_stock_predictor/forecasting_backtest_Predictor.py`.
  - Decouple data scraping from model training.
  - Ensure the LSTM model accepts the dynamic `device` argument.

## 4. UI/UX Guidelines (Next.js + Tailwind)
- **Theme:** Dark Mode (Financial Dashboard aesthetic).
- **Layout:** Sidebar navigation + Main Content Grid.
- **Components:**
  - `TickerTape`: Scrolling summary of watched stocks.
  - `PredictionCard`: Displays [Current Price] vs [Intrinsic Value] vs [LSTM Prediction] vs [Confidence Score].
  - `EngineLogs`: A real-time view of the Docker container logs (to show training progress).
- **Visuals:** Use Green (`#10B981`) for Bullish/Buy, Red (`#F43F5E`) for Bearish/Sell.

## 5. Coding Standards (TDD)
1. **Plan:** Outline the file structure before writing code.
2. **Test:** Write a failing `pytest` unit test based on the expected Input/Output.
3. **Implement:** Write minimum code to pass.
4. **Modularity:** No files over 200 lines. Break logic into `services/`, `models/`, `utils/`.

---

## 6. MASTER PROJECT CHECKLIST
**Phase 1: Infrastructure**
- [ ] **Task 1.1:** Initialize `docker-compose.yml` (Postgres/TimescaleDB, Python Backend, Node Frontend).
- [ ] **Task 1.2:** Configure NVIDIA Container Toolkit in Docker Compose for dynamic GPU pass-through.
- [ ] **Task 1.3:** Set up `.env` template and `requirements.txt`.

**Phase 2: Data & Backend (Python/FastAPI)**
- [ ] **Task 2.1:** Create `db_manager.py` with TimescaleDB schema for OHLCV data.
- [x] **Task 2.2:** Create `data_ingest.py` for Alpaca Markets (Stock & Crypto support).
- [ ] **Task 2.3:** **Refactor Intrinsic Logic:** Port `1-produce_data.ipynb` to `IntrinsicValueCalculator` class.
- [ ] **Task 2.4:** **Test Intrinsic Logic:** Verify output against `HistoricalPrices.csv`.
- [ ] **Task 2.5:** **Refactor LSTM Logic:** Port `forecasting_backtest_Predictor.py` to `LSTMModel` class.
- [ ] **Task 2.6:** **GPU Verification:** Add startup log to print "Running on [Device Name]".

**Phase 3: Frontend (Next.js)**
- [ ] **Task 3.1:** Scaffold Next.js app with Tailwind CSS (Dark Mode).
- [ ] **Task 3.2:** Build `TickerTape` component.
- [ ] **Task 3.3:** Build `PredictionCard` component with Chart.js/Recharts.
- [ ] **Task 3.4:** Build `LogViewer` component (streaming logs from backend).
- [ ] **Task 3.5:** Connect Frontend to Backend API.
