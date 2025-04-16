# Personal Budget API

An AI-driven personal finance application that helps users manage expenses using envelope budgeting and machine learning-powered forecasts.

## 🚀 Features

- Create and manage budget envelopes
- Forecast category-wise or total expenses using an LSTM model
- RESTful API backend built with Node.js and Express
- AI microservice powered by Python (Flask + TensorFlow)
- PostgreSQL database integration
- Modular, scalable, and production-ready codebase

---

## 📂 Project Structure

```
personal-budget-api/
│
├── ai-service/              # Python ML service (LSTM)
│   ├── lstm_model.py        # Training & prediction logic
│   ├── app.py               # Flask API
│   └── ...
│
├── backend/                # Node.js backend
│   ├── controllers/         
│   ├── routes/              
│   └── index.js
│
├── database/               # DB setup & data
│   ├── init.sql             # PostgreSQL schema
│   ├── populate.py          # Loads data from CSV
│   └── 11 march 2025.csv    # Expense dataset (CSV)
│
├── .gitignore
├── README.md
└── ...
```

---

## ⚙️ Prerequisites

- Node.js (v18+)
- Python (>=3.8)
- PostgreSQL (>=13)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Personal-Budget-API.git
cd Personal-Budget-API
```

### 2. Setup the PostgreSQL database
```bash
cd database
psql -U postgres -f init.sql
python populate.py  # Ensure your PostgreSQL credentials are correct in the script
```
> The dataset `11 march 2025.csv` is provided in the `database/` folder.

### 3. Setup the AI service (Flask)
```bash
cd ai-service
python -m venv venv
venv\Scripts\activate    # Windows
# or source venv/bin/activate for macOS/Linux
pip install -r requirements.txt

# ✅ IMPORTANT: Train the model before running the Flask server
python lstm_model.py

# Then run the API
python app.py  # Runs on port 5000 by default
```

### 4. Start the Node.js backend
```bash
cd backend
npm install
node index.js  # Runs on port 3000
```

---

## 📡 API Usage

### Create Envelope
```http
POST /envelopes
Content-Type: application/json
{
  "name": "Groceries",
  "amount": 200
}
```

### Get All Envelopes
```http
GET /envelopes
```

### Forecast Total or Category Expenses
```http
POST /envelopes/forecast
Content-Type: application/json
{
  "category": "Groceries"  // Or "total" for all
}
```

---

## 🧠 Forecasting Model
- Model: CNN + LSTM hybrid
- Forecast window: 30 days
- Evaluation metrics: MSE, MAE
- Features: Moving averages, std dev, weekday/month sin-cos encoding, etc.

Forecast results are delivered in real currency values.

---

## 🧪 Testing

The backend was tested using **Postman** with different request scenarios:
- Valid/invalid envelope creation
- Transfers between envelopes
- Forecasting requests for valid and invalid categories

Model predictions were compared against test data split from historical records.

---

## 🛠️ Tech Stack

| Layer     | Technology             |
|-----------|------------------------|
| Backend   | Node.js, Express       |
| AI        | Python, TensorFlow, Flask |
| Database  | PostgreSQL             |
| API Comm  | REST (HTTP/JSON)       |

---

## 📄 License
MIT

---

## 👨‍💻 Author
Jannatul Islam Fahim — [GitHub](https://github.com/JIFAHIM175)
