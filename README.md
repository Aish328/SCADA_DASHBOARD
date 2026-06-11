# SCADA Intelligence Dashboard

A modern SCADA-style analytics dashboard for monitoring electrical feeder health, anomaly detection, and KPI visualization using FastAPI, Chart.js, and Leaflet.js.

The platform provides real-time operational insights into voltage and current behavior through interactive charts, KPI metrics, and geographic substation mapping.

---

## Features

### Real-Time Monitoring

* Live voltage and current trend visualization
* Dynamic KPI cards
* Interactive feeder and substation filtering
* Geographic substation mapping with Leaflet.js

### KPI & Anomaly Analytics

* Voltage surge and dip detection
* Current surge and drop detection
* Episode duration analysis
* Threshold-based anomaly tracking
* Statistical KPI computation

### Interactive Dashboard

* SCADA-inspired dark UI
* Responsive Chart.js visualizations
* Combined trend analysis
* Duration and anomaly charts
* Dynamic dashboard updates

### Backend Processing

* Automated KPI derivation
* Time-series preprocessing
* Threshold-based event detection
* CSV export support
* REST API architecture using FastAPI

---

## Tech Stack

### Backend

* Python
* FastAPI
* Pandas
* NumPy
* Uvicorn

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js
* Leaflet.js

---

## Project Structure

```plaintext
scada_dashboard/
│
├── main.py
├── data/
├── routes/
├── services/
├── static/
│   ├── css/
│   └── js/
├── templates/
└── README.md
```

---

## KPI Definitions

| KPI  | Description                   |
| ---- | ----------------------------- |
| FVHI | Feeder Voltage High Indicator |
| FVHD | Feeder Voltage High Duration  |
| FVLI | Feeder Voltage Low Indicator  |
| FVLD | Feeder Voltage Low Duration   |
| FCHI | Feeder Current High Indicator |
| FCHD | Feeder Current High Duration  |
| FCLI | Feeder Current Low Indicator  |
| FCLD | Feeder Current Low Duration   |

---

## Anomaly Detection

Threshold-based anomaly detection is used to identify abnormal feeder behavior.

### Voltage Thresholds

```python
VOLTAGE_HIGH = 11.2
VOLTAGE_LOW  = 10.5
```

### Current Thresholds

```python
CURRENT_HIGH = 100
CURRENT_LOW  = 50
```

An anomaly episode is triggered when readings continuously violate configured thresholds.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/scada_dashboard.git
cd scada_dashboard
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy jinja2 python-multipart
```

---

## Run Application

```bash
uvicorn main:app --reload --port 8000
```

Open in browser:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

| Endpoint          | Description       |
| ----------------- | ----------------- |
| `/kpi/`           | KPI metrics       |
| `/kpi/daily`      | Daily KPI summary |
| `/kpi/timeseries` | KPI time-series   |
| `/kpi/export`     | CSV export        |
| `/data/series`    | Chart-ready data  |
| `/filters/`       | Available filters |

---

## Sample KPI Response

```json
{
  "fvhi_count": 40,
  "fvhd_total": 1287,
  "fvli_count": 6,
  "fvld_total": 51,
  "avg_voltage": 10.9
}
```

---

## Future Enhancements

* Real-time SCADA/MQTT integration
* Predictive anomaly detection using LSTM
* Alarm and notification system
* Grafana
 
 ## Authors

* Aishanya Singh    
* Akash Mall 
