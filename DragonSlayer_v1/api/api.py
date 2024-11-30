from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test API")


@app.get("/")
def homepage():
    return {"message": "Hello World!"}

### Predict now (http://127.0.0.1:8000/predict?ticker=query)
@app.get("/predict")
def predict(ticker):
    return {"ticker": ticker}

### Backtest (http://127.0.0.1:8000/backtest/{json})
### in json: {ticker, start_date, end_date}
@app.get("/backtest/{ticker}/{start_date}/{end_date}")
def backtest(ticker, start_date, end_date):
    backtest_info = {"ticker":ticker,
                     "start_date":start_date,
                     "end_date":end_date
                     }
    return backtest_info

if __name__ == "__main__":
    uvicorn.run("api:app",
                host="0.0.0.0",
                port=8000,
                reload=True)