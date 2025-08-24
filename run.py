import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.discriminant_analysis import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# Load dataset
df = pd.read_csv("datasets/dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# fungsi rank koin menggunakan clustering DBScan
def rank_coins():
    # perhitungan dasar
    df_grouped = df.groupby("Crypto").agg({
        "Close": ["first", "last", "mean", "std"]
    }).reset_index()
    df_grouped.columns = ["Crypto", "First", "Last", "AvgPrice", "Std"]

    df_grouped["Return"] = (df_grouped["Last"] - df_grouped["First"]) / df_grouped["First"] * 100  # persen
    df_grouped["Volatility"] = (df_grouped["Std"] / df_grouped["AvgPrice"]) * 100  # persen

    # clustering dengan DBSCAN 
    X = df_grouped[["Return", "Volatility"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = DBSCAN(eps=0.5, min_samples=2)
    df_grouped["Cluster"] = clustering.fit_predict(X_scaled)

    # scoring: return tinggi & volatility rendah lebih bagus
    df_grouped["Score"] = df_grouped["Return"] - df_grouped["Volatility"]

    df_ranked = df_grouped.sort_values(
        by=["Score"], ascending=False
    ).reset_index(drop=True)

    df_ranked["Rank"] = range(1, len(df_ranked) + 1)
    df_ranked["Return"] = df_ranked["Return"].round(2)
    df_ranked["Volatility"] = df_ranked["Volatility"].round(2)
    df_ranked["AvgPrice"] = df_ranked["AvgPrice"].round(2)

    # konklusi dinamis
    top = df_ranked.iloc[0]
    worst = df_ranked.iloc[-1]

    conclusion = (
        f"Dari hasil ranking:\n"
        f"- **{top['Crypto']}** paling unggul karena return {top['Return']}% "
        f"dengan volatilitas {top['Volatility']}% (stabil & menguntungkan).\n"
        f"- Sebaliknya, **{worst['Crypto']}** kurang menarik dengan return {worst['Return']}% "
        f"dan volatilitas {worst['Volatility']}%.\n\n"
    )

    return df_ranked[["Crypto", "Return", "Volatility", "AvgPrice", "Rank"]], conclusion

# prediksi menggunakan SVR 
current_prices = {
    "BTC": 114640,   # Bitcoin
    "ETH": 4742,    # Ethereum
    "XRP": 3,    # Ripple (XRP)
    "LTC": 118       # Litecoin
}

def train_model(coin):
    sub = df[df['Crypto']==coin].sort_values('Date').copy()
    sub["t"] = (sub["Date"] - sub["Date"].min()).dt.days
    X = sub[["t"]].values
    y = np.log(sub["Close"].values)

    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.1))
    model.fit(X, y)
    return model, sub

def predict_prices(coin):
    model, sub = train_model(coin)
    current_price = current_prices.get(coin, None)
    if current_price is None:
        return {"Error": f"Current price untuk {coin} belum di-hardcode."}

    horizons = {"1 Minggu": 7, "1 Bulan": 30, "1 Tahun": 365}
    results = {}

    last_t = (pd.Timestamp.now() - sub["Date"].min()).days
    base_pred_log = model.predict([[last_t]])[0]

    for label, days_ahead in horizons.items():
        future_t = np.array([[last_t + days_ahead]])
        pred_log = model.predict(future_t)[0]

        rel_growth = np.exp(pred_log - base_pred_log)
        pred_price = current_price * rel_growth
        results[label] = round(float(pred_price), 2)

    return results

# klasifikasi label dari hasil prediksi
def classify_action(coin):
    preds = predict_prices(coin)
    current_price = current_prices.get(coin, None)
    if current_price is None:
        return {"Error": f"Current price untuk {coin} belum di-hardcode."}

    results = {}
    for horizon, pred in preds.items():
        if isinstance(pred, str):
            results[horizon] = pred
            continue
        ret = (pred - current_price) / current_price
        if ret > 0.05:   
            label = "Buy"
        elif ret < -0.05: 
            label = "Sell"
        else:
            label = "Hold"
        results[horizon] = {"Prediksi Harga": pred, "Saran": label}
    return results

# =============================
# 4. Gradio UI
# =============================
rank_df, conclusion = rank_coins()

def ui_rank():
    return rank_df, conclusion

def ui_predict(coin):
    return predict_prices(coin)

def ui_classify(coin):
    return classify_action(coin)

coin_list = df['Crypto'].unique().tolist()

with gr.Blocks() as demo:
    gr.Markdown(" # Crypto Analysis App")

    # Section 1
    gr.Markdown("## Section 1: Coin Rank (Clustering)")
    gr.Markdown("## LEGEND")
    gr.Markdown(" - Return : Persentase keuntungan")
    gr.Markdown(" - Volatility : Persentase volatilitas")
    gr.Markdown(" - Avg Price : harga rata-rata coin")
    rank_btn = gr.Button("Tampilkan Ranking")
    rank_out = gr.DataFrame()
    rank_conclusion = gr.Markdown()
    rank_btn.click(fn=ui_rank, outputs=[rank_out, rank_conclusion])

    # Section 2
    gr.Markdown("## Section 2: Prediksi Harga (Regression)")
    coin_dropdown_pred = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    pred_out = gr.JSON()
    coin_dropdown_pred.change(fn=ui_predict, inputs=coin_dropdown_pred, outputs=pred_out)

    # Section 3
    gr.Markdown("## Section 3: Klasifikasi Aksi (Classification)")
    coin_dropdown_class = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    class_out = gr.JSON()
    coin_dropdown_class.change(fn=ui_classify, inputs=coin_dropdown_class, outputs=class_out)

# Run demo
demo.launch()
