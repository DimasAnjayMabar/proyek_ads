import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import silhouette_score, mean_squared_error
import gradio as gr
from scipy.stats import skew
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("datasets/dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# ranking coin menggunakan cluster DBScan
def rank_coins():
    # feature engineering untuk menambahkan first,last, mean, std
    df_grouped = df.groupby("Crypto").agg({
        "Close": ["first", "last", "mean", "std"]
    }).reset_index()
    df_grouped.columns = ["Crypto", "First", "Last", "AvgPrice", "Std"]

    df_grouped["Return"] = (df_grouped["Last"] - df_grouped["First"]) / df_grouped["First"] * 100
    df_grouped["Volatility"] = (df_grouped["Std"] / df_grouped["AvgPrice"]) * 100

    X = df_grouped[["Return", "Volatility"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = DBSCAN(eps=0.5, min_samples=2)
    df_grouped["Cluster"] = clustering.fit_predict(X_scaled)

    df_grouped["Score"] = df_grouped["Return"] - df_grouped["Volatility"]

    df_ranked = df_grouped.sort_values(
        by=["Score"], ascending=False
    ).reset_index(drop=True)

    df_ranked["Rank"] = range(1, len(df_ranked) + 1)
    df_ranked["Return"] = df_ranked["Return"].round(2)
    df_ranked["Volatility"] = df_ranked["Volatility"].round(2)
    df_ranked["AvgPrice"] = df_ranked["AvgPrice"].round(2)

    top = df_ranked.iloc[0]
    worst = df_ranked.iloc[-1]

    conclusion = (
        f"Dari hasil ranking:\n"
        f"- **{top['Crypto']}** paling unggul karena return {top['Return']}% "
        f"dengan volatilitas {top['Volatility']}%.\n"
        f"- Sebaliknya, **{worst['Crypto']}** kurang menarik dengan return {worst['Return']}% "
        f"dan volatilitas {worst['Volatility']}%.\n\n"
    )

    return df_ranked[["Crypto", "Return", "Volatility", "AvgPrice", "Rank"]], conclusion

# evaluasi clustering menggunakan sillhouete
def evaluate_clustering():
    df_grouped = df.groupby("Crypto").agg({
        "Close": ["first", "last", "mean", "std"]
    }).reset_index()
    df_grouped.columns = ["Crypto", "First", "Last", "AvgPrice", "Std"]

    df_grouped["Return"] = (df_grouped["Last"] - df_grouped["First"]) / df_grouped["First"] * 100
    df_grouped["Volatility"] = (df_grouped["Std"] / df_grouped["AvgPrice"]) * 100

    X = df_grouped[["Return", "Volatility"]]
    X_scaled = StandardScaler().fit_transform(X)

    clustering = DBSCAN(eps=0.5, min_samples=2)
    labels = clustering.fit_predict(X_scaled)

    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        score = silhouette_score(X_scaled[mask], labels[mask])
        return round(score, 3)
    else:
        return "titik terlalu sedikit"

# prediksi harga koin menggunakan SVR (support vector regression)
current_prices = {
    "BTC": 114640,   
    "ETH": 4742,    
    "XRP": 3,        
    "LTC": 118       
}

def train_model(coin):
    # train model berdasarkan tanggal dan harga closenya
    sub = df[df['Crypto']==coin].sort_values('Date').copy()
    sub["t"] = (sub["Date"] - sub["Date"].min()).dt.days
    X = sub[["t"]].values
    y = np.log(sub["Close"].values)

    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.1))
    model.fit(X, y)
    return model, sub

def predict_prices(coin):
    # prediksi dilakukan dari hari ini
    model, sub = train_model(coin)
    current_price = current_prices.get(coin, None)
    if current_price is None:
        return {"Error": f"Current price untuk {coin} belum di-hardcode."}

    horizons = {"1 Minggu": 7, "1 Bulan": 30, "1 Tahun": 365}
    results = {}

    last_t = (pd.Timestamp.now() - sub["Date"].min()).days
    base_pred_log = model.predict([[last_t]])[0]

    for label, days_ahead in horizons.items():
        pred_log = model.predict([[last_t + days_ahead]])[0]
        rel_growth = np.exp(pred_log - base_pred_log)
        pred_price = current_price * rel_growth
        results[label] = round(float(pred_price), 2)

    return results

# evaluasi regression menggunakan MSE dan RMSE
def evaluate_regression(coin):
    sub = df[df['Crypto']==coin].sort_values('Date').copy()
    sub["t"] = (sub["Date"] - sub["Date"].min()).dt.days
    X = sub[["t"]].values
    y = np.log(sub["Close"].values)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {"MSE": round(mse, 4), "RMSE": round(rmse, 4)}

# klasifikasi sederhana, tanpa model khusus
def classify_action(coin):
    preds = predict_prices(coin)
    current_price = current_prices.get(coin, None)
    if current_price is None:
        return {"Error": f"Current price untuk {coin} belum di-hardcode."}

    avg_price = df[df['Crypto']==coin]["Close"].mean()

    results = {}
    for horizon, pred in preds.items():
        ret = (pred - current_price) / current_price

        if pred > current_price * 1.02:  
            if pred < avg_price:
                label = "Buy karena tren naik"
            else:
                label = "Sell untuk ambil profit"
        elif pred < current_price * 0.98: 
            label = "Cut loss untuk menghindari tren turun"
        else:
            label = "Hold karena sinyal tidak jelas atau pasar masih stabil"

        results[horizon] = {"Prediksi Harga": round(pred,2), "Saran": label}

    return results

# plot dataset
def plot_return_volatility():
    df_grouped = df.groupby("Crypto").agg({
        "Close": ["first", "last", "mean", "std"]
    }).reset_index()
    df_grouped.columns = ["Crypto", "First", "Last", "AvgPrice", "Std"]

    df_grouped["Return"] = (df_grouped["Last"] - df_grouped["First"]) / df_grouped["First"] * 100
    df_grouped["Volatility"] = (df_grouped["Std"] / df_grouped["AvgPrice"]) * 100

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(df_grouped["Volatility"], df_grouped["Return"], s=200)

    for i, row in df_grouped.iterrows():
        ax.text(row["Volatility"]+0.2, row["Return"], row["Crypto"])

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Distribusi Return vs Volatility (2018–2023)")
    ax.grid(True)
    return fig

def plot_distribution_and_skew(coin):
    sub = df[df['Crypto']==coin].copy()
    data = sub["Close"]
    log_data = np.log(data)

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].hist(data, bins=30, edgecolor="k")
    axes[0].set_title(f"Distribusi Harga {coin}")
    axes[0].set_xlabel("Close Price")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(log_data, bins=30, edgecolor="k")
    axes[1].set_title(f"Distribusi Log Harga {coin}")
    axes[1].set_xlabel("Log(Close)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()

    skewness_info = {
        "Skewness Harga Asli": float(round(skew(data), 3)),
        "Skewness Log Harga": float(round(skew(log_data), 3))
    }

    return fig, skewness_info 

# ui gradio
rank_df, conclusion = rank_coins()

def ui_rank():
    return rank_df, conclusion

def ui_predict(coin):
    return predict_prices(coin)

def ui_classify(coin):
    return classify_action(coin)

def ui_eval_cluster():
    return {"Silhouette Score": evaluate_clustering()}

def ui_eval_regression(coin):
    return evaluate_regression(coin)

coin_list = df['Crypto'].unique().tolist()

with gr.Blocks() as demo:
    gr.Markdown(" # Crypto Analysis App")

    gr.Markdown("## Visualisasi Distribusi Return vs Volatility")
    plot_btn = gr.Button("Tampilkan Scatter Plot")
    plot_out = gr.Plot()
    plot_btn.click(fn=plot_return_volatility, outputs=plot_out)

    gr.Markdown("## Visualisasi Distribusi Harga dan Skewness")
    coin_dropdown_vis = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    vis_out_plot = gr.Plot()
    vis_out_skew = gr.JSON()
    coin_dropdown_vis.change(
        fn=plot_distribution_and_skew,
        inputs=coin_dropdown_vis,
        outputs=[vis_out_plot, vis_out_skew]
    )

    gr.Markdown("## Section 1: Coin Rank (Clustering)")
    rank_btn = gr.Button("Tampilkan Ranking")
    rank_out = gr.DataFrame()
    rank_conclusion = gr.Markdown()
    rank_btn.click(fn=ui_rank, outputs=[rank_out, rank_conclusion])

    gr.Markdown("## Evaluasi Clustering")
    eval_cluster_btn = gr.Button("Hitung Silhouette Score")
    eval_cluster_out = gr.JSON()
    eval_cluster_btn.click(fn=ui_eval_cluster, outputs=eval_cluster_out)

    gr.Markdown("## Section 2: Prediksi Harga (Regression)")
    coin_dropdown_pred = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    pred_out = gr.JSON()
    coin_dropdown_pred.change(fn=ui_predict, inputs=coin_dropdown_pred, outputs=pred_out)

    gr.Markdown("## Evaluasi Regression (MSE & RMSE)")
    coin_dropdown_eval = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    eval_reg_out = gr.JSON()
    coin_dropdown_eval.change(fn=ui_eval_regression, inputs=coin_dropdown_eval, outputs=eval_reg_out)

    gr.Markdown("## Section 3: Klasifikasi Aksi (Classification)")
    coin_dropdown_class = gr.Dropdown(choices=coin_list, label="Pilih Coin")
    class_out = gr.JSON()
    coin_dropdown_class.change(fn=ui_classify, inputs=coin_dropdown_class, outputs=class_out)

# run demo
demo.launch()

