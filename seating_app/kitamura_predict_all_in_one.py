import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

# ========== あなたの環境に合わせる場所 ==========
BASE = r"C:\Users\ippei\Desktop\kitamura"
DATA_DIR = rf"{BASE}\data_clean"
OUT_DIR  = rf"{BASE}\_pred"   # 出力先フォルダ

FILES = {
    "mario":   rf"{DATA_DIR}\スタジオマリオ購買_202004-202509_cleaned.csv",
    "ec":      rf"{DATA_DIR}\キタムラEC購買実績_202004-202509_cleaned.csv",
    "web":     rf"{DATA_DIR}\WEBサイト行動_EC_20230401-20250930_cleaned.csv",
    "member":  rf"{DATA_DIR}\キタムラ会員情報_20250930_cleaned.csv",  
    "product": rf"{DATA_DIR}\商品マスタ_cleaned.csv",
    "used":    rf"{DATA_DIR}\中古品番マスタ_cleaned.csv",
}

HORIZON = 90              # ASOFから何日以内に再購入があるか
ASOF: Optional[str] = None  # 例: "2025-06-30"。Noneなら自動選定

# ========== 列名マッピング ==========
MAPPING = {
    # 共通ID・時間
    "member_id": ["member_id", "会員ID", "会員番号", "マリオ会員番号", "net_member_id", "mario_member_id"],
    "order_time": ["order_time", "注文日時", "受注日時", "購入日時", "購買日", "order_datetime", "purchase_date"],
    "revenue": ["revenue", "売上金額", "売上", "金額", "amount", "total", "total_amount", "price",
                "支払金額", "税込金額", "税抜金額", "金額（税込）", "ご請求金額", "sales"],
    "order_id": ["order_id", "注文ID", "受注ID", "オーダーID", "受付番号(レシート番号)", "receipt_no", "detail_no"],
    "channel": ["channel", "チャネル", "チャネル名", "購入チャネル", "スタジオ", "ECサイト"],
    # Webログ
    "event_time": ["event_time", "イベント時刻", "イベント日時", "発生日時", "実行日時", "event_datetime"],
    # 会員（今回は必須ではない）
    "birthdate": ["birthdate", "生年月日", "誕生日"],
    "gender":    ["gender", "性別", "会員性別"],
    "prefecture":["prefecture", "都道府県(住所)", "都道府県"],
}

# ========== 便利関数 ==========
def read_csv_u(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise FileNotFoundError(f"読み込み失敗: {path}")

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def must_col(df: pd.DataFrame, key: str) -> str:
    col = pick_col(df, MAPPING[key])
    if col is None:
        raise KeyError(f"必要な列が見つかりません: {key} -> 候補 {MAPPING[key]}\n実列: {list(df.columns)[:10]} ...")
    return col

def to_datetime_safe(s: pd.Series) -> pd.Series:
    # 元データは変更せず内部処理のみ
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)

def to_numeric_money(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace("¥", "", regex=False)
         .str.replace("￥", "", regex=False)
         .str.replace("円", "", regex=False)
         .str.strip()
         .pipe(pd.to_numeric, errors="coerce")
    )

def ensure_member_key(df: pd.DataFrame, label="member_id") -> pd.DataFrame:
    """df内の会員キー列を 'member_id' に強制統一"""
    if "member_id" in df.columns:
        df["member_id"] = df["member_id"].astype(str).str.strip()
        return df
    cand = pick_col(df, MAPPING["member_id"])
    if cand:
        df = df.rename(columns={cand: "member_id"})
        df["member_id"] = df["member_id"].astype(str).str.strip()
        return df
    # どうしても無ければそのまま返す（後でデバッグ用に列一覧を見る）
    return df

# ========== 1) 読み込み & 標準化 ==========
def load_tables():
    mario  = read_csv_u(FILES["mario"])
    ec     = read_csv_u(FILES["ec"])
    web    = read_csv_u(FILES["web"])
    # member = read_csv_u(FILES["member"])  # 今は使わないが将来活用
    return mario, ec, web

def build_combined_orders(mario: pd.DataFrame, ec: pd.DataFrame) -> pd.DataFrame:
    # ---- EC ----
    ec_mid  = must_col(ec, "member_id")
    ec_time = must_col(ec, "order_time")
    ec_amt  = pick_col(ec, MAPPING["revenue"])
    ec_oid  = pick_col(ec, MAPPING["order_id"]) or ec_time

    ec_std = pd.DataFrame({
        "member_id": ec[ec_mid].astype(str).str.strip(),
        "order_time": ec[ec_time],
        "revenue": ec[ec_amt] if ec_amt else np.nan,
        "order_id": ec[ec_oid] if ec_oid else np.nan,
        "channel": "EC",
    })

    # ---- マリオ ----
    mario_mid  = pick_col(mario, ["mario_member_id", "net_member_id", "member_id"]) or must_col(mario, "member_id")
    mario_time = must_col(mario, "order_time")
    mario_amt  = pick_col(mario, MAPPING["revenue"])
    mario_oid  = pick_col(mario, MAPPING["order_id"]) or mario_time

    mario_std = pd.DataFrame({
        "member_id": mario[mario_mid].astype(str).str.strip(),
        "order_time": mario[mario_time],
        "revenue": mario[mario_amt] if mario_amt else np.nan,
        "order_id": mario[mario_oid] if mario_oid else np.nan,
        "channel": "STUDIO",
    })

    # ---- 結合 + 内部正規化（元CSV不変）----
    orders = pd.concat([ec_std, mario_std], ignore_index=True)

    orders["order_time"] = to_datetime_safe(orders["order_time"])
    orders["revenue"]    = to_numeric_money(orders["revenue"])
    orders["member_id"]  = orders["member_id"].astype(str).str.strip()
    orders["order_id"]   = orders["order_id"].astype(str).str.strip()

    # 最低限の欠損除外
    orders = orders.dropna(subset=["member_id", "order_time"]).reset_index(drop=True)
    return orders

# ========== 2) ASOF & ラベル ==========
def count_labels_at_asof(orders: pd.DataFrame, asof: pd.Timestamp, horizon: int) -> Tuple[int, int]:
    t = orders["order_time"]
    hist = orders[t <= asof]
    futr = orders[(t > asof) & (t <= asof + pd.Timedelta(days=horizon))]
    if hist.empty:
        return 0, 0
    used = hist.groupby("member_id")["order_time"].count().reset_index(name="cnt")
    ids = set(used[used["cnt"] >= 1]["member_id"])
    fut_ids = set(futr["member_id"].unique())
    return len(ids), len(ids & fut_ids)

def choose_asof_auto(orders: pd.DataFrame, horizon: int, n_candidates: int = 12) -> Optional[pd.Timestamp]:
    if orders.empty:
        return None
    tmin, tmax = orders["order_time"].min(), orders["order_time"].max()
    if pd.isna(tmin) or pd.isna(tmax):
        return None
    latest_valid = tmax - pd.Timedelta(days=horizon)
    if latest_valid <= tmin:
        return latest_valid
    candidates = pd.date_range(tmin + pd.Timedelta(days=30), latest_valid - pd.Timedelta(days=30), periods=n_candidates)
    best = (None, -1)
    for asof in candidates:
        _, pos = count_labels_at_asof(orders, asof, horizon)
        if pos > best[1]:
            best = (asof, pos)
    return best[0] if best[0] is not None else latest_valid

def make_labels(orders: pd.DataFrame, asof: pd.Timestamp, horizon: int) -> pd.DataFrame:
    t = orders["order_time"]
    hist = orders[t <= asof].copy()
    futr = orders[(t > asof) & (t <= asof + pd.Timedelta(days=horizon))].copy()
    if hist.empty:
        return pd.DataFrame(columns=["member_id", f"label_within{horizon}d"])
    future_flag = futr.groupby("member_id").size().rename("future_cnt").reset_index()
    df = (hist.groupby("member_id").size().rename("hist_cnt").reset_index()
          .merge(future_flag, on="member_id", how="left"))
    df["future_cnt"] = df["future_cnt"].fillna(0)
    df[f"label_within{horizon}d"] = (df["future_cnt"] > 0).astype(int)
    return df[["member_id", f"label_within{horizon}d"]]

# ========== 3) 特徴量 ==========
def build_features(orders: pd.DataFrame, web: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    # RFM
    hist = orders[orders["order_time"] <= asof].copy()
    if hist.empty:
        return pd.DataFrame(columns=["member_id", "recency_days", "freq", "monetary", "web_30d_events"])

    # order_id が無ければ order_time を代用してユニーク件数
    oid = pick_col(hist, MAPPING["order_id"]) or "order_time"

    agg = (hist
           .groupby("member_id")
           .agg(last_order=("order_time", "max"),
                freq=(oid, "nunique"),
                monetary=("revenue", "sum"))
           .reset_index())
    agg["recency_days"] = (asof - agg["last_order"]).dt.days
    agg = agg.drop(columns=["last_order"])
    agg["monetary"] = agg["monetary"].astype(float)

    # Web 直近30日イベント数
    feats = agg.copy()
    if web is not None and not web.empty:
        wmid = pick_col(web, MAPPING["member_id"])
        wt   = pick_col(web, MAPPING["event_time"])
        if wmid and wt:
            web = web.rename(columns={wmid: "member_id", wt: "event_time"})
            web["member_id"] = web["member_id"].astype(str).str.strip()
            web["event_time"] = to_datetime_safe(web["event_time"])
            w30 = web[(web["event_time"] <= asof) & (web["event_time"] >= asof - pd.Timedelta(days=30))]
            web_agg = w30.groupby("member_id").size().rename("web_30d_events").reset_index()
            feats = feats.merge(web_agg, on="member_id", how="left")
    feats["web_30d_events"] = feats["web_30d_events"].fillna(0)
    return feats

# ========== 4) 学習・評価・出力 ==========
def train_and_predict(data: pd.DataFrame, out_dir: str, horizon: int):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 必須列確認
    y_cols = [c for c in data.columns if c.startswith("label_within")]
    if not y_cols:
        data.to_csv(Path(out_dir) / "training_data_inspect.csv", index=False, encoding="utf-8-sig")
        raise SystemExit("[ERROR] ラベル列がありません。training_data_inspect.csv を確認してください。")
    y_col = y_cols[0]

    need = ["member_id", "recency_days", "freq", "monetary", "web_30d_events", y_col]
    miss = [c for c in need if c not in data.columns]
    if miss:
        data.to_csv(Path(out_dir) / "training_data_inspect.csv", index=False, encoding="utf-8-sig")
        raise SystemExit(f"[ERROR] 学習に必要な列が不足: {miss}")

    # 学習セット
    data = ensure_member_key(data)
    y = data[y_col].astype(int)
    X = data[["recency_days", "freq", "monetary", "web_30d_events"]].astype(float)

    if len(data) == 0 or y.nunique() < 2:
        data.to_csv(Path(out_dir) / "training_data_inspect.csv", index=False, encoding="utf-8-sig")
        print("[INFO] 教師データ不足（または単一クラス）: training_data_inspect.csv を確認してください。")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Logistic（必須）
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    p_log = clf.predict_proba(X_te)[:, 1]
    auc_log = roc_auc_score(y_te, p_log)
    rpt_log = classification_report(y_te, (p_log > 0.5).astype(int), output_dict=True)

    metrics = [{
        "model": "Logistic",
        "AUC": auc_log,
        "precision": rpt_log["1"]["precision"],
        "recall": rpt_log["1"]["recall"],
        "f1": rpt_log["1"]["f1-score"]
    }]

    # XGBoost（入っていれば実行）
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
            reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist",
            eval_metric="auc"
        )
        xgb.fit(X_tr, y_tr)
        p_x = xgb.predict_proba(X_te)[:, 1]
        auc_x = roc_auc_score(y_te, p_x)
        fpr, tpr, thr = roc_curve(y_te, p_x)
        idx = int(np.argmax(tpr - fpr))
        thr_best = float(thr[idx])
        rpt_x = classification_report(y_te, (p_x > thr_best).astype(int), output_dict=True)
        metrics.append({
            "model": "XGBoost",
            "AUC": auc_x,
            "precision": rpt_x["1"]["precision"],
            "recall": rpt_x["1"]["recall"],
            "f1": rpt_x["1"]["f1-score"],
            "best_threshold": thr_best,
        })

        # 全件スコア
        all_proba = xgb.predict_proba(X)[:, 1]
    except Exception as e:
        print("[INFO] XGBoost skipped:", e)
        all_proba = clf.predict_proba(X)[:, 1]

    # 出力
    pred = data[["member_id"]].copy()
    pred[f"proba_within{horizon}d"] = all_proba
    pred.to_csv(Path(out_dir) / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(Path(out_dir) / "metrics.csv", index=False, encoding="utf-8-sig")
    data[["member_id", y_col, "recency_days", "freq", "monetary", "web_30d_events"]] \
        .to_csv(Path(out_dir) / "training_data_used.csv", index=False, encoding="utf-8-sig")

# ========== メイン ==========
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1) 読み込み
    mario, ec, web = load_tables()

    # 2) 注文結合（内部標準化）
    orders = build_combined_orders(mario, ec)
    orders = ensure_member_key(orders)  # 念のため

    # 3) ASOF
    if ASOF:
        asof = pd.to_datetime(ASOF)
    else:
        asof = choose_asof_auto(orders, HORIZON)
    if pd.isna(asof):
        raise SystemExit("ASOF を確定できませんでした（データに日時が無い可能性）。")
    print(f"[INFO] ASOF = {asof}")

    # 4) ラベル
    label = make_labels(orders, asof, HORIZON)
    label = ensure_member_key(label)

    # 5) 特徴量
    feats = build_features(orders, web, asof)
    feats = ensure_member_key(feats)

    # 6) マージ（キーを 'member_id' で統一済み）
    data = label.merge(feats, on="member_id", how="inner")
    if data.empty:
        # デバッグ用に列一覧を出力
        print("[DEBUG] label columns:", list(label.columns))
        print("[DEBUG] feats  columns:", list(feats.columns))
        data.to_csv(Path(OUT_DIR) / "training_data_inspect.csv", index=False, encoding="utf-8-sig")
        raise SystemExit("[ERROR] label と feats が結合できませんでした。列名/IDの正規化を確認してください。")

    # 7) 学習・評価・出力
    train_and_predict(data, OUT_DIR, HORIZON)

    print("\n[DONE] 出力先:", OUT_DIR)
    print(" - metrics.csv            … モデル指標")
    print(" - predictions.csv        … 会員別 予測確率（全件）")
    print(" - training_data_used.csv … 学習に使った特徴量とラベル")
    print(" - training_data_inspect.csv …（教師データ不足/結合失敗時のみ）")

if __name__ == "__main__":
    main()
