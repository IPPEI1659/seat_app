import os, io, random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from flask import (
    Flask, render_template, request, session, redirect,
    url_for, flash, send_file
)
import pandas as pd
from io import BytesIO
import qrcode

# =========================
# 基本設定
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")  # 本番は環境変数に
DEFAULT_CSV = "members.csv"

# =========================
# ユーティリティ
# =========================
def normalize_gender(g: str) -> str:
    """性別を 'M'/'F'/'' に正規化"""
    s = (g or "").strip().lower()
    if s in {"m", "male", "man", "boy", "男", "男性"}:
        return "M"
    if s in {"f", "female", "woman", "girl", "女", "女性"}:
        return "F"
    return ""

def read_members(path: str) -> List[Dict[str, str]]:
    """
    CSV/TSV を読み込み [{'name':..., 'gender':...}, ...] を返す
    ・UTF-8-SIGを優先
    ・区切り自動検出（pandas engine=python）
    ・列名は 名前/氏名/name/Name、性別/gender/Gender を許容
    """
    if not os.path.exists(path):
        return []

    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        with open(path, "r", encoding="utf-8-sig") as f:
            sample = f.read()
        delim = "\t" if "\t" in (sample.splitlines()[0] if sample.splitlines() else "") else ","
        df = pd.read_csv(io.StringIO(sample), sep=delim)

    cols = {str(c).strip(): c for c in df.columns}
    name_col = cols.get("名前") or cols.get("氏名") or cols.get("name") or cols.get("Name")
    gender_col = cols.get("性別") or cols.get("gender") or cols.get("Gender")

    if not name_col:
        raise ValueError("CSVに『名前/氏名/name/Name』列がありません。")

    if not gender_col:
        df["gender"] = ""
        gender_col = "gender"

    out = []
    for _, r in df.iterrows():
        nm = ("" if pd.isna(r[name_col]) else str(r[name_col])).strip()
        if not nm:
            continue
        gd = ("" if pd.isna(r[gender_col]) else str(r[gender_col])).strip()
        out.append({"name": nm, "gender": normalize_gender(gd)})
    return out

def display_name(full_name: str) -> str:
    """表示向けに下の名前を返す（スペース/全角スペース対応）"""
    if not full_name:
        return ""
    for sep in [" ", "　", "\u3000"]:
        if sep in full_name:
            parts = [p for p in full_name.split(sep) if p]
            return parts[-1] if parts else full_name
    return full_name

def split_by_gender(people: List[Dict[str, Any]]):
    males, females, unknown = [], [], []
    for p in people:
        g = p.get("gender") or ""
        if g == "M": males.append(p)
        elif g == "F": females.append(p)
        else: unknown.append(p)
    return males, females, unknown

def make_tables(num_tables: int, default_size: int | None, names: List[str], sizes: List[int | None]):
    tables = []
    for i in range(1, num_tables + 1):
        title = (names[i-1] if i-1 < len(names) and (names[i-1] or "").strip() else f"テーブル {i}").strip()
        sz = sizes[i-1] if i-1 < len(sizes) else None
        if sz is None:
            sz = default_size
        tables.append({"idx": i, "title": title, "size": (int(sz) if sz else None)})
    return tables

def finalize_table_sizes(present_count: int, tables: List[Dict[str, Any]], equalize: bool = True) -> List[Dict[str, Any]]:
    """
    テーブルごとの size を確定する。
    equalize=True のとき：
      - 『出席人数 ÷ 卓数』で完全均等割り（余りは先頭から+1）
      - 合計は present_count に一致し、各卓の差は最大1
    """
    n = max(1, len(tables))
    if equalize:
        base = present_count // n
        rem  = present_count % n
        for i, t in enumerate(tables, start=1):
            t["size"] = base + (1 if i <= rem else 0)
        return tables

    # 参考: equalize=False（未使用）では既存指定をなるべく尊重して均等化
    fixed_total = sum((t.get("size") or 0) for t in tables)
    if any(t.get("size") is None for t in tables):
        unspecified = [t for t in tables if not t.get("size")]
        remaining = max(0, present_count - fixed_total)
        base = remaining // max(1, len(unspecified))
        rem  = remaining %  max(1, len(unspecified))
        for i, t in enumerate(unspecified, start=1):
            t["size"] = max(1, base + (1 if i <= rem else 0))
    else:
        diff = present_count - fixed_total
        step = 1 if diff > 0 else -1
        i = 0
        while diff != 0 and n > 0:
            tables[i % n]["size"] = max(1, (tables[i % n]["size"] or 1) + step)
            diff -= step
            i += 1
    return tables

def balanced_assign(people: List[Dict[str,Any]], tables: List[Dict[str,Any]], seed: int = 42) -> Dict[str, Dict[str,Any]]:
    """
    各卓の定員（size）に対して、男女比が全体比に近づくように割当。
    - 各卓の「女性目標数」を最大剰余法（Hamilton法）で配分
    - 目標Fは『全女性数』と『各卓定員』の両制約を満たす
    - その後 Female→Male→Unknown の順で充填、足りない分は他性別で補完
    """
    rnd = random.Random(seed)
    # 正規化＆シャッフル
    norm = [{"name": p["name"], "gender": normalize_gender(p.get("gender",""))} for p in people]
    females = [p for p in norm if p["gender"] == "F"]
    males   = [p for p in norm if p["gender"] == "M"]
    unknown = [p for p in norm if p["gender"] not in ("F","M")]

    rnd.shuffle(females); rnd.shuffle(males); rnd.shuffle(unknown)

    total_F = len(females)
    total_M = len(males)
    total_cap = sum(t.get("size") or 0 for t in tables)

    # --- 1) 各卓の女性目標数（公平丸め） ---
    denom = max(1, total_F + total_M)  # unknownは後補完
    ratio_F = total_F / denom

    raw_targets = []
    floor_sum = 0
    for t in tables:
        cap = t["size"] or 0
        ideal = ratio_F * cap
        flo = int(ideal)
        flo = min(flo, cap)
        raw_targets.append({
            "idx": t["idx"], "cap": cap, "title": t["title"],
            "ideal": ideal, "floor": flo, "frac": ideal - flo
        })
        floor_sum += flo

    remaining_F = max(0, total_F - floor_sum)
    raw_targets.sort(key=lambda x: x["frac"], reverse=True)
    for i in range(len(raw_targets)):
        if remaining_F <= 0:
            break
        if raw_targets[i]["floor"] < raw_targets[i]["cap"]:
            raw_targets[i]["floor"] += 1
            remaining_F -= 1

    raw_targets.sort(key=lambda x: x["idx"])
    target_F_list = [min(t["floor"], t["cap"]) for t in raw_targets]

    # --- 2) 充填 ---
    out: Dict[str, Dict[str,Any]] = {}
    for t, target_F in zip(tables, target_F_list):
        cap = t["size"] or 0
        assigned = []

        # 女性
        for _ in range(target_F):
            if females:
                assigned.append(females.pop())
            else:
                break

        # 男性
        while len(assigned) < cap and males:
            assigned.append(males.pop())

        # Unknown
        while len(assigned) < cap and unknown:
            assigned.append(unknown.pop())

        # まだ足りなければ残りの性別
        while len(assigned) < cap and (females or males):
            if females:
                assigned.append(females.pop())
            elif males:
                assigned.append(males.pop())

        # 座席番号
        for i, p in enumerate(assigned, start=1):
            out[p["name"]] = {
                "table": t["idx"],
                "seat": i,
                "table_name": t["title"],
                "gender": p["gender"],
            }

    # 念のため：空きがあれば余りを回す（保険）
    leftovers = females + males + unknown
    if leftovers:
        empty_slots = []
        for t in tables:
            cap = t["size"] or 0
            filled = sum(1 for v in out.values() if v["table"] == t["idx"])
            for pos in range(filled + 1, cap + 1):
                empty_slots.append((t, pos))
        for p, slot in zip(leftovers, empty_slots):
            t, seat_no = slot
            out[p["name"]] = {
                "table": t["idx"], "seat": seat_no, "table_name": t["title"], "gender": p["gender"]
            }

    return out

def build_seating(assigns: Dict[str, Dict[str, Any]]):
    tables = defaultdict(list)
    for name, info in assigns.items():
        tables[info["table"]].append({
            "name": name, "seat": info.get("seat"),
            "table_name": info.get("table_name"), "gender": info.get("gender")
        })
    return {k: sorted(v, key=lambda x: (x.get("seat") is None, x.get("seat", 0))) for k, v in tables.items()}

def build_seat_df():
    seating = session.get("seating", {})
    rows = []
    for t_idx, seats in seating.items():
        for s in seats:
            rows.append({
                "テーブル番号": t_idx,
                "テーブル名": s.get("table_name"),
                "席番号": s.get("seat"),
                "名前": s.get("name"),
            })
    columns = ["テーブル番号", "テーブル名", "席番号", "名前"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

# =========================
# ルート
# =========================
@app.route("/")
def index():
    # アプリのルート基準でメンバーCSVを読む
    path = Path(app.root_path) / DEFAULT_CSV
    people = []
    try:
        people = read_members(str(path))
    except Exception as e:
        flash(f"CSV読込エラー: {e}")

    session["people_all"] = people
    # 初期化
    for k in ["tables_def", "assignments", "seating", "participant_names", "name_map", "teacher_on", "teacher_name"]:
        session.pop(k, None)
    return render_template("index.html", people_count=len(people))

@app.route("/setup", methods=["GET", "POST"])
def setup():
    if request.method == "POST":
        # --- 基本入力 ---
        num_tables = int(request.form.get("num_tables", 1))
        size_per = request.form.get("size_per")
        size_per = int(size_per) if size_per else None

        # --- テーブル名とサイズ（任意） ---
        table_info = []
        for i in range(1, num_tables + 1):
            name = (request.form.get(f"table_name_{i}") or f"テーブル{i}").strip()
            size = request.form.get(f"table_size_{i}")
            size = int(size) if size else None
            table_info.append({"name": name, "size": size})

        # --- 先生入力 ---
        teacher_on = bool(request.form.get("has_teacher"))
        t_name = (request.form.get("teacher_name") or "").strip()
        t_gender = normalize_gender(request.form.get("teacher_gender") or "")

        # --- メンバー取得 & 先生追加 ---
        people_all = session.get("people_all", [])
        if teacher_on:
            if not t_name:
                t_name = "先生"
            found = False
            for i, p in enumerate(people_all):
                if p.get("name") == t_name:
                    people_all[i] = {"name": t_name, "gender": t_gender}
                    found = True
                    break
            if not found:
                people_all = people_all + [{"name": t_name, "gender": t_gender}]
            session["teacher_on"] = True
            session["teacher_name"] = t_name
        else:
            session["teacher_on"] = False
            session.pop("teacher_name", None)
        session["people_all"] = people_all

        total_people = len(people_all)
        if total_people == 0:
            flash("出席者候補が0人です。CSVを確認してください。")
            return redirect(url_for("index"))

        # --- テーブルサイズ確定（個別指定が無ければ size_per を使うが、最終的に attendance で均等化する） ---
        if any(t["size"] for t in table_info):
            assigned = sum([t["size"] or 0 for t in table_info])
            remaining = max(total_people - assigned, 0)
            auto_targets = [t for t in table_info if not t["size"]]
            if auto_targets:
                per = remaining // len(auto_targets) if len(auto_targets) else 0
                for t in auto_targets:
                    t["size"] = per if per >= 1 else None
        else:
            if size_per:
                for t in table_info:
                    t["size"] = size_per
            else:
                # 仮サイズ（後でattendanceで均等化）
                per = max(1, total_people // max(1, num_tables))
                for t in table_info:
                    t["size"] = per

        names = [t["name"] for t in table_info]
        sizes = [t["size"] for t in table_info]
        tables = make_tables(num_tables, None, names, sizes)
        session["tables_def"] = tables

        return redirect(url_for("attendance"))

    # GET
    people = session.get("people_all", [])
    return render_template("setup.html", people_count=len(people))

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    people_all = session.get("people_all", [])
    # 表示名マップ（下の名前 → フルネーム）
    name_map = {display_name(p["name"]): p["name"] for p in people_all}
    disp_people = [display_name(p["name"]) for p in people_all]

    if request.method == "POST":
        present_disp = request.form.getlist("present")
    else:
        # 初期表示は全員出席扱い
        present_disp = disp_people

    present_full = [name_map[d] for d in present_disp if d in name_map]

    # 先生がONなら強制的に含める
    if session.get("teacher_on"):
        t = session.get("teacher_name")
        if t and t not in present_full:
            present_full.append(t)

    present = [p for p in people_all if p["name"] in present_full]

    tables = session.get("tables_def", [])
    # ★ ここで完全均等（±1以内）に確定
    tables = finalize_table_sizes(len(present), tables, equalize=True)
    session["tables_def"] = tables

    assigns = balanced_assign(present, tables, seed=42)
    seating = build_seating(assigns)

    session["assignments"] = assigns
    session["seating"] = seating
    session["participant_names"] = [display_name(p["name"]) for p in present]
    session["name_map"] = name_map

    return render_template("attendance.html", people=disp_people, tables=tables, seating=seating)

@app.route("/result")
def result():
    participants = session.get("participant_names", [])
    return render_template("result.html", participants=participants)

@app.route("/list_participants")
def list_participants():
    seating = session.get("seating", {})
    tables_out = []
    for t_idx in sorted(seating.keys()):
        seats = seating[t_idx]
        title = seats[0]["table_name"] if seats and seats[0].get("table_name") else f"テーブル {t_idx}"
        rows = [{"seat": s.get("seat"), "name": s.get("name")} for s in seats]
        tables_out.append({"title": title, "rows": rows, "count": len(rows)})
    return render_template("list_participants.html", tables=tables_out)

@app.route("/my_seat/<participant>")
def my_seat(participant):
    name_map = session.get("name_map", {})
    full = name_map.get(participant) or participant  # 直接フルネーム指定にも対応
    assigns = session.get("assignments", {})
    if not full or full not in assigns:
        flash("席情報が見つかりません。")
        return redirect(url_for("result"))
    return render_template("my_seat.html", participant=participant, seat=assigns[full])

# =========================
# ダウンロード & QR
# =========================
@app.get("/download/csv")
def download_csv():
    df = build_seat_df()
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="seating.csv")

@app.get("/download/xlsx")
def download_xlsx():
    df = build_seat_df()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="seating")
    buf.seek(0)
    return send_file(buf,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True,
                     download_name="seating.xlsx")

@app.get("/qr/<participant>")
def qr(participant):
    base = request.host_url.rstrip("/")
    target = base + url_for("my_seat", participant=participant)
    img = qrcode.make(target)
    b = BytesIO()
    img.save(b, format="PNG")
    b.seek(0)
    return send_file(b, mimetype="image/png")

@app.get("/reset")
def reset():
    for k in ["people_all","tables_def","assignments","seating",
              "participant_names","name_map","teacher_on","teacher_name"]:
        session.pop(k, None)
    flash("設定をリセットしました。")
    return redirect(url_for("index"))

# =========================
# エントリ
# =========================
if __name__ == "__main__":
    print(">> アプリ起動: http://127.0.0.1:5000")
    app.run(debug=True)
