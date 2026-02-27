import pandas as pd
import scheduler

from ortools.sat.python import cp_model

class Cheker:
    def __init__(self, df_lock, df_result, clean_rules, clean_time=60, max_clean_worker=1):
        self.df_lock = df_lock
        self.df_result = df_result
        self.clean_rules = clean_rules
        self.clean_time = clean_time
        self.max_clean_worker = max_clean_worker

    def add_clean_rows_global(self, df):
            """
            グローバルに掃除行(job=-1)を挿入する（見える化用）
            - solver本体は「掃除 interval を作らず、ギャップで確保」しているため、
            掃除の“実施時刻”は未定義（自由）です。
            - ここでは “見える化用” として、ギャップ内のどこに掃除を置くかを
            追加の小さなCP-SATで割り付けます（ロックと重複しない / 同時掃除上限を守る）。

            可変掃除時間対応:
            - clean_rules が与えられた場合、材料変化点ごとに
            ct = clean_time(mach, from_mat, to_mat, qty, clean_rules, default=CLEAN_TIME)
            を計算して、その ct 分の掃除行を作成します。
            - clean_rules が None の場合は従来どおり CLEAN_TIME 固定で動作します。
            """
            if df is None or len(df) == 0:
                return df
            if self.clean_time is None:
                self.clean_time = 0
            self.clean_time = int(self.clean_time)

            # job=-1（既存の掃除行）はいったん除外して作り直す
            df_n = df[df["job"] != -1].copy()

            # 数値列の安全化（NaNが混ざっても落ちないように）
            num_cols = ["mach", "mat", "start_setup", "end_proc", "start_proc", "qty"]
            for col in num_cols:
                if col in df_n.columns:
                    df_n[col] = pd.to_numeric(df_n[col], errors="coerce").fillna(0).astype(int)
            # ---------
            # utility
            # ---------
            def _merge_intervals(intervals):
                intervals = sorted((int(s), int(e)) for s, e in intervals if e > s)
                merged = []
                for s, e in intervals:
                    if not merged or s > merged[-1][1]:
                        merged.append([s, e])
                    else:
                        merged[-1][1] = max(merged[-1][1], e)
                return [(s, e) for s, e in merged]

            def _free_segments(w0: int, w1: int, busy, dur: int):
                """
                # [w0,w1] から busy を引いて、dur を置ける free 区間のリストを返す
                """
                dur = int(dur)
                if dur <= 0:
                    return []
                if w1 - w0 < dur:
                    return []

                clip = []
                for s, e in busy:
                    s2 = max(int(s), w0)
                    e2 = min(int(e), w1)
                    if e2 > s2:
                        clip.append((s2, e2))
                busy_m = _merge_intervals(clip)

                segs = []
                cur = w0
                for s, e in busy_m:
                    if s - cur >= dur:
                        segs.append((cur, s))
                    cur = max(cur, e)
                if w1 - cur >= dur:
                    segs.append((cur, w1))
                return segs

            def _calc_ct(mach: int, from_mat: int, to_mat: int, qty: int) -> int:
                """
                可変掃除時間（rulesがなければ固定CLEAN_TIME）
                clean_time() は solve_job_v8.py 内の関数を想定
                """
                if self.clean_rules is None:
                    return max(0, self.clean_time)
                return int(scheduler.clean_time(
                    mach=int(mach),
                    mat_from=int(from_mat),
                    mat_to=int(to_mat),
                    qty=int(qty),
                    rules=self.clean_rules,
                    default=max(0, self.clean_time),
                ))
        
            # ----------------------------
            # 1) 掃除イベント（= 材料変化点）を抽出し、置けるウィンドウ(複数区間)を作る
            # ----------------------------
            events = []
            for mch, block in df_n.groupby("mach"):
                mch = int(mch)
                if mch == 0:
                    continue

                block = block.sort_values("start_proc").reset_index(drop=True)

                last_real_mat = None
                last_real_end = None
                busy_between = []  # last_real_end 以降に挟まる占有区間（machine側）

                for _, r in block.iterrows():
                    cur_mat = int(r.get("mat", 0))
                    st_setup = int(r.get("start_setup", 0))
                    en_proc = int(r.get("end_proc", st_setup))
                    cur_qty = int(r.get("qty", 1))

                    # last_real_mat が確定していて、次の real(mat!=0) が来たときに変化判定
                    if last_real_mat is not None and cur_mat != 0:
                        if cur_mat != last_real_mat:
                            from_mat = int(last_real_mat)
                            to_mat = int(cur_mat)
                            ct = _calc_ct(mch, from_mat, to_mat, cur_qty)

                            # ct<=0 は掃除不要（mat=0/同材など）
                            if ct > 0:
                                w0 = int(last_real_end)
                                w1 = int(st_setup)
                                segs = _free_segments(w0, w1, busy_between, dur=ct)

                                events.append({
                                    "mach": mch,
                                    "win_start": w0,
                                    "win_end": w1,
                                    "segments": segs,
                                    "from_mat": from_mat,
                                    "to_mat": to_mat,
                                    "qty": cur_qty,
                                    "ct": ct,
                                })

                            # 基準更新（この行が新しい real）
                            last_real_mat = cur_mat
                            last_real_end = en_proc
                            busy_between = []
                            continue

                    # busy_between に占有を追加（last_real が決まった後だけ）
                    if last_real_end is not None:
                        busy_between.append((st_setup, en_proc))

                    # last_real_* の初期化 / 更新（realのみ）
                    if cur_mat != 0:
                        last_real_mat = cur_mat
                        last_real_end = en_proc
                        busy_between = []

            placeable = [ev for ev in events if ev["segments"]]
            unplaceable = [ev for ev in events if not ev["segments"]]

            # ----------------------------
            # 2) 追加の小さなCP-SATで、同時掃除上限(Max_clean_worker)を守りつつ開始時刻を決める
            # ----------------------------
            clean_rows = []

            if placeable:
                cap = int(self.max_clean_worker) if self.max_clean_worker is not None else 1
                cap = max(1, cap)

                mdl = cp_model.CpModel()
                starts = []
                intervals = []
                slacks = []

                horizon = int(df_n["end_proc"].max()) if "end_proc" in df_n.columns else 1000000
                horizon = max(horizon, 1)

                for i, ev in enumerate(placeable):
                    w0 = int(ev["win_start"])
                    w1 = int(ev["win_end"])
                    dur = int(ev["ct"])

                    # 置ける範囲
                    s = mdl.NewIntVar(w0, max(w0, w1 - dur), f"clean_s_{i}")
                    e = mdl.NewIntVar(w0 + dur, max(w0 + dur, w1), f"clean_e_{i}")
                    mdl.Add(e == s + dur)
                    iv = mdl.NewIntervalVar(s, dur, e, f"clean_iv_{i}")

                    segs = ev["segments"]
                    if len(segs) == 1:
                        a, b = segs[0]
                        mdl.Add(s >= int(a))
                        mdl.Add(s <= int(b) - dur)
                    else:
                        sels = []
                        for k, (a, b) in enumerate(segs):
                            bsel = mdl.NewBoolVar(f"clean_sel_{i}_{k}")
                            sels.append(bsel)
                            mdl.Add(s >= int(a)).OnlyEnforceIf(bsel)
                            mdl.Add(s <= int(b) - dur).OnlyEnforceIf(bsel)
                        mdl.Add(sum(sels) == 1)

                    slack = mdl.NewIntVar(0, horizon, f"clean_slack_{i}")
                    mdl.Add(slack == int(w1) - (s + dur))

                    starts.append(s)
                    intervals.append(iv)
                    slacks.append(slack)

                mdl.AddCumulative(intervals, [1] * len(intervals), cap)
                mdl.Minimize(sum(slacks))

                slv = cp_model.CpSolver()
                slv.parameters.max_time_in_seconds = 2.0
                slv.parameters.num_search_workers = 1
                st = slv.Solve(mdl)

                if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    for i, ev in enumerate(placeable):
                        dur = int(ev["ct"])
                        cs = int(slv.Value(starts[i]))
                        ce = cs + dur
                        clean_rows.append({
                            "job": -1,
                            "route": 0,
                            "step": 0,
                            "mach": int(ev["mach"]),
                            "work": -1,
                            "setup": 0,
                            "proc": dur,
                            "prio": 0,
                            "mat": 0,
                            "mat_wait": 0,
                            "start": cs,
                            "end": ce,
                            "start_setup": cs,
                            "end_setup": cs,
                            "start_proc": cs,
                            "end_proc": ce,
                            "active": 1,
                            "qty": 0,
                        })
                else:
                    # 置けなければ全部 unplaceable 扱いへ
                    unplaceable = events

            # ----------------------------
            # 3) 置けないイベントは従来方式（次ジョブ直前に固定）で挿入（※重複の可能性あり）
            # ----------------------------
            if unplaceable:
                for ev in unplaceable:
                    dur = int(ev.get("ct", max(0, self.clean_time)))
                    if dur <= 0:
                        continue

                    w0 = int(ev["win_start"])
                    w1 = int(ev["win_end"])

                    cs = w1 - dur
                    if cs < w0:
                        cs = w0
                    ce = cs + dur

                    clean_rows.append({
                        "job": -1,
                        "route": 0,
                        "step": 0,
                        "mach": int(ev["mach"]),
                        "work": -1,
                        "setup": 0,
                        "proc": dur,
                        "prio": 0,
                        "mat": 0,
                        "mat_wait": 0,
                        "start": cs,
                        "end": ce,
                        "start_setup": cs,
                        "end_setup": cs,
                        "start_proc": cs,
                        "end_proc": ce,
                        "active": 1,
                        "qty": 0,
                    })

            # ----------------------------
            # 4) 返却
            # ----------------------------
            if clean_rows:
                df_clean = pd.DataFrame(clean_rows)
                out = pd.concat([df_n, df_clean], ignore_index=True)
            else:
                out = df_n

            out = out.sort_values(["mach", "start_proc"]).reset_index(drop=True)
            return out
    
    def check_machine_overlap_total(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        df_n = df[df["job"] != -1].copy()
        for mch, block in df_n.groupby("mach"):
            # ★ mach=0（ダミー機械）はチェック不要
            if int(mch) == 0:
                continue
            block = block.sort_values("start_setup")
            last_row = None
            last_end = None
            for _, r in block.iterrows():
                ts = int(r["start_setup"])
                te = int(r["end_proc"])
                if last_end is not None and ts < last_end:
                    rows.append({
                        "mach": mch,
                        "job_a": int(last_row["job"]),
                        "job_b": int(r["job"]),
                        "a_start_total": int(last_row["start_setup"]),
                        "a_end_total":   int(last_row["end_proc"]),
                        "b_start_total": ts,
                        "b_end_total":   te,
                    })
                if last_end is None or te > last_end:
                    last_row = r
                    last_end = te
        return pd.DataFrame(rows)

    def check_worker_overlap_setup(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        # job=-1（掃除）、work<0 は除外
        df_n = df[(df["job"] != -1) & (df["work"] >= 0)].copy()

        # ★ 追加：mach=0（ダミー）を worker 重複チェックからも除外
        df_n = df_n[df_n["mach"] != 0]

        # ★ 追加：setup>0 の行だけが「作業者を実際にブロックする」対象
        df_n = df_n[df_n["setup"] > 0]
        
        for w, block in df_n.groupby("work"):
            block = block.sort_values("start_setup")
            last_row = None
            last_end = None
            for _, r in block.iterrows():
                ts = int(r["start_setup"])
                te = int(r["end_setup"])
                if last_end is not None and ts < last_end:
                    rows.append({
                        "work": w,
                        "job_a": int(last_row["job"]),
                        "job_b": int(r["job"]),
                        "a_start_setup": int(last_row["start_setup"]),
                        "a_end_setup":   int(last_row["end_setup"]),
                        "b_start_setup": ts,
                        "b_end_setup":   te,
                    })
                if last_end is None or te > last_end:
                    last_row = r
                    last_end = te
        return pd.DataFrame(rows)
 
    def check_step_order(self, df: pd.DataFrame, min_gap: int = 0) -> pd.DataFrame:
        rows = []
        # ★ 変更：実ジョブ（proc>0）のみ step順をチェック
        df_n = df[(df["job"] != -1) & (df["proc"] > 0)].copy()
        
        for (job, route), block in df_n.groupby(["job", "route"]):
            block = block.sort_values("step")
            prev = None
            for _, r in block.iterrows():
                if prev is not None:
                    required = int(prev["end_proc"]) + int(min_gap)
                    if int(r["start_setup"]) < required:
                        rows.append({
                            "job": int(job),
                            "route": int(route),
                            "prev_step": int(prev["step"]),
                            "next_step": int(r["step"]),
                            "prev_end_proc": int(prev["end_proc"]),
                            "next_start_setup": int(r["start_setup"]),
                            "min_gap": int(min_gap),
                        })
                prev = r
        return pd.DataFrame(rows)
 
    def check_clean_gap(
        self,
        df: pd.DataFrame,
        clean_time_const: int = 60,
        *,
        clean_rules: dict | None = None,
        default_clean_time: int | None = None,
    ) -> pd.DataFrame:
        """
        掃除ギャップ違反チェック
        - clean_rules があれば可変（clean_time()関数に委譲）
        - clean_rules がなければ固定（clean_time 引数）
        重要: mat=0（休み/停止/メンテ等）は「掃除判定の基準(前側)」にしない
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # 掃除行はチェック対象外（見える化用）
        d = df[df["job"] != -1].copy()

        # 必要列の安全化
        for c in ["mach", "mat", "start_setup", "end_proc", "qty"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
            else:
                if c == "qty":
                    d["qty"] = 1

        # 機械ごと・開始順
        d = d.sort_values(["mach", "start_setup", "start_proc"]).reset_index(drop=True)

        violations = []

        # default_clean_time の決定（可変モードのフォールバック）
        if default_clean_time is None:
            default_clean_time = int(self.clean_time)

        for mach, block in d.groupby("mach", sort=True):
            mach = int(mach)
            if mach == 0:
                continue

            last_real_mat = None
            last_real_row = None  # ★ここが重要（mat!=0 の行だけ保持）

            for _, r in block.iterrows():
                cur_mat = int(r.get("mat", 0))
                start_setup = int(r.get("start_setup", 0))
                end_proc = int(r.get("end_proc", start_setup))
                qty_next = int(r.get("qty", 1))

                # mat=0 は掃除判定の「前側」に使わない（ただし順序の中には居る）
                if cur_mat == 0:
                    continue

                # 直近の実材料がある & 材料が変わった → 掃除ギャップ判定
                if last_real_mat is not None and cur_mat != last_real_mat:
                    if clean_rules is not None:
                        ct = scheduler.clean_time(
                            mach, last_real_mat, cur_mat, qty_next,
                            clean_rules, default=default_clean_time
                        )
                    else:
                        # 固定モード
                        ct = int(self.clean_time)

                    required_start = int(last_real_row["end_proc"]) + int(ct)

                    if start_setup < required_start:
                        violations.append({
                            "mach": mach,
                            "mat_prev": int(last_real_mat),
                            "mat_next": int(cur_mat),
                            "qty_next": int(qty_next),
                            "end_proc_prev": int(last_real_row["end_proc"]),
                            "start_setup_next": int(start_setup),
                            "required_start_setup": int(required_start),
                            "clean_time": int(ct),
                            "shortage": int(required_start - start_setup),
                        })

                # 実材料の基準更新
                last_real_mat = cur_mat
                last_real_row = r

        return pd.DataFrame(violations)

    def check_schedule_all(self,
                        df: pd.DataFrame,
                        min_gap: int = 1,#許容誤差
                        clean_time_const: int | None = None,
                        *,
                        clean_rules: dict | None = None,
                        default_clean_time: int | None = None,
                        verbose: bool = True,
                        ):
        """
        まとめてチェック（可変掃除時間対応・最終版）

        Parameters
        ----------
        df : DataFrame
            スケジュール結果（job=-1 の掃除行が含まれていてもOK）
        min_gap : int
            工程順序の最小ギャップ
        clean_time_const : int | None
            固定掃除時間でチェックしたい場合のみ指定
            （可変掃除を使う場合は None 推奨）
        clean_rules : dict | None
            可変掃除時間ルール（DB由来）
        default_clean_time : int | None
            clean_rules 未ヒット時のフォールバック値
        verbose : bool
            結果を print するか
        """       
    
        result: dict[str, pd.DataFrame] = {}

        # -------------------------------------------------
        # ① 機械排他（total）
        # -------------------------------------------------
        ng1 = self.check_machine_overlap_total(df)
        result["machine_overlap_total"] = ng1

        # -------------------------------------------------
        # ② 作業者排他（setup）
        # -------------------------------------------------
        ng2 = self.check_worker_overlap_setup(df)
        result["worker_overlap_setup"] = ng2

        # -------------------------------------------------
        # ③ 工程順序
        # -------------------------------------------------
        ng3 = self.check_step_order(df, min_gap=min_gap)
        result["step_order"] = ng3

        # -------------------------------------------------
        # ④ 掃除ギャップ（固定 or 可変）
        # -------------------------------------------------
        if self.clean_rules is not None or clean_time_const is not None:
            ct_const = int(clean_time_const) if clean_time_const is not None else 0

            ng4 = self.check_clean_gap(
                df,
                clean_time_const=ct_const,   # ★名前衝突しない引数名
                clean_rules=self.clean_rules,
                default_clean_time=default_clean_time,
            )
        else:
            ng4 = pd.DataFrame()

        result["clean_gap_violations"] = ng4

        # -------------------------------------------------
        # 表示
        # -------------------------------------------------
        if verbose:

            def _show(name: str, d: pd.DataFrame):
                if isinstance(d, pd.DataFrame) and not d.empty:
                    print(f"[NG] {name}: {len(d)} rows")
                else:
                    print(f"[OK] {name}")

            _show("machine_overlap_total", ng1)
            _show("worker_overlap_setup", ng2)
            _show("step_order", ng3)

            if clean_rules is not None:
                print("[INFO] clean_gap: variable mode (self.clean_rules)")
            elif clean_time_const is not None:
                print(f"[INFO] clean_gap: constant mode (clean_time={clean_time_const})")

            _show("clean_gap_violations", ng4)

        return result

    def check(self):
        if self.df_result:
            df_list = []
    
            #初期ロックを出力に含める
            df_list.append(self.df_lock.copy())
            
            #ループ結果
            df_list.append(pd.concat(self.df_result, ignore_index=True))
            
            df_all_raw = pd.concat(df_list, ignore_index=True)
            
            # グローバルに掃除 job=-1 行を付ける
            df_all = self.add_clean_rows_global(df_all_raw)
            
            #ここで不要カラム削除
            DROP_COLS = ["mat_wait","start","end"]
            df_all = df_all.drop(columns=DROP_COLS, errors="ignore")

            #チェック関数利用
            res = self.check_schedule_all(
                df_all,
                min_gap=0,
                clean_rules=self.clean_rules,
                default_clean_time=self.clean_time,  # ルール未ヒット時のフォールバック
                verbose=True,
            )

            #カラム並び替え
            col = [
                "job","route","step","mach","work","mat","prio","setup","qty",
                "start_setup","end_setup","start_proc","end_proc","active",
                ]
            df_all=df_all[col]

            return df_all ,res

if __name__=="__main__":
    pass