import random 
import pandas as pd
import numpy as np

from collections import defaultdict
from ortools.sat.python import cp_model

def clean_time(mach, mat_from, mat_to, qty, rules, default=60):
    mach = int(mach)
    mat_from = int(mat_from)
    mat_to   = int(mat_to)
    qty = max(1, int(qty))

    # ★材料が未定義/なし(0や-1)なら掃除しない
    if mat_from <= 0 or mat_to <= 0:
        return 0

    # 同一材料なら掃除しない
    if mat_from == mat_to:
        return 0

    key = (mach, mat_from, mat_to)
    if key in rules:
        for qmin, qmax, ct in rules[key]:
            if qmin <= qty <= qmax:
                return int(ct)

    return int(default)

class Solver:
    def __init__(self, seed=42):
        self.seed=seed
        
    def solve_job(
                self,
                jobs,
                locked_jobs=None,
                horizon=0,
                CLEAN_TIME=60,
                MIN_GAP=0,
                cpu_count=1,
                time_limit=60,
                clean_rule=None
                ):
        if clean_rule is None:
            raise ValueError("clean_rule is None")
        
        def norm_row(j):
            """
            正規化関数
            """
            jj = dict(j)
            jj["job"]   = int(jj["job"])
            jj["route"] = int(jj.get("route", 1))
            jj["step"]  = int(jj["step"])
            jj["mach"]  = int(jj["mach"])
            jj["work"]  = int(jj.get("work", -1))
            jj["setup"] = int(jj.get("setup", 0))
            jj["proc"]  = int(jj.get("proc", 0))
            jj["mat"]   = int(jj.get("mat", -1))
            jj["prio"]  = int(jj.get("prio", 0))
            jj["mat_wait"] = int(jj.get("mat_wait", 0))
            jj["qty"] = int(jj.get("qty",1))
            return jj
        
        #安全にソルバーが処理できる形に正規化
        jobs = [norm_row(j) for j in jobs]
        # 候補を (job,route,step) ごとに束ねる
        cands_by_jrs = defaultdict(list)
        for j in jobs:
            cands_by_jrs[(j["job"], j["route"], j["step"])].append(j)

        # 機械一覧
        machines = sorted({j["mach"] for j in jobs})

        #horizon設定=0の場合は240日分のホライゾン確保
        if horizon == 0: 
            horizon = (60 * 24 * 240) // 1  # かなり大きめ（過大ホライゾン前提）
        
        # locked_jobs から機械・作業者を補完
        locked_jobs = locked_jobs or []
        if locked_jobs:
            #ダミー機除外処理
            machines = sorted(
                    set(machines) | {int(lj["mach"]) for lj in locked_jobs if int(lj["mach"]) > 0}
                )
            
        # ---------------------------------------   
        # ---------------- model構築-------------
        # ---------------------------------------    
        random.seed(self.seed)
        np.random.seed(self.seed)
        m = cp_model.CpModel()

        Active, SS, ES, SP, EP = {}, {}, {}, {}, {}
        IV_SETUP, IV_PROC, IV_TOTAL = {}, {}, {}
        CandKey = []

        RouteSel = {}
        routes_by_job = defaultdict(set)

        # 各候補ごとに可変Intervalを作成
        for (job, route, step), cand_list in cands_by_jrs.items():
            for idx, j in enumerate(cand_list):
                key = (job, route, step, idx)
                CandKey.append(key)
                a = m.NewBoolVar(f"a_{key}")
                Active[key] = a

                # proc区間
                SP[key] = m.NewIntVar(0, horizon, f"sp_{key}")
                EP[key] = m.NewIntVar(0, horizon, f"ep_{key}")
                dur_proc = max(1, j["proc"])
                IV_PROC[key] = m.NewOptionalIntervalVar(
                    SP[key], dur_proc, EP[key], a, f"iprc_{key}"
                )
                m.Add(EP[key] - SP[key] == dur_proc).OnlyEnforceIf(a)

                # setup区間
                SS[key] = m.NewIntVar(0, horizon, f"ss_{key}")
                ES[key] = m.NewIntVar(0, horizon, f"es_{key}")
                if j["setup"] > 0 and j["work"] >= 0:
                    dur_setup = max(1, j["setup"])
                    IV_SETUP[key] = m.NewOptionalIntervalVar(
                        SS[key], dur_setup, ES[key], a, f"iset_{key}"
                    )
                    m.Add(ES[key] - SS[key] == dur_setup).OnlyEnforceIf(a)
                    # 連結 setup→proc
                    m.Add(SP[key] == ES[key]).OnlyEnforceIf(a)
                else:
                    IV_SETUP[key] = None
                    m.Add(SS[key] == SP[key]).OnlyEnforceIf(a)
                    m.Add(ES[key] == SP[key]).OnlyEnforceIf(a)

                # 機械占有トータル区間 (setup開始 → proc終了)
                total_dur = max(1, j["setup"] + j["proc"])
                IV_TOTAL[key] = m.NewOptionalIntervalVar(
                    SS[key], total_dur, EP[key], a, f"itot_{key}"
                )

                # 材料待ち
                if j["mat_wait"] > 0:
                    m.Add(SP[key] >= j["mat_wait"]).OnlyEnforceIf(a)

            # --- route選択変数（job,route） ---
            rkey = (job, route)
            if rkey not in RouteSel:
                RouteSel[rkey] = m.NewBoolVar(f"r_{job}_{route}")
            routes_by_job[job].add(route)

            # --- (job,route,step) の候補は「routeが選ばれたら1つ、選ばれなければ0」 ---
            xs = [Active[(job, route, step, i)] for i in range(len(cand_list))]
            if len(xs) == 1:
                m.Add(xs[0] == RouteSel[rkey])
            else:
                m.Add(sum(xs) == RouteSel[rkey])

        for job, routes in routes_by_job.items():
            rvars = [RouteSel[(job, r)] for r in sorted(routes)]
            if len(rvars) == 1:
                m.Add(rvars[0] == 1)
            else:
                m.Add(sum(rvars) == 1)  

        # ---------------- precedence (step順) ----------------
        steps_by_jr = defaultdict(set)
        for (job, route, step) in cands_by_jrs.keys():
            steps_by_jr[(job, route)].add(step)

        for (job, route), steps in steps_by_jr.items():
            steps = sorted(steps)
            for s, t in zip(steps[:-1], steps[1:]):
                cands_s = cands_by_jrs[(job, route, s)]
                cands_t = cands_by_jrs[(job, route, t)]
                for i in range(len(cands_s)):
                    key_s = (job, route, s, i)
                    for k in range(len(cands_t)):
                        key_t = (job, route, t, k)
                        m.Add(SS[key_t] >= EP[key_s] + MIN_GAP).OnlyEnforceIf(
                            [Active[key_s], Active[key_t]]
                        )

        # =========================================================
        # 掃除ギャップ制約（同一 solve 内、可動ジョブ同士）
        # =========================================================
        for mch in machines:
            # この機械に割り当て可能な全候補キー
            keys_m = []
            for key in CandKey:
                j = cands_by_jrs[(key[0], key[1], key[2])][key[3]]
                if j["mach"] == mch:
                    keys_m.append((key, j))
            if not keys_m:
                continue

            # 全ペアに対して（順序はソルバーに任せる）
            n = len(keys_m)
            for i in range(n):
                (a, ja) = keys_m[i]
                for j_idx in range(i + 1, n):
                    (b, jb) = keys_m[j_idx]
                    # ▼ 追加：どちらかの mat が 0 なら掃除不要自作ロックジョブ用
                    if ja["mat"] == 0 or jb["mat"] == 0:
                        continue
                    if ja["mat"] == jb["mat"]:
                        continue  # 同材ならギャップ不要

                    # どちらが先かを選ばせる
                    a_before_b = m.NewBoolVar(f"a_before_b_{a}_{b}")
                    both = m.NewBoolVar(f"both_{a}_{b}")
                    # both = Active[a] AND Active[b]
                    m.Add(both <= Active[a])
                    m.Add(both <= Active[b])
                    m.Add(both >= Active[a] + Active[b] - 1)

                    #---------------------------------------------------------
                    # a→b の場合: EP[a] + CLEAN_TIME <= SS[b]
                    #m.Add(EP[a] + CLEAN_TIME <= SS[b]).OnlyEnforceIf([both, a_before_b])
                    # b→a の場合: EP[b] + CLEAN_TIME <= SS[a]
                    #m.Add(EP[b] + CLEAN_TIME <= SS[a]).OnlyEnforceIf([both, a_before_b.Not()])
                    #----------------------------------------------------------
                    ct_ab = clean_time(
                            mch,
                            ja["mat"],
                            jb["mat"],
                            qty=jb.get("qty", 1),   # ← 後ジョブの個数を使うのが自然
                            rules=clean_rule,
                            default=CLEAN_TIME
                        )

                    ct_ba = clean_time(
                        mch,
                        jb["mat"],
                        ja["mat"],
                        qty=ja.get("qty", 1),
                        rules=clean_rule,
                        default=CLEAN_TIME
                    )

                    m.Add(EP[a] + ct_ab <= SS[b]).OnlyEnforceIf([both, a_before_b])
                    m.Add(EP[b] + ct_ba <= SS[a]).OnlyEnforceIf([both, a_before_b.Not()])

        # =========================================================
        # locked_jobs → 固定Interval化（掃除ギャップの相手にも使う）
        # =========================================================
        fixed_by_mach = defaultdict(list)
        fixed_by_work = defaultdict(list)
        locked_info_by_mach = defaultdict(list)  # 掃除ギャップ用に start_setup/end_proc/mat を保持

        for lj in locked_jobs:
            mach = int(lj["mach"])
            work = int(lj.get("work", -1))
            setup = int(lj.get("setup", 0))
            start_proc = int(lj["start"])
            end_proc = int(lj["end"])
            #mat = int(lj.get("mat", -1))
            mat = int(lj.get("mat", 0))
            qty = int(lj.get("qty", 1))  # ★追加

            # 段取開始 = 加工開始 - setup（最低0）
            start_setup = max(0, start_proc - setup)
            total_dur = max(1, end_proc - start_setup)

            # -------------------------
            # 機械固定（mach > 0 のときだけ）
            if mach > 0:
                iv_m = m.NewIntervalVar(start_setup, total_dur, end_proc,
                                        f"fixed_m{mach}_job{lj['job']}_{start_proc}")
                fixed_by_mach[mach].append(iv_m)
        
                # ★掃除ギャップ用（ここで1回だけ）
                locked_info_by_mach[mach].append({
                    "job": int(lj["job"]),
                    "start_setup": start_setup,
                    "end_proc": end_proc,
                    "mat": mat,
                    "qty": qty,
                })   
                
            # 作業者固定（setupのみ）
            if work >= 0 and setup > 0:
                iv_w = m.NewIntervalVar(
                    start_setup, setup, start_proc,
                    f"fixed_w{work}_job{lj['job']}_{start_setup}"
                )
                fixed_by_work[work].append(iv_w)

        # =========================================================
        # 掃除ギャップ制約（locked_jobs vs 可動ジョブ）
        # =========================================================
        for mch in machines:
            locked_infos = locked_info_by_mach.get(mch, [])
            if not locked_infos:
                continue
            # この機械に割り当て可能な全候補キー
            keys_m = []

            for key in CandKey:
                j = cands_by_jrs[(key[0], key[1], key[2])][key[3]]
                if j["mach"] == mch:
                    keys_m.append((key, j))

            if not keys_m:
                continue

            for (key, j) in keys_m:
                for li in locked_infos:
                    # locked か可動どちらかの mat が 0 → 掃除不要自作ロックジョブ用
                    if li["mat"] == 0 or j["mat"] == 0:
                        continue
                    if li["mat"] == j["mat"]:
                        continue  # 同材なら掃除不要

                    # locked と可動ジョブのどちらが先かを選ぶ
                    # locked は常に存在、可動側だけ Active[key] を見る
                    l_before_j = m.NewBoolVar(f"locked_before_{li['job']}_{key}")

                    #------------------------------------------------------------------------------
                    # job が locked の前に来る場合: EP[j] + CLEAN_TIME <= start_setup_locked
                    #m.Add(EP[key] + CLEAN_TIME <= li["start_setup"]).OnlyEnforceIf(
                    #    [Active[key], l_before_j.Not()]
                    #)
                    # locked が先に来る場合: end_locked + CLEAN_TIME <= SS[j]
                    #m.Add(li["end_proc"] + CLEAN_TIME <= SS[key]).OnlyEnforceIf(
                    #    [Active[key], l_before_j]
                    #)
                    #------------------------------------------------------------------------------
                    ct_lj = clean_time(
                            mch,
                            li["mat"],
                            j["mat"],
                            qty=j.get("qty", 1),
                            rules=clean_rule,
                            default=CLEAN_TIME,
                            )
                
                    ct_jl = clean_time(
                        mch,
                        j["mat"],
                        li["mat"],
                        qty=li.get("qty", 1),
                        rules=clean_rule,
                        default=CLEAN_TIME,
                        )

                    m.Add(EP[key] + ct_jl <= li["start_setup"]).OnlyEnforceIf(
                        [Active[key], l_before_j.Not()]
                    )
                    m.Add(li["end_proc"] + ct_lj <= SS[key]).OnlyEnforceIf(
                        [Active[key], l_before_j]
                    )
        # =========================================================
        # Machine: トータル占有 + locked
        # =========================================================
        for mch in machines:
            ivs = []
            # 可動ジョブ
            for (job, route, step), cand_list in cands_by_jrs.items():
                for idx, j in enumerate(cand_list):
                    if j["mach"] != mch:
                        continue
                    key = (job, route, step, idx)
                    ivs.append(IV_TOTAL[key])
            # locked
            ivs += fixed_by_mach.get(mch, [])

            if len(ivs) >= 2:
                m.AddNoOverlap(ivs)

        # =========================================================
        # Worker: setup のみ NoOverlap + locked
        # =========================================================
        workers = sorted({j["work"] for j in jobs if j.get("work", -1) >= 0}
                        | {int(lj.get("work", -1)) for lj in locked_jobs if lj.get("work", -1) >= 0})

        for w in workers:
            ivs = []
            # 可動ジョブ
            for (job, route, step), cand_list in cands_by_jrs.items():
                for idx, j in enumerate(cand_list):
                    if j.get("work", -1) != w:
                        continue
                    key = (job, route, step, idx)
                    if IV_SETUP[key] is not None:
                        ivs.append(IV_SETUP[key])
            # locked
            ivs += fixed_by_work.get(w, [])

            if ivs:
                m.AddNoOverlap(ivs)

        # =========================================================
        # 目的関数：makespan
        # =========================================================
        max_end = m.NewIntVar(0, horizon, "max_end")
        for key in CandKey:
            m.Add(max_end >= EP[key])

        # =========================================================
        # 目的関数：掃除回数の下界
        #   機械ごとの「使用材料種類数 − 1」を最小化
        # =========================================================
        clean_lb_terms = []
        
        for mch in machines:
            # この機械に来る候補ジョブ
            keys_m = []
            for key in CandKey:
                j = cands_by_jrs[(key[0], key[1], key[2])][key[3]]
                if j["mach"] == mch:
                    keys_m.append((key, j))
        
            locked_infos = locked_info_by_mach.get(mch, [])
        
            if not keys_m and not locked_infos:
                continue
        
            # この機械で使われうる材料（mat=0 は除外）
            mats = set()
        
            # 可動ジョブ側
            for key, j in keys_m:
                mat = int(j.get("mat", 0))
                if mat > 0:
                    mats.add(mat)
        
            # locked 側（必ず使用される）
            locked_mats = set()
            for li in locked_infos:
                mat = int(li.get("mat", 0))
                if mat > 0:
                    locked_mats.add(mat)
                    mats.add(mat)
        
            if not mats:
                continue
        
            # 各材料について「使われたか？」の Bool
            used_flags = []
        
            for mat in mats:
                # locked に含まれていれば必ず使用
                if mat in locked_mats:
                    used_flags.append(m.NewConstant(1))
                    continue
        
                # 可動ジョブでこの材料を使ったか
                actives = []
                for key, j in keys_m:
                    if int(j.get("mat", 0)) == mat:
                        actives.append(Active[key])
        
                if not actives:
                    used_flags.append(m.NewConstant(0))
                else:
                    u = m.NewBoolVar(f"used_m{mch}_mat{mat}")
                    m.AddMaxEquality(u, actives)  # u = OR(actives)
                    used_flags.append(u)
        
            # 材料種類数
            mat_kind = m.NewIntVar(0, len(mats), f"mat_kind_m{mch}")
            m.Add(mat_kind == sum(used_flags))
        
            # 掃除回数の下界 = max(mat_kind - 1, 0)
            clean_lb = m.NewIntVar(0, len(mats), f"clean_lb_m{mch}")
            m.Add(clean_lb >= mat_kind - 1)
        
            clean_lb_terms.append(clean_lb)
        
        #clean_lb_total = sum(clean_lb_terms) if clean_lb_terms else 0
        clean_lb_total = cp_model.LinearExpr.Sum(clean_lb_terms)
        m.Minimize(max_end + clean_lb_total)
        # ---------------- solve ----------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = int(time_limit)
        solver.parameters.num_search_workers = int(max(1, cpu_count))
        solver.parameters.random_seed = int(self.seed)
        solver.parameters.stop_after_first_solution = False

        solver.parameters.cp_model_presolve = True #初期解の質向上
        solver.parameters.use_optional_variables = True
        solver.parameters.linearization_level = 0
        solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

        print(">>> entering solver.Solve")
        #status = solver.Solve(m)
        status = solver.SolveWithSolutionCallback(
                        m,
                        cp_model.ObjectiveSolutionPrinter()
                    )
        print(">>> finished solver.Solve")

        print("CpStatus:", solver.StatusName(status))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return pd.DataFrame(), pd.DataFrame()
                
        # ---------------- extract ----------------
        rows = []
        for key in CandKey:
            j = cands_by_jrs[(key[0], key[1], key[2])][key[3]]
            if solver.BooleanValue(Active[key]):
                rows.append({
                    "job": int(j["job"]),
                    "route": int(j["route"]),
                    "step": int(j["step"]),
                    "mach": int(j["mach"]),
                    "work": int(j.get("work", -1)),
                    "start_setup": solver.Value(SS[key]),
                    "end_setup": solver.Value(ES[key]),
                    "start_proc": solver.Value(SP[key]),
                    "end_proc": solver.Value(EP[key]),
                    "setup": int(j["setup"]),
                    "proc": int(j["proc"]),
                    "mat": int(j.get("mat", -1)),
                    "prio": int(j.get("prio", 0)),
                    "qty": int(j.get("qty", 1)),
                    "active": 1,
                })

        df_jobs_full = pd.DataFrame(rows)
        if df_jobs_full.empty:
            return df_jobs_full, df_jobs_full

        # タイムライン（この時点では掃除行は入れない）
        df_timeline = df_jobs_full.sort_values(["mach", "start_proc"]).reset_index(drop=True)

        return df_jobs_full, df_timeline

    def to_locked_jobs(self, df_result: pd.DataFrame):
        """
        solve_jobs() の結果 df_timeline から locked_jobs を生成する。
        ※ job = -1（掃除）はロック対象に含めない（見た目専用）。
        """
        locked = []
        if df_result is None or df_result.empty:
            return locked

        required = {
            "job","route","step","mach","work",
            "setup","proc","mat","prio","start_proc","end_proc"
        }
        if not required.issubset(df_result.columns):
            raise KeyError(
                f"to_locked_jobs: 必須列不足: {sorted(required - set(df_result.columns))}"
            )

        for row in df_result.itertuples(index=False):
            job = int(getattr(row, "job"))
            if job == -1:
                # 掃除はロックしない
                continue

            route = int(getattr(row, "route"))
            step  = int(getattr(row, "step"))
            mach  = int(getattr(row, "mach"))
            work  = int(getattr(row, "work"))
            setup = int(getattr(row, "setup"))
            proc  = int(getattr(row, "proc"))
            mat   = int(getattr(row, "mat"))
            prio  = int(getattr(row, "prio"))
            s_fix = int(getattr(row, "start_proc"))
            e_fix = int(getattr(row, "end_proc"))

            if e_fix <= s_fix:
                e_fix = s_fix + 1

            locked.append({
                "job": job,
                "route": route,
                "step": step,
                "mach": mach,
                "work": work,
                "setup": setup,
                "proc": proc,
                "mat": mat,
                "prio": prio,
                "qty": int(getattr(row, "qty", 1)),
                "mat_wait": 0,
                "start": s_fix,
                "end": e_fix,
            })

        return locked



if __name__=="__main__":
    pass