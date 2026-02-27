"""
Optimization Scheduler 実行エントリーポイント

本スクリプトはローリングホライゾン型スケジューリング処理を実行します。

処理フロー：
データ生成 → 前処理 → 最適化計算 → 整合性検証 → 可視化

"""

import pandas as pd

import data_generator
import preprocess
import scheduler
import consistency_check
import visualizer

# =========================
# 設定値 (デモ用パラメータ)
# =========================
CONFIG = {
    "seed": 42,                 # 乱数シード（再現性確保）
    "loop_num": 6,              # 優先度グループ数（ローリング投入回数）
    "horizon": 60 * 24 * 240,   # 240日のホライゾン
    "clean_time": 60,           # 基本清掃時間（分）
}

def main():
    #--------------------------------------------------
    # テストデータ生成 優先度に応じてグループ分割を実施
    #--------------------------------------------------
    job_maker = data_generator.JobMaker(CONFIG["seed"])
    jobs = job_maker.get_baseline()
    jobs = job_maker.split_prio_group(CONFIG["loop_num"])

    df_job = pd.DataFrame(jobs)
    print('\n----選択肢一覧-------------')
    print(df_job.head(10))

    #-----------------------------------------------------------
    # 設備・作業者リソースブロック生成
    # jobid >= 50 をブロック用途として使用
    # resource : (machID,workID)  標準デモ仕様では、machID & workID =0~3
    # machine,workどちらか一方をブロックする場合はID=0で利用 
    # block_time : (start,end)    0 <= start <= end で利用
    #------------------------------------------------------------
    resource_block = job_maker.add_manual_block(
                                jobid=50,
                                block_resource=(0,1),
                                block_time=(10,100)
                                )
    df_block = pd.DataFrame(resource_block)

    print('\n----リソースブロック一覧-------------')
    print(df_block.head(10))

    #----------------------------------------------------------
    # 掃除ルール生成
    # mach別、mat別、qty別を扱えます。デモ版はqty1~9999
    # 機械毎に材質毎に仕事ボリューム別に推移時間（掃除時間）を扱う
    #----------------------------------------------------------
    rule_generator = data_generator.CleanRuleMaker(job_maker.jobs, job_maker.rng)
    df_clean = rule_generator.make_clean_rule_df()

    print('\n----推移ルール一覧-------------')
    print(df_clean.sample(10))
    print(len(df_clean))

    

    #ソルバー投入用にdict変換
    clean_rule = rule_generator.to_rule_dict(df_clean)

    #優先度大→小
    GROUP_ORDER = list(range(CONFIG["loop_num"]-1, -1, -1))

    print("\n----グールプ別選択肢数------------")
    print(df_job["group"].value_counts().sort_index())

    #初回ロックジョブ有無判定
    if df_block is None:
        df_block = pd.DataFrame(columns=df_job.columns)
        locked_jobs = []
    else:
        locked_jobs = df_block.to_dict("records")

    # --------------------------------------------
    # ローリングホライゾン投入処理
    #
    # 優先度グループ単位でジョブを段階投入し、
    # 確定済み作業をロックしながら再最適化を実施
    # --------------------------------------------
    strategy_compass = preprocess.SearchOptimazer(CONFIG["seed"])
    solver = scheduler.Solver(CONFIG["seed"])

    #ソルバーの出力を受け取るリスト
    df_result= []

    #df_job.to_excel('move.xlsx',index=False)
    #df_block.to_excel('block.xlsx',index=False)
    #df_clean.to_excel('clean.xlsx',index=False)

    for g in GROUP_ORDER:
        print(f"=== グループ {g} 投入 ===")

        df_in = strategy_compass.strategy_sort(df_job, g)
        jobs_in = df_in.to_dict("records")

        if len(jobs_in) == 0:
            print("投入グループに含まれるジョブ=0 -> continue")
            continue

        df_jobs_full, df_timeline = solver.solve_job(
            jobs_in,
            locked_jobs=locked_jobs,
            CLEAN_TIME=CONFIG["clean_time"],
            time_limit=60,
            horizon=CONFIG["horizon"],
            clean_rule=clean_rule
        )

        # 解が生成できない場合は計画破綻として中断
        if df_timeline.empty:
            print(f"🟥 グループ {g}: 解無し → ここで中断")
            break

        # 確定作業をロックジョブへ変換し次回計算へ引き継ぐ
        df_lock_src = df_timeline[df_timeline["active"] == 1].copy()
        df_lock_src["prio"] = g

        df_result.append(df_lock_src)
        locked_jobs.extend(solver.to_locked_jobs(df_lock_src))

    #------------------------------------------
    # スケジュール整合性検証
    # （リソース競合・工程順序・清掃制約）
    #------------------------------------------
    checker = consistency_check.Cheker(df_block, df_result, clean_rule)
    df_all, res = checker.check()

    #df_all.to_excel('result.xlsx', index=False)

    #------------------------------------------
    # ガントチャートによるスケジュール可視化
    #------------------------------------------
    vis = visualizer.ScheduleVisualizer(df_all)
    vis.plot_gantt_by_machine(save_html=True)

    print(f'\n{vis.calc_kpi()}\n\n')

if __name__=="__main__":
    main()
    pass