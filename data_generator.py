import random
import math
import pandas as pd
import numpy as np

class JobMaker:
    """
    ダミージョブ情報を作成するクラス
    """
    def __init__(self,seed = 42):
        self.rng = random.Random(seed)
        self.jobs = None
        self.blocks = []

    def get_baseline(
        self,
        jobs: int = 10,
        routes: int = 2,
        steps: int = 3,
        machines: int = 3,
        workers: int = 3,
        mat: int = 4,
    ):
        """
        job: 仕事数
        routes: ルートパターン数（最大）
        steps: 1ルート内の工程数(最大)
        machines: 1step内の機械候補数（最大）
        workers: 1machine内の作業者候補数（最大）
        mat: 素材種類数（最大）
        """
        if jobs > 20:
            raise ValueError("デモはジョブ数上限規制 = 20です")
        
        result=[]
        for job_id in range(1, jobs + 1):
            # ✅ ランダムは材料のみ（job単位で固定）
            mat_id = self.rng.randint(1, mat)
            for route_id in range(1, routes + 1):
                for step_no in range(1, steps + 1):
                    for mach_id in range(1, machines + 1):
                        for worker_id in range(1, workers + 1):
                            result.append(
                                {
                                    "job": job_id,
                                    "route": route_id,
                                    "step": step_no,
                                    "mach": mach_id,
                                    "work": worker_id,
                                    "mat": mat_id,
                                }
                            )
        self.jobs = result
        self.add_info()
        return self.jobs
    
    def get_realistic(
        self,
        jobs: int = 10,
        routes: int = 2,
        steps: int = 3,
        machines: int = 3,
        workers: int = 3,
        mat: int = 4,
    ):
        """
        realistic版（人は必ずworkers分展開）
        - step数はrouteごとに変動
        - machine候補はstepごとにランダム抽出
        - workerは必ず全展開（1～workers）
        """
        if jobs > 20:
            raise ValueError("デモはジョブ数上限規制 = 20です")
        
        result = []

        for job_id in range(1, jobs + 1):

            mat_id = self.rng.randint(1, mat)

            for route_id in range(1, routes + 1):

                step_count = self.rng.randint(1, steps)

                for step_no in range(1, step_count + 1):

                    machine_count = self.rng.randint(1, machines)

                    mach_ids = self.rng.sample(
                        range(1, machines + 1),
                        machine_count
                    )

                    for mach_id in mach_ids:

                        # 🔥 人は必ず full 展開
                        for worker_id in range(1, workers + 1):

                            result.append(
                                {
                                    "job": job_id,
                                    "route": route_id,
                                    "step": step_no,
                                    "mach": mach_id,
                                    "work": worker_id,
                                    "mat": mat_id,
                                }
                            )
        self.jobs = result
        self.add_info()  
        return self.jobs
    
    def add_info(self):
        self.add_prio()
        self.add_qty()
        self.add_mat_wait()
        self.add_setup_time()
        self.add_proc()
 
    def add_setup_time(self):
        """
        各レコードに setup_time を追加する
        30～100 の 10刻みランダム
        """

        new_jobs = []

        for row in self.jobs:
            setup_time = self.rng.randrange(30, 101, 10)
            # 30 <= x < 101 なので 100まで含まれる

            new_row = row.copy()
            new_row["setup"] = setup_time

            new_jobs.append(new_row)

        self.jobs = new_jobs
        return

    def add_qty(self):
        """
        同一jobidごとに qty を 30～100 のランダム値で付与する
        """

        # jobごとにqtyを決める
        job_qty_map = {}

        for row in self.jobs:
            job_id = row["job"]

            if job_id not in job_qty_map:
                job_qty_map[job_id] = self.rng.randrange(30, 101, 10)

        # 付与
        new_jobs = []

        for row in self.jobs:
            new_row = row.copy()
            new_row["qty"] = job_qty_map[row["job"]]
            new_jobs.append(new_row)
        
        self.jobs = new_jobs
        return     

    def add_proc(self, sigma: float = 0.3):
        """
        - (job, route, step) 単位で base_time
        - 機械ごとに分散
        - proc = base_time × machine_factor × qty
        - キーは増やさない（base_time, procのみ）
        """

        step_base_map = {}
        machine_factor_map = {}

        # ① 純粋な工程標準時間
        for row in self.jobs:
            step_key = (row["job"], row["route"], row["step"])
            if step_key not in step_base_map:
                step_base_map[step_key] = self.rng.randint(1, 5)

        # ② 機械性能差（内部用）
        for row in self.jobs:
            mach = row["mach"]
            if mach not in machine_factor_map:
                factor = 1 + self.rng.gauss(0, sigma)
                machine_factor_map[mach] = max(0.5, factor)

        # ③ proc計算
        new_jobs = []

        for row in self.jobs:
            new_row = row.copy()

            step_key = (row["job"], row["route"], row["step"])
            base_time_min = step_base_map[step_key]
            machine_factor = machine_factor_map[row["mach"]]
            qty = row["qty"]

            # 🔥 分単位で小数を許容
            base_time_real = base_time_min * machine_factor

            proc_min = base_time_real * qty

            new_row["base_time"] = round(base_time_real, 3)
            new_row["proc"] = max(1, math.ceil(proc_min))

            new_jobs.append(new_row)

        self.jobs = new_jobs

    def add_mat_wait(self):
        """
        各ジョブへの材料待ち時間設定
        0=いつでも着手可能という設定値
        """
        for job in self.jobs:
            job["mat_wait"]=0
        return
    
    def add_prio(self):
        """
        各 job_id に対して一意な優先度(prio)を付与する
        min=1, max=job数
        重複なし
        """
        # ① ユニークjob_id取得
        job_ids = sorted({row["job"] for row in self.jobs})
        n = len(job_ids)

        # ② 1～nの順位を作成
        prios = list(range(1, n + 1))

        # ③ シャッフル
        self.rng.shuffle(prios)

        # ④ job_id → prio マッピング
        job_prio_map = dict(zip(job_ids, prios))

        # ⑤ 各レコードに付与
        new_jobs = []

        for row in self.jobs:
            new_row = row.copy()
            new_row["prio"] = job_prio_map[row["job"]]
            new_jobs.append(new_row)

        self.jobs = new_jobs

        return
    
    def add_manual_block(
                        self,
                        *,
                        jobid=50,
                        block_resource=(1,1),
                        block_time=(100,300),
                        ):
        """
        resource:(machID,workID)
        block_time:(start,end)
        """
        job = {
            "job": jobid,
            "mach": block_resource[0],
            "work": block_resource[1],
            "setup": block_time[1] - block_time[0],
            "start": block_time[1],
            "end": block_time[1],
            "start_setup":block_time[0],
            "start_proc":block_time[0],
            "end_setup": block_time[1],
            "end_proc": block_time[1],
            "prio": 1000,
            "active": 1,
            }

        for key in ['route', 'step', 'proc', 'mat', 'mat_wait','qty']:
            job[key] = 0

        self.blocks.append(job)

        return self.blocks

    def split_prio_group(self, n: int):
        """
        prio順に job を並べ、nグループに均等分割
        ソルバーループ回数に合わせる
        """
        df = pd.DataFrame(self.jobs)

        # job単位で代表prio取得
        job_prio = (
            df.groupby("job", as_index=False)["prio"]
            .max()
            .sort_values("prio", ascending=False)
            .reset_index(drop=True)
        )

        # 均等分割
        splits = np.array_split(job_prio["job"], n)

        group_map = {}
        for g, jobs in enumerate(splits):
            for j in jobs:
                group_map[j] = g

        # 元dfへ反映
        df["group"] = df["job"].map(group_map)

        self.jobs = df.to_dict("records")
        return self.jobs
        
class CleanRuleMaker:
    """
    材料切替コストルールをjobsから自動作成
    """
    def __init__(self, jobs, rng):
        self.jobs = jobs
        self.rng = rng

    def make_clean_rule_df(self, min_clean=30, max_clean=100, step=10):
        """
        材料切替掃除ルールを生成
        min_clean, max_clean は min単位 step分刻みランダム
        """

        machs = sorted({row["mach"] for row in self.jobs if row["mach"] > 0})
        mats = sorted({row["mat"] for row in self.jobs if row["mat"] > 0})

        rules = []

        for mach in machs:
            for from_mat in mats:
                for to_mat in mats:

                    if from_mat == to_mat:
                        clean_time = 0
                    else:
                        clean_time = self.rng.randrange(
                            min_clean,
                            max_clean + step,
                            step
                        )

                    rules.append({
                        "mach": mach,
                        "from_mat": from_mat,
                        "to_mat": to_mat,
                        "min_qty":1,
                        "max_qty":9999,
                        "clean_time": clean_time
                    })

        return pd.DataFrame(rules)

    def to_rule_dict(self, df_clean: pd.DataFrame):
        """
        clean_rule_df を → solver用dictへ変換
        (mach, from_mat, to_mat) →
            [(min_qty, max_qty, clean_time), ...]
        """
        rules = {}

        for r in df_clean.itertuples(index=False):

            key = (int(r.mach), int(r.from_mat), int(r.to_mat))

            rules.setdefault(key, []).append(
                (int(r.min_qty), int(r.max_qty), int(r.clean_time))
            )

        # qty下限順にソート（Solver探索安定化）
        for k in rules:
            rules[k].sort(key=lambda x: x[0])

        return rules  

if __name__=='__main__':
    pass