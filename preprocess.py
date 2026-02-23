import numpy as np
import pandas as pd 

class SearchOptimazer:
    """
    探索方針の前処理
    """
    def __init__(self, seed=42):
        self.seed = seed
        
    def strategy_sort(self, df, g):
        """
        材料ソート + 候補順をソルバーへ正しく渡せるようにする
        """
        df = df[df["group"] == g].copy()           
        df = df.sort_values(by=["mat"], kind="mergesort")
            
        # 骨格保証：工程順
        df = df.sort_values(by=["job", "route", "step"], kind="mergesort")

        # 偏った選択がないように選択肢をシャッフルする
        df = self.shuffle_mach_work_candidates(df)

        return df

    def shuffle_mach_work_candidates(self, df):
        """
        job・route・step 単位で、
        mach/work の候補順だけをシャッフルする安全な関数。

        - 行の意味（mach, work, setup, proc 等）は一切変更しない
        - job/route/step の構造は壊さない
        - 同一条件時間が存在する場合の、選択結果の偏りを抑制
        """
        rng = np.random.default_rng(self.seed)

        blocks = []
        for _, g in df.groupby(["job", "route", "step"], sort=False):
            # 同一 job/route/step 内で順番だけをシャッフル
            g2 = g.sample(
                frac=1,
                random_state=rng.integers(1_000_000_000)
            )
            blocks.append(g2)

        return pd.concat(blocks).reset_index(drop=True)

if __name__=="__main__":
    pass