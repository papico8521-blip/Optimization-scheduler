import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class ScheduleVisualizer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # カンマ除去 & 数値化
        time_cols = ["start_setup", "end_setup",
                     "start_proc", "end_proc"]
        for col in time_cols:
            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(",", "")
                .astype(float)
            )

        # active=1のみ
        self.df = self.df[self.df["active"] == 1]

    # ==============================
    # ガントチャート
    # ==============================
    def plot_gantt_by_machine(self, save_html=False, path="gantt.html"):

        fig = go.Figure()
        machines = sorted(self.df["mach"].unique())

        # ---------- 材料カラー設定 ----------
        materials = sorted(self.df["mat"].unique())
        palette = px.colors.qualitative.Safe

        mat_color_map = {
            m: palette[i % len(palette)]
            for i, m in enumerate(materials)
        }


        for mach in machines:
            df_m = self.df[self.df["mach"] == mach]

            for _, row in df_m.iterrows():

                # ---------- SETUP ----------
                setup_dur = row["end_setup"] - row["start_setup"]
                if setup_dur > 0:

                    fig.add_trace(go.Bar(
                        x=[setup_dur],
                        y=[f"M{mach}"],
                        base=row["start_setup"],
                        orientation="h",
                        width=0.5,
                        marker=dict(
                            color="orange",
                            line=dict(
                                color="black" if row["job"] == -1
                                else mat_color_map[row["mat"]],
                                width=3
                            )
                        ),
                        customdata=[[
                            row["job"],
                            row["route"],
                            row["step"],
                            row["mach"],
                            row["mat"],
                            row["prio"],
                            row["work"],
                            row["qty"],
                            row["end_setup"]
                        ]],

                        hovertemplate=(
                            "<b>SETUP</b><br>"
                            "Job: %{customdata[0]}<br>"
                            "Route: %{customdata[1]} Step: %{customdata[2]}<br>"
                            "Machine: %{customdata[3]}<br>"
                            "Material: %{customdata[4]}<br>"
                            "Priority: %{customdata[5]}<br>"
                            "Worker: %{customdata[6]}<br>"
                            "Qty: %{customdata[7]}<br>"
                            "Start: %{base}<br>"
                            "End: %{customdata[8]}<br>"
                            "<extra></extra>"
                        ),

                        showlegend=False
                    ))

                # ---------- PROC ----------
                proc_dur = row["end_proc"] - row["start_proc"]
                if proc_dur > 0:

                    color = "red" if row["job"] == -1 else "steelblue"

                    fig.add_trace(go.Bar(
                        x=[proc_dur],
                        y=[f"M{mach}"],
                        base=row["start_proc"],
                        orientation="h",
                        width=0.5,
                        marker=dict(
                            color=color,
                            line=dict(
                                color="black" if row["job"] == -1
                                else mat_color_map[row["mat"]],
                                width=3
                            )
                        ),
                        customdata=[[
                            row["job"],
                            row["route"],
                            row["step"],
                            row["mach"],
                            row["mat"],
                            row["prio"],
                            row["work"],
                            row["qty"],
                            row["end_proc"]
                        ]],

                        hovertemplate=(
                            "<b>PROC</b><br>"
                            "Job: %{customdata[0]}<br>"
                            "Route: %{customdata[1]} Step: %{customdata[2]}<br>"
                            "Machine: %{customdata[3]}<br>"
                            "Material: %{customdata[4]}<br>"
                            "Priority Group: %{customdata[5]}<br>"
                            "Worker: %{customdata[6]}<br>"
                            "Qty: %{customdata[7]}<br>"
                            "Start: %{base}<br>"
                            "End: %{customdata[8]}<br>"
                            "<extra></extra>"
                        ),

                        showlegend=False
                    ))

        fig.update_layout(
            barmode="overlay",
            title="Machine Schedule Gantt",
            xaxis_title="Time",
            yaxis_title="Machine",
            height=400 + len(machines) * 90,
            template="plotly_white"
        )

        if save_html:
            fig.write_html(path)
      
        fig.show()
    # ==============================
    # KPI表示
    # ==============================
    def calc_kpi(self):

        makespan = self.df["end_proc"].max()

        dummy_time = self.df[self.df["job"] == -1] \
            .apply(lambda r: r["end_proc"] - r["start_proc"], axis=1) \
            .sum()

        total_proc = self.df \
            .apply(lambda r: r["end_proc"] - r["start_proc"], axis=1) \
            .sum()

        return {
            "makespan": makespan,
            "清掃時間 合計": dummy_time,
            "各mach + 各掃除時間 合計": total_proc
        }


if __name__=="__main__":
    pass