import numpy as np

def compare_policies(df, supervised_probs, threshold, rl_actions=None):
    supervised_actions = (supervised_probs < threshold).astype(int)

    df_comp = df.copy()
    df_comp["supervised_prob"] = supervised_probs
    df_comp["supervised_action"] = supervised_actions

    if rl_actions is not None:
        df_comp["rl_action"] = rl_actions
        df_disagree = df_comp[df_comp["supervised_action"] != df_comp["rl_action"]]
        return df_comp, df_disagree

    else:
        df_comp["rl_action"] = np.nan
        return df_comp, None
