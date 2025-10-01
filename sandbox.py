import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
from textwrap import shorten
from dotenv import load_dotenv

# Optional viz (guarded imports)
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='whitegrid')
    plt.rcParams['figure.dpi'] = 140
except Exception:
    plt = None
    sns = None

# OpenAI
from openai import OpenAI


def load_data(kaggle=True, path=None):
    if path:
        csvs = glob.glob(os.path.join(path, "*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return pd.read_csv(csvs[0])

    if kaggle:
        try:
            import kagglehub
            path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
            csvs = glob.glob(os.path.join(path, "*.csv"))
            if not csvs:
                raise FileNotFoundError("No CSVs in KaggleHub dataset path.")
            return pd.read_csv(csvs[0])
        except Exception as e:
            raise RuntimeError(f"Failed to load Kaggle dataset: {e}")

    raise ValueError("Provide either a local path or use Kaggle mode.")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Parse datetimes
    for c in ['First Response Time', 'Time to Resolution', 'Date of Purchase']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    # Flags
    df['has_ttr'] = (
        ('Time to Resolution' in df.columns and 'First Response Time' in df.columns)
        and df['Time to Resolution'].notna() & df['First Response Time'].notna()
    )
    df['has_csat'] = (
        'Customer Satisfaction Rating' in df.columns and df['Customer Satisfaction Rating'].notna()
    )
    df['has_resolution'] = 'Resolution' in df.columns and df['Resolution'].notna()
    df['has_first_resp'] = 'First Response Time' in df.columns and df['First Response Time'].notna()

    # Synthetic created_at
    frt = df.get('First Response Time')
    dop = df.get('Date of Purchase')
    if frt is not None or dop is not None:
        df['created_at'] = (frt if frt is not None else pd.Series(index=df.index)).fillna(
            dop if dop is not None else pd.NaT
        )
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['weekday'] = df['created_at'].dt.day_name()
        df['hour'] = df['created_at'].dt.hour

    return df


def filter_negative_ttr(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['has_ttr'] & (df['Time to Resolution'] < df['First Response Time'])
    bad = (
        df.loc[mask, ['Ticket ID', 'First Response Time', 'Time to Resolution', 'Ticket Status', 'Ticket Channel']]
        if 'Ticket ID' in df.columns
        else df.loc[mask]
    )
    bad_ratio = len(bad) / (df['has_ttr'].sum() or 1)
    print(
        f"Negative TTR rows: {len(bad)} ({bad_ratio:.1%} of TTR-eligible). Excluding from analysis."
    )
    return df.loc[~mask].copy()


def ticket_type_head(df: pd.DataFrame, coverage=0.8):
    tbl = df['Ticket Type'].value_counts().to_frame('count')
    tbl['share'] = (tbl['count'] / tbl['count'].sum()).round(3)
    tbl['cum_share'] = tbl['share'].cumsum().round(3)
    head_types = tbl.index[tbl['cum_share'] <= coverage].tolist()
    print("Head types (~80% of tickets):", head_types)
    return tbl, head_types


def channel_mix_vs_head(df: pd.DataFrame, head_types):
    ch_all = df['Ticket Channel'].value_counts(normalize=True).mul(100).round(1).to_frame('% overall')
    ch_head = (
        df[df['Ticket Type'].isin(head_types)]['Ticket Channel']
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .to_frame('% in head types')
    )
    out = (
        ch_all.join(ch_head, how='outer')
        .fillna(0)
        .sort_values('% overall', ascending=False)
    )
    try:
        top_overall = ch_all['% overall'].sort_values(ascending=False).head(2)
        msg = 'Top channels overall: ' + ', '.join([
            str(i) + ' (' + str(v) + '%)'
            for i, v in top_overall.items()
        ])
        print(msg)
    except Exception:
        pass
    return out


def make_snippet(row, max_len=400):
    subj = str(row.get('Ticket Subject', '') or '')
    desc = str(row.get('Ticket Description', '') or '')
    txt = (subj.strip() + ' — ' + desc.strip()).strip(' —')
    txt = ' '.join(txt.split())
    return shorten(txt, width=max_len, placeholder='…')


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def build_snippets(df: pd.DataFrame, target_type: str, target_channel: str, N_cap=200):
    subset = df[(df['Ticket Type'] == target_type) & (df['Ticket Channel'] == target_channel)].copy()
    print(f"Subset size for {target_type} via {target_channel}: {len(subset)}")
    if subset.empty:
        raise ValueError("No rows match the selected Type–Channel. Adjust target_type/channel.")
    subset['snippet'] = subset.apply(make_snippet, axis=1)
    N = min(len(subset), N_cap)
    sample = subset.sample(N, random_state=42) if len(subset) > N else subset
    print("Using", len(sample), "snippets for theme extraction")
    return sample


def summarize_with_llm(sample_snippets,
                        model_batch="gpt-4o-mini",
                        model_merge="gpt-4o-mini",
                        batch_size=20,
                        temp_batch=0.2,
                        temp_merge=0.0,
                        max_tokens_batch=700,
                        max_tokens_merge=900):
    client = OpenAI()

    def summarize_batch(snips):
        bullets = "- " + "\n- ".join(snips)
        prompt = f"""You are analyzing customer support tickets. Each line is a short ticket summary.
- Identify 5–8 recurring themes with:
  - (a) concise theme name
  - (b) brief description
  - (c) 2–3 representative phrases (verbatim)
  - (d) an estimated prevalence percentage for this batch; keep batch total ~100%
- Be specific, avoid PII, avoid repeating phrases, and do not use placeholders like {{product_purchased}}; use generic product/device wording.

Tickets:
{bullets}
"""
        resp = client.chat.completions.create(
            model=model_batch,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temp_batch,
            max_tokens=max_tokens_batch,
        )
        return resp.choices[0].message.content

    snippets = list(sample_snippets)
    batches = list(chunked(snippets, batch_size))
    print(f"Batches: {len(batches)} (batch size={batch_size})")

    batch_summaries = []
    for i, b in enumerate(batches, 1):
        t0 = time.time()
        summary = summarize_batch(b)
        batch_summaries.append(summary)
        print(f"Summarized batch {i}/{len(batches)} (size={len(b)}) in {time.time()-t0:.1f}s")

    bulleted_summaries = "\n\n---\n\n".join(batch_summaries)
    merge_prompt = f"""You are given multiple theme summaries extracted from the same ticket set.
Merge them into a single, non-redundant list of 6–10 themes.
For each theme, provide: name, 1–2 sentence description, 2–3 representative phrases, and an estimated prevalence percentage across all tickets (sum to ~100%).
Order by estimated prevalence. Avoid duplication and keep it concise.

Batch summaries:

{bulleted_summaries}
"""
    final = client.chat.completions.create(
        model=model_merge,
        messages=[{'role': 'user', 'content': merge_prompt}],
        temperature=temp_merge,
        max_tokens=max_tokens_merge,
    )
    final_text = final.choices[0].message.content
    return batch_summaries, final_text


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Customer support EDA + theme extraction")
    parser.add_argument("--path", type=str, default=None, help="Local folder with CSV files. If omitted, uses KaggleHub.")
    parser.add_argument("--target_type", type=str, default="Technical issue")
    parser.add_argument("--target_channel", type=str, default="Social media")
    parser.add_argument("--N", type=int, default=200, help="Cap of snippets to use from the subset")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--temp_batch", type=float, default=0.2)
    parser.add_argument("--temp_merge", type=float, default=0.0)
    parser.add_argument("--max_tokens_batch", type=int, default=700)
    parser.add_argument("--max_tokens_merge", type=int, default=900)
    parser.add_argument("--fast", action="store_true", help="Faster settings: N=80, batch_size=40, reduced tokens")
    args = parser.parse_args()

    df = load_data(kaggle=(args.path is None), path=args.path)
    print("Loaded data:", df.shape)

    df = basic_clean(df)
    clean_df = filter_negative_ttr(df)

    # Ticket type head
    tbl, head_types = ticket_type_head(clean_df)
    print("\nTicket Type distribution:")
    print(tbl)

    # Channel mix compare
    print("\nChannel mix vs head types:")
    ch_compare = channel_mix_vs_head(clean_df, head_types)
    print(ch_compare)

    # Theme extraction
    N_cap = args.N
    batch_size = args.batch_size
    temp_batch = args.temp_batch
    temp_merge = args.temp_merge
    max_tokens_batch = args.max_tokens_batch
    max_tokens_merge = args.max_tokens_merge

    if args.fast:
        N_cap = min(N_cap, 80)
        batch_size = max(batch_size, 40)
        max_tokens_batch = min(max_tokens_batch, 400)
        max_tokens_merge = min(max_tokens_merge, 600)

    sample = build_snippets(clean_df, args.target_type, args.target_channel, N_cap=N_cap)
    snippets = sample['snippet'].tolist()

    batch_summaries, final_text = summarize_with_llm(
        snippets,
        batch_size=batch_size,
        temp_batch=temp_batch,
        temp_merge=temp_merge,
        max_tokens_batch=max_tokens_batch,
        max_tokens_merge=max_tokens_merge,
    )

    print("\nFinal Themes for", args.target_type, "via", args.target_channel)
    print(final_text)

    # Optional: save artifacts next to script
    try:
        df_snippets = sample[['Ticket Subject', 'Ticket Description', 'Ticket Type', 'Ticket Channel', 'snippet']].copy()
        df_snippets.to_csv("snippets.csv", index=False)
        pd.DataFrame({
            'batch': list(range(1, len(batch_summaries) + 1)),
            'summary': batch_summaries
        }).to_csv("batch_summaries.csv", index=False)
        pd.DataFrame([{
            'target_type': args.target_type,
            'target_channel': args.target_channel,
            'subset_size': len(sample),
            'n_batches': int(np.ceil(len(snippets) / batch_size)),
            'final_themes': final_text,
        }]).to_csv("final_themes.csv", index=False)
        print("\nSaved: snippets.csv, batch_summaries.csv, final_themes.csv")
    except Exception as e:
        print("Save failed:", e)


if __name__ == "__main__":
    main()

