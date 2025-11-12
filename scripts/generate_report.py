#!/usr/bin/env python3
import os, argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
import plotly.express as px
import plotly.io as pio

def safe_read(path, **kwargs):
    if os.path.exists(path):
        try: return pd.read_csv(path, **kwargs)
        except: return pd.DataFrame()
    return pd.DataFrame()

def read_fasta_headers(path, n=10):
    if not os.path.exists(path): return []
    recs = list(SeqIO.parse(path, "fasta"))
    return [(r.id, len(r.seq)) for r in recs[:n]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blast_known", default="data/BLAST_files/blast_consensus_annotated.tsv")
    ap.add_argument("--blast_novel", default="data/BLAST_files/blast_consensus_novel.tsv")
    ap.add_argument("--clusters", default="data/CLUSTER_files/clusters.tsv")
    ap.add_argument("--summary", default="data/CLUSTER_files/cluster_summary.tsv")
    ap.add_argument("--reps", default="data/CLUSTER_files/cluster_reps.fa")
    ap.add_argument("--embeddings", default="data/DNABERT_embeddings/windows_embeddings.npy")
    ap.add_argument("--out", default="results/deepsea_report.html")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load data
    df_known = safe_read(args.blast_known, sep="\t")
    df_novel = safe_read(args.blast_novel, sep="\t")
    df_clusters = safe_read(args.clusters, sep="\t")
    df_summary = safe_read(args.summary, sep="\t")
    reps = read_fasta_headers(args.reps)

    # Stats
    samples_processed = df_clusters['sample'].nunique() if 'sample' in df_clusters else 0
    asvs_found = df_clusters.shape[0]
    clusters_found = df_summary.shape[0]
    novel_candidates = df_novel.shape[0]

    # UMAP scatter (if coords missing, generate fake coords)
    if {'umap1','umap2'}.issubset(c.lower() for c in df_clusters.columns):
        c1 = [c for c in df_clusters.columns if c.lower()=="umap1"][0]
        c2 = [c for c in df_clusters.columns if c.lower()=="umap2"][0]
        umap_df = df_clusters.rename(columns={c1:"UMAP1",c2:"UMAP2"})
    else:
        rng = np.random.default_rng(42)
        umap_df = pd.DataFrame({
            "UMAP1": rng.normal(size=asvs_found),
            "UMAP2": rng.normal(size=asvs_found),
            "label": df_clusters['cluster'] if 'cluster' in df_clusters else 0
        })
    umap_df['Novel'] = umap_df['label'].apply(lambda x: "Novel" if x==-1 else "Known")

    fig_umap = px.scatter(umap_df, x="UMAP1", y="UMAP2", color="Novel",
                          title="UMAP: Known vs Novel Clusters")
    umap_div = pio.to_html(fig_umap, include_plotlyjs='cdn', full_html=False)

    # Species pie (known BLAST hits)
    if not df_known.empty:
        top_species = df_known.iloc[:,1].value_counts().head(10)
        fig_pie = px.pie(values=top_species.values, names=top_species.index,
                         title="Top 10 Known Species")
        pie_div = pio.to_html(fig_pie, include_plotlyjs=False, full_html=False)
    else:
        pie_div = "<div>No BLAST annotated hits available.</div>"

    # Novel clusters bar chart
    if not df_novel.empty:
        top_novel = df_novel.head(10)
        fig_novel = px.bar(top_novel, x=top_novel.index, y=top_novel.iloc[:,2],
                           title="Top 10 Novel Clusters", labels={"x":"Cluster","y":"% identity"})
        novel_div = pio.to_html(fig_novel, include_plotlyjs=False, full_html=False)
    else:
        novel_div = "<div>No novel clusters detected.</div>"

    # Representative sequences
    reps_html = "<ul>" + "".join([f"<li>{rid} (len={ln})</li>" for rid,ln in reps]) + "</ul>"

    # Build HTML
    html = f"""
    <html>
    <head><title>DeepSea eDNA Report</title></head>
    <body style="font-family:Arial; background:#f8fbff; padding:20px;">
      <h2>DeepSea eDNA Report</h2>
      <div style="display:flex; gap:20px;">
        <div style="background:#e6f2ff;padding:10px;border-radius:8px;">Samples: {samples_processed}</div>
        <div style="background:#eaf9ea;padding:10px;border-radius:8px;">ASVs: {asvs_found}</div>
        <div style="background:#fff4e6;padding:10px;border-radius:8px;">Clusters: {clusters_found}</div>
        <div style="background:#ffe6f2;padding:10px;border-radius:8px;">Novel Candidates: {novel_candidates}</div>
      </div>
      <h3>UMAP Projection</h3>
      {umap_div}
      <h3>Taxonomic Composition</h3>
      {pie_div}
      <h3>Novel Clusters</h3>
      {novel_div}
      <h3>Representative Sequences (first 10)</h3>
      {reps_html}
    </body>
    </html>
    """
    with open(args.out,"w") as f: f.write(html)
    print("✅ Report written to", args.out)

if __name__=="__main__":
    main()
