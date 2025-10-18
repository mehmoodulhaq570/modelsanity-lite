import React, { useState } from "react";

type AnalyzeResult = {
  dataset_name: string;
  rows: number;
  cols: number;
  missing_per_column: Record<string, number>;
  dtypes: Record<string, string>;
  target_balance?: Record<string, number> | null;
  sample_leakage_warnings: string[];
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function UploadForm(): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState<string>("");
  const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
  const [trainRes, setTrainRes] = useState<any | null>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  async function postForm(path: string, body: FormData) {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        body,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `${res.status} ${res.statusText}`);
      }
      return await res.json();
    } finally {
      setLoading(false);
    }
  }

  async function analyze(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return alert("Please select a CSV file.");
    const fd = new FormData();
    fd.append("file", file);
    if (target) fd.append("target_column", target);
    const data = await postForm("/analyze", fd);
    setAnalysis(data);
  }

  async function train(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !target) return alert("Select CSV and target column to train.");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("target_column", target);
    const data = await postForm("/train", fd);
    setTrainRes(data);
  }

  async function fetchRuns() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/runs`);
      const data = await res.json();
      setRuns(data.runs || []);
    } catch (err) {
      alert("Failed to load runs: " + String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 760 }}>
      <form style={{ display: "grid", gap: 8 }}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
        />
        <input
          placeholder="target column (optional for analyze)"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={analyze} disabled={loading}>
            Analyze
          </button>
          <button onClick={train} disabled={loading}>
            Train Baseline
          </button>
          <button
            type="button"
            onClick={fetchRuns}
            disabled={loading}
            style={{ marginLeft: "auto" }}
          >
            Load Runs
          </button>
        </div>
      </form>

      {loading && <p>Working…</p>}

      {analysis && (
        <section style={{ marginTop: 16 }}>
          <h3>Analysis</h3>
          <pre style={{ background: "#f6f8fa", padding: 12, overflow: "auto" }}>
            {JSON.stringify(analysis, null, 2)}
          </pre>
        </section>
      )}

      {trainRes && (
        <section style={{ marginTop: 16 }}>
          <h3>Train Result</h3>
          <pre style={{ background: "#f6f8fa", padding: 12 }}>
            {JSON.stringify(trainRes, null, 2)}
          </pre>
        </section>
      )}

      <section style={{ marginTop: 16 }}>
        <h3>Runs</h3>
        <pre style={{ background: "#f6f8fa", padding: 12 }}>
          {JSON.stringify(runs, null, 2)}
        </pre>
      </section>
    </div>
  );
}
