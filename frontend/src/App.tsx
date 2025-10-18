import React from "react";
import UploadForm from "./components/UploadForm";

export default function App() {
  return (
    <div style={{ padding: 24, fontFamily: "Inter, system-ui, sans-serif" }}>
      <h1>ModelSanity Lite</h1>
      <p>Upload a CSV to analyze or train a baseline model.</p>
      <UploadForm />
    </div>
  );
}
