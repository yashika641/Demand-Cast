import { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertTriangle, Download, X } from 'lucide-react';

export default function DataUpload() {
  const [uploadStep, setUploadStep] = useState(1);

  // multiple files
  const [files, setFiles] = useState([]);
  const [fileNames, setFileNames] = useState([]);

  const [isDragging, setIsDragging] = useState(false);

  const [detectedSchemas, setDetectedSchemas] = useState([]); // array of files: [{file_name, columns, preview}]
  const [validationResults, setValidationResults] = useState([]);

  // -----------------------
  // STEP 1 — MULTIPLE FILE SELECTION
  // -----------------------
  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files);
    if (selected.length > 0) {
      setFiles(selected);
      setFileNames(selected.map(f => f.name));
      setUploadStep(2);
      handleSchemaDetection(selected);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const dropped = Array.from(e.dataTransfer.files);
    if (dropped.length > 0) {
      setFiles(dropped);
      setFileNames(dropped.map(f => f.name));
      setUploadStep(2);
      handleSchemaDetection(dropped);
    }
  };

  // ------------------------------------------------------------
  // STEP 2 — Send ALL files to backend for schema detection
  // ------------------------------------------------------------
  const handleSchemaDetection = async (fileList) => {
    const formData = new FormData();

    fileList.forEach((f) => formData.append("files", f));

    const res = await fetch("http://localhost:8000/upload/schema-detect", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    setDetectedSchemas(data); // array: [{file_name, columns, preview}]
  };

  // ---------------------
  // STEP 3 — Validation
  // ---------------------
  const handleValidate = async () => {
    const payload = {
      files: detectedSchemas.map(f => ({
        file_name: f.file_name,
        preview: f.preview
      }))
    };

    const res = await fetch("http://localhost:8000/upload/validate-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    setValidationResults(data.results);
    setUploadStep(3);
  };

  // --------------------------------------
  // STEP 4 — MULTIPLE FINAL UPLOAD
  // --------------------------------------
  const handleConfirmUpload = async () => {
    setUploadStep(4);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    const token = sessionStorage.getItem("token");

    await fetch("http://localhost:8000/upload/upload", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData
    });

    setTimeout(() => {
      setUploadStep(1);
      setFiles([]);
      setFileNames([]);
      setDetectedSchemas([]);
      setValidationResults([]);
    }, 3000);
  };

  return (
    <div className="p-4 lg:p-8 space-y-6">
      
      {/* HEADER */}
      <div>
        <h1 className="text-gray-900 text-3xl mb-2">Data Upload & Schema Mapping</h1>
        <p className="text-gray-600">Import historical demand data and configure field mappings</p>
      </div>

      {/* STEPS */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between">
          {[
            { step: 1, label: 'Upload Files' },
            { step: 2, label: 'Map Schema' },
            { step: 3, label: 'Validate Data' },
            { step: 4, label: 'Confirm' },
          ].map((item, i) => (
            <div key={item.step} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <div
                  className={`w-10 h-10 rounded-xl mb-2 flex items-center justify-center ${
                    uploadStep > item.step
                      ? 'bg-green-100 text-green-600'
                      : uploadStep === item.step
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-400'
                  }`}
                >
                  {uploadStep > item.step ? <CheckCircle2 className="w-5 h-5" /> : item.step}
                </div>
                <p className={`text-sm ${uploadStep >= item.step ? 'text-gray-900' : 'text-gray-500'}`}>
                  {item.label}
                </p>
              </div>
              {i < 3 && <div className={`h-0.5 flex-1 -mt-6 ${uploadStep > item.step ? 'bg-green-500' : 'bg-gray-200'}`} />}
            </div>
          ))}
        </div>
      </div>

      {/* STEP 1 — MULTIPLE FILE UPLOAD */}
      {uploadStep === 1 && (
        <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-2xl p-12 text-center transition-colors ${
              isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
            }`}
          >
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>

              <h3 className="text-gray-900 mb-2">Drop your files here</h3>
              <p className="text-gray-600 mb-6">or click to browse</p>

              <label className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 cursor-pointer">
                Select Files
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>

              {fileNames.length > 0 && (
                <p className="mt-4 text-gray-700 text-sm">
                  Selected: {fileNames.join(", ")}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* STEP 2 — MULTI FILE SCHEMA MAPPING */}
      {uploadStep === 2 && (
        <div className="space-y-8">

          {detectedSchemas.map((fileSchema, idx) => (
            <div key={idx} className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-gray-900">
                  Schema: {fileSchema.file_name}
                </h3>
              </div>

              <table className="w-full">
                <thead>
                  <tr className="border-b">
                    <th className="py-2 px-4">Column</th>
                    <th className="py-2 px-4">Type</th>
                  </tr>
                </thead>
                <tbody>
                  {fileSchema.columns.map((col, i) => (
                    <tr key={i} className="border-b">
                      <td className="py-2 px-4">{col.name}</td>
                      <td className="py-2 px-4">{col.type}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h4 className="text-gray-900 mt-6 mb-2">Preview</h4>

              <table className="w-full text-sm">
                <thead>
                  <tr>
                    {fileSchema.preview.length > 0 &&
                      Object.keys(fileSchema.preview[0]).map((key) => (
                        <th key={key} className="py-2 px-4">{key}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {fileSchema.preview.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="py-2 px-4">{val}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          <div className="flex justify-end gap-3">
            <button onClick={() => setUploadStep(1)} className="px-6 py-3 border rounded-xl">Back</button>
            <button onClick={handleValidate} className="px-6 py-3 bg-blue-500 text-white rounded-xl">Validate Data</button>
          </div>
        </div>
      )}

      {/* STEP 3 — MULTI FILE VALIDATION */}
      {uploadStep === 3 && (
        <div className="space-y-6">
          {validationResults.map((res, i) => (
            <div key={i} className="bg-white rounded-2xl p-6 shadow-sm border">
              <h3 className="text-gray-900 mb-3">Validation: {res.file_name}</h3>

              {res.errors.length > 0 ? (
                res.errors.map((err, j) => (
                  <div key={j} className="p-4 bg-amber-50 border-amber-200 border rounded-xl mb-2">
                    <p className="text-amber-900">Row {err.row} — {err.column}: {err.issue}</p>
                  </div>
                ))
              ) : (
                <p className="text-green-600">No issues found.</p>
              )}
            </div>
          ))}

          <div className="flex justify-end gap-3">
            <button onClick={() => setUploadStep(2)} className="px-6 py-3 border rounded-xl">Back</button>
            <button onClick={handleConfirmUpload} className="px-6 py-3 bg-blue-500 text-white rounded-xl">Confirm Upload</button>
          </div>
        </div>
      )}

      {/* STEP 4 — SUCCESS */}
      {uploadStep === 4 && (
        <div className="bg-white p-12 rounded-2xl text-center shadow-sm">
          <div className="w-20 h-20 bg-green-100 rounded-full mx-auto mb-6 flex items-center justify-center">
            <CheckCircle2 className="w-10 h-10 text-green-600" />
          </div>
          <h3 className="text-gray-900 text-2xl mb-3">Upload Successful!</h3>
          <p className="text-gray-600 mb-8">
            Your files have been imported and processing is in progress.
          </p>
        </div>
      )}
    </div>
  );
}
