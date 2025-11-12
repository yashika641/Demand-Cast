import { auth } from "../firebase";
import React, { useRef, useState } from "react";
import bg_image from '../assets/bg1.png';
import logo from '../assets/logo1.png';
import { useNavigate } from "react-router-dom";



const fileTypeColors = {
  csv: "bg-green-400",
  cxml: "bg-indigo-400",
  xlsx: "bg-yellow-400",
  xlxx: "bg-yellow-400",
};

function getFileType(name) {
  const ext = name.split(".").pop().toLowerCase();
  if (fileTypeColors[ext]) return ext;
  return "csv";
}


export default function Uploader() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadingIdx, setUploadingIdx] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFilesChange = (e) => {
    setSelectedFiles([...e.target.files]);
  };

  const handleRemoveFile = (idx) => {
    setSelectedFiles(selectedFiles.filter((_, i) => i !== idx));
  };

  const handleClearAll = () => {
    setSelectedFiles([]);
    setUploadingIdx(null);
  };

  const handleUpload = async () => {
    if (selectedFiles.length > 0) {
      setUploadingIdx(0);
      try {
        for (let i = 0; i < selectedFiles.length; i++) {
          const file = selectedFiles[i];
          const formData = new FormData();
          formData.append('file', file);

          const user = auth.currentUser;
          if (!user) {
            alert('You must be logged in!');
            setUploadingIdx(null);
            return;
          }
          const idToken = await user.getIdToken();

          const response = await fetch("http://localhost:8000/upload", {
            method: "POST",
            body: formData,
            headers: {
              "Authorization": `Bearer ${idToken}`,
            }
          });

          if (!response.ok) {
            const error = await response.json();
            alert(`Upload failed for file "${file.name}": ${error.detail || response.statusText}`);
            setUploadingIdx(null);
            return;
          }

          setUploadingIdx(i + 1);
        }
        setUploadingIdx(null);
        navigate("/services");
      } catch (err) {
        setUploadingIdx(null);
        alert("Upload failed");
      }
    }
  };


  const handleDeleteFiles = async () => {
    const user = auth.currentUser;
    if (!user) {
      alert("You must be logged in!");
      return;
    }

    const confirmDelete = window.confirm("Are you sure you want to delete all your files?");
    if (!confirmDelete) return;

    try {
      const idToken = await user.getIdToken();

      const response = await fetch("http://localhost:8000/delete-files", {
        method: "DELETE",
        headers: {
          "Authorization": `Bearer ${idToken}`,
        },
      });

      const data = await response.json();

      if (response.ok) {
        alert(data.message);
      } else {
        alert(`Delete failed: ${data.detail || "Unknown error"}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className=" w-360 min-h-screen overflow-x-hidden relative font-sans -ml-20 ">
      <img src={bg_image} alt="Background" className="fixed inset-0 object-cover w-full -z-10" />

      <header className="flex items-center border-b border-gray-200 backdrop-blur-2xl  justify-center w-full">
        <img src={logo} alt="Logo" className="  w-15 h-15" />
        <div className="font-bold text-4xl text-[#4976a4]">Demand Cast</div>
        <nav className="ml-auto flex gap-8 text-base mr-5">
          <span className="font-bold border-b-2 border-[#586A84] text-2xl text-blue-300 pb-1 cursor-pointer">Dashboard</span>
          <span className="text-blue-300 cursor-pointer text-2xl">History</span>
          <span className="text-blue-700 cursor-pointer text-2xl">Settings</span>
          <span className="text-blue-700 cursor-pointer text-2xl">Help</span>
        </nav>
      </header>

      <main className="w-full h-150 mx-auto mt-12 backdrop-blur-2xl rounded-xl shadow-lg ">
        <h2 className="text-2xl font-semibold mb-1">Upload Your Data Files</h2>
        <p className="mb-6 text-gray-600 leading-tight">
          Select one or more CSV, CXML, or XLISX files to import.<br />
          Files will processed sequentially.
        </p>

        <div className="border-2 border-dashed border-[#b6bde3] backdrop-blur-2xl rounded-xl px-8 py-6 text-center mb-6">
          <div
            className="flex flex-col items-center"
            onClick={() => fileInputRef.current.click()}
            style={{ cursor: "pointer" }}>
            <span className="text-4xl mb-1">☁️</span>
            <span className="font-semibold text-lg text-amber-50">
              Drag & Drop Files Here
            </span>
            <span className="text-amber-50 text-sm mt-1">Click to Browse Files</span>
            <span className="text-xs mt-2 text-amber-50">.csv, cxml, xlxx</span>
          </div>
          <input
            type="file"
            multiple
            className="hidden"
            ref={fileInputRef}
            onChange={handleFilesChange}
          />
          <button
            onClick={() => fileInputRef.current.click()}
            className="mt-6 bg-[#3d5fc3] text-white rounded-md px-5 py-2 font-medium mb-20"
          >
            Browse Files
          </button>
        </div>

        {selectedFiles.length > 0 && (
          <div className="mt-2 mb-4">
            <h4 className="font-semibold mb-1 text-base">Selected Files</h4>
            {selectedFiles.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center bg-[#4f7ce4] rounded-md px-3 py-2 mb-2 text-sm gap-2"
              >
                <div className={`w-5 h-5 ${fileTypeColors[getFileType(file.name)]} rounded mr-2`} />
                <span className="truncate max-w-[140px]">{file.name}</span>
                <span className="ml-2 text-xs text-[#c1c4cd] font-mono">
                  {(file.size / 1024 / 1024).toFixed(1)} MB
                </span>
                <span className="ml-2 text-xs text-[#c1c4cd] font-mono">
                  {Math.round(file.size / 1024)} KB
                </span>
                <button
                  className="ml-auto text-[#667bad] hover:text-red-500 text-base"
                  onClick={() => handleRemoveFile(idx)}
                >
                  ✖
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex gap-4 mt-4 justify-center">
          <button
            className="bg-[#3d5fc3] text-white px-5 py-2 rounded font-medium mb-20"
            onClick={handleUpload}
          >
            Upload Selected Files
          </button>
          <button
            className="bg-[#e2e8f0] text-[#3d5fc3] px-5 py-2 rounded font-medium mb-20"
            onClick={handleClearAll}
          >
            Clear All
          </button>
        </div>

        {uploadingIdx !== null && (
          <div className="mt-6 bg-[#eef1fa] rounded-lg px-3 py-2 text-center text-base">
            <span>
              Uploading {uploadingIdx + 1} of {selectedFiles.length} files...
            </span>
            <div className="w-full bg-[#cfd6ec] rounded h-2 mt-2 overflow-hidden">
              <div
                className="bg-[#3d5fc3] h-full transition-all duration-300"
                style={{
                  width: `${((uploadingIdx + 1) / selectedFiles.length) * 100}%`,
                }}
              />
            </div>
          </div>
        )}
        <div className="h-10">
          <p className="text-gray-500 text-sm mt-4">
            Note: Large files may take several minutes to upload and process.
          </p>
          <p className="text-gray-500 text-sm">
            if you already have uploaded files you can proceed to next step by clicking "next" button.
          </p>
          <button
            onClick={() => navigate("/services")}
            className="mt-2 bg-green-500 text-white px-5 py-2 rounded font-medium"
          >
            Next
          </button>
        </div>

        <button
          onClick={handleDeleteFiles}
          className="bg-red-500 text-white px-5 py-2 rounded font-medium"
        >
          Delete My Files
        </button>
      </main>
    </div>
  );
}
