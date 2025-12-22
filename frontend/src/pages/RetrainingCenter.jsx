import { useEffect, useState, useRef } from 'react';
import { Play, CheckCircle2, AlertCircle, Clock, Database, GitBranch, Zap } from 'lucide-react';
import KPICard from '../components/KPICard';
import ChartCard from '../components/ChartCard';
import Modal from '../components/Modal';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const FALLBACK_MODELS = ["TFT", "N-BEATS", "DeepAR", "Prophet", "RandomForest", "XGBoost", "LSTM"];

export default function RetrainingCenter() {
  // UI state
  const [showTrigger, setShowTrigger] = useState(false);
  const [showComparison, setShowComparison] = useState(false);

  // retrain state
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainingProgress, setRetrainingProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('');
  const [pipelineSteps, setPipelineSteps] = useState([]); // [{step, status, duration?}]
  const [estimatedTime, setEstimatedTime] = useState('');

  // available model types
  const [modelTypes, setModelTypes] = useState(FALLBACK_MODELS);
  const [selectedModelType, setSelectedModelType] = useState(FALLBACK_MODELS[0]);
  const [dataRange, setDataRange] = useState('Last 12 months');
  const [autoDeploy, setAutoDeploy] = useState(true);

  // metrics / history / logs
  const [metrics, setMetrics] = useState(null); // {mape, rmse, r2, coverage}
  const [trainingHistory, setTrainingHistory] = useState([]); // array of runs
  const [accuracyTrendData, setAccuracyTrendData] = useState([]); // for chart
  const [logs, setLogs] = useState([]);

  // run id for current retrain
  const runIdRef = useRef(null);
  const pollingRef = useRef(null);
  // -----------------------
  // SSE Stream Reader
  // -----------------------
  const readSSE = async (reader) => {
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value) continue;

      // SSE packets come separated by "\n\n"
      const events = value.split("\n\n");

      for (let e of events) {
        if (!e.startsWith("data:")) continue;

        const jsonStr = e.replace("data: ", "");

        try {
          const event = JSON.parse(jsonStr);

          console.log("SSE EVENT:", event);

          // -----------------------------
          // 1. Add log to UI
          // -----------------------------
          setLogs((prev) => [
            ...prev,
            {
              time: new Date(event.ts).toLocaleTimeString(),
              level: event.status === "failed" ? "ERROR" : "INFO",
              message: `${event.step}: ${event.log}`,
            },
          ]);

          // -----------------------------
          // 2. Global progress bar (top progress)
          // -----------------------------
          if (
            event.status === "progress" &&
            event.payload &&
            event.payload.percent !== undefined
          ) {
            setRetrainingProgress(event.payload.percent);
            setCurrentStep(event.log);
          }

          // -----------------------------
          // 3. Pipeline step progress
          // -----------------------------
          setPipelineSteps((prev) => {
            const exists = prev.find((s) => s.step === event.step);

            const updated = {
              step: event.step,
              status: event.status,
              log: event.log,
              percent: event.payload?.percent ?? exists?.percent ?? 0,
            };

            if (exists) {
              return prev.map((s) =>
                s.step === event.step ? updated : s
              );
            }

            return [...prev, updated];
          });

          // -----------------------------
          // 4. Training finished
          // -----------------------------
          if (event.step === "done") {
            setRetrainingProgress(100);
            setIsRetraining(false);
          }
        } catch (parseErr) {
          console.error("SSE parse error:", parseErr);
        }
      }
    }
  } catch (err) {
    console.error("SSE reader error:", err);
  }
};


  // helper: auth token
  const getToken = () => sessionStorage.getItem('token');

  // -----------------------
  // Initial data load
  // -----------------------
  useEffect(() => {
    loadModelTypes();
    loadMetrics();
    loadHistory();
    loadLogs();

    // cleanup on unmount
    return () => {
      stopPolling();
    };
  }, []);

  // -----------------------
  // Load model types (backend or fallback)
  // -----------------------
  const loadModelTypes = async () => {
    try {
      const res = await fetch('/retrain/model-types'); // expected array
      if (!res.ok) throw new Error('no model types');
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        setModelTypes(data);
        setSelectedModelType(data[0]);
      } else {
        setModelTypes(FALLBACK_MODELS);
      }
    } catch (e) {
      // fallback silently
      setModelTypes(FALLBACK_MODELS);
      setSelectedModelType(FALLBACK_MODELS[0]);
    }
  };

  // -----------------------
  // Load metrics & history & logs
  // -----------------------
  const loadMetrics = async () => {
    try {
      const res = await fetch('/retrain/metrics');
      if (!res.ok) return;
      const data = await res.json();
      setMetrics(data);
    } catch (e) {
      // ignore
    }
  };

  const loadHistory = async () => {
    try {
      const res = await fetch('/retrain/history');
      if (!res.ok) return;
      const data = await res.json();
      setTrainingHistory(data || []);
      // build chart data
      setAccuracyTrendData((data || []).map((r) => ({ version: r.version, mape: r.mape })));
    } catch (e) {
      // ignore
    }
  };

  const loadLogs = async (run_id = null) => {
    try {
      const url = run_id ? `/retrain/logs?run_id=${run_id}` : '/retrain/logs';
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      setLogs(data || []);
    } catch (e) {
      // ignore
    }
  };

  // -----------------------
  // Trigger retrain
  // -----------------------
  const handleTriggerRetrain = async () => {
    setShowTrigger(false);

    const token = getToken();

    const payload = {
      token: token,                   // <-- backend expects this
      model_type: selectedModelType,
      data_range: dataRange,
      auto_deploy: !!autoDeploy,
      // file_path: selectedFilePath || null,
      // file_table_id: selectedFileId || null
    };

    try {
      const res = await fetch("http://localhost:8000/retrain/start", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        console.error("start failed", await res.text());
        return;
      }

      // ❗ IMPORTANT: Response is SSE, NOT JSON
      const reader = res.body
        .pipeThrough(new TextDecoderStream())
        .getReader();

      // Your SSE handler function
      readSSE(reader);

      // Update UI state
      setIsRetraining(true);
      setRetrainingProgress(0);
      setCurrentStep("Starting");
      setPipelineSteps([]);

    } catch (e) {
      console.error(e);
    }
  };


  // -----------------------
  // Polling logic
  // -----------------------
  const startPolling = (run_id) => {
    stopPolling();
    pollingRef.current = setInterval(async () => {
      await pollProgress(run_id);
    }, 2000);
  };

  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const pollProgress = async (run_id) => {
    try {
      const res = await fetch(`/retrain/progress?run_id=${encodeURIComponent(run_id)}`);
      if (!res.ok) {
        // if server returns 404 or finished, stop
        // but we'll still attempt to load final metrics
        return;
      }
      const data = await res.json();
      // expected: { progress, estimated_time, current_step, pipeline_steps }
      setRetrainingProgress(Number(data.progress ?? 0));
      setEstimatedTime(data.estimated_time ?? '');
      setCurrentStep(data.current_step ?? '');
      setPipelineSteps(data.pipeline_steps ?? []);

      // update logs incrementally
      await loadLogs(run_id);

      // If progress >= 100 or pipeline shows finished, finalize
      if ((Number(data.progress ?? 0) >= 100) || (data.status === 'finished')) {
        // stop polling and refresh metrics/history
        stopPolling();
        setIsRetraining(false);
        setRetrainingProgress(100);
        runIdRef.current = null;
        await loadMetrics();
        await loadHistory();
        await loadLogs(); // global logs
      }
    } catch (e) {
      // ignore transient errors
      console.error('poll error', e);
    }
  };

  // -----------------------
  // Manual refresh handlers
  // -----------------------
  const refreshAll = async () => {
    await loadMetrics();
    await loadHistory();
    await loadLogs();
  };

  // -----------------------
  // Modal form inputs handlers
  // -----------------------
  const onModelTypeChange = (e) => setSelectedModelType(e.target.value);
  const onDataRangeChange = (e) => setDataRange(e.target.value);
  const onAutoDeployChange = (e) => setAutoDeploy(e.target.checked);

  // -----------------------
  // Render helpers
  // -----------------------
  const renderPipelineStepsDesktop = () => (
    <div className="hidden lg:block">
      <div className="flex items-center justify-between">
        {pipelineSteps.length > 0 ? pipelineSteps.map((step, index) => (
          <div key={step.step || index} className="flex items-center flex-1">
            <div className="flex flex-col items-center flex-1">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${step.status === 'complete' ? 'bg-green-100' :
                step.status === 'running' ? 'bg-blue-100' : 'bg-gray-100'
                }`}>
                {step.status === 'complete' ? (
                  <CheckCircle2 className="w-6 h-6 text-green-600" />
                ) : step.status === 'running' ? (
                  <Zap className="w-6 h-6 text-blue-600" />
                ) : (
                  <Clock className="w-6 h-6 text-gray-400" />
                )}
              </div>
              <p className="text-sm text-gray-900 mt-2 text-center">{step.step}</p>
              {step.duration && <p className="text-xs text-gray-500 mt-1">{step.duration}</p>}
            </div>
            {index < pipelineSteps.length - 1 && (
              <div className="flex-1 h-0.5 bg-green-200 mx-2" />
            )}
          </div>
        )) : (
          // If no pipelineSteps loaded, show default placeholders (keeps appearance)
          <div className="flex items-center flex-1">
            <div className="flex flex-col items-center flex-1">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center bg-gray-100">
                <Clock className="w-6 h-6 text-gray-400" />
              </div>
              <p className="text-sm text-gray-900 mt-2 text-center">Waiting</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // -----------------------
  // UI return
  // -----------------------
  return (
    <div className="p-4 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-gray-900 text-3xl mb-2">Model Retraining Center</h1>
          <p className="text-gray-600">Automated ML pipeline management and monitoring</p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowTrigger(true)}
            className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors flex items-center gap-2"
          >
            <Play className="w-5 h-5" />
            Trigger Retrain
          </button>

          <button
            onClick={refreshAll}
            className="px-4 py-2 border border-gray-200 rounded-xl hover:bg-gray-50"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Retraining Progress */}
      {isRetraining && (
        <div className="bg-blue-50 border border-blue-200 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="animate-spin">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
            <div>
              <h3 className="text-blue-900">Retraining in Progress</h3>
              <p className="text-sm text-blue-700">{currentStep} • Estimated time: {estimatedTime}</p>
            </div>
          </div>

          <div className="w-full bg-blue-200 rounded-full h-3">
            <div
              className="bg-blue-600 h-3 rounded-full transition-all duration-1000"
              style={{ width: `${retrainingProgress}%` }}
            />
          </div>

          <div className="flex justify-between text-sm text-blue-700 mt-2">
            <span>Progress</span>
            <span>{retrainingProgress}%</span>
          </div>
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">
        <KPICard
          title="Last Training"
          value={trainingHistory[0]?.date || '-'}
          icon={Clock}
        />
        <KPICard
          title="Current Model"
          value={trainingHistory[0]?.version || '-'}
          icon={GitBranch}
        />
        <KPICard
          title="Training Duration"
          value={trainingHistory[0]?.duration ? parseInt(trainingHistory[0].duration) : '-'}
          unit="min"
          icon={Zap}
        />
        <KPICard
          title="Data Volume"
          value={metrics?.data_volume || '-'}
          unit="records"
          icon={Database}
        />
      </div>

      {/* Pipeline Diagram */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-gray-900">ML Pipeline Status</h3>
          <span className="px-3 py-1 bg-green-100 text-green-700 rounded-lg text-sm flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4" />
            All Steps
          </span>
        </div>

        {renderPipelineStepsDesktop()}

        {/* Mobile pipeline */}
        <div className="lg:hidden space-y-3">
          {(pipelineSteps.length > 0 ? pipelineSteps : [
            { step: 'Ingestion', status: 'pending', duration: '' },
            { step: 'Validation', status: 'pending', duration: '' },
            { step: 'Feature Engineering', status: 'pending', duration: '' },
            { step: 'Model Training', status: 'pending', duration: '' },
            { step: 'Evaluation', status: 'pending', duration: '' },
            { step: 'Model Registry', status: 'pending', duration: '' },
            { step: 'Deployment', status: 'pending', duration: '' },
          ]).map((step, idx) => (
            <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl">
              <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${step.status === 'complete' ? 'bg-green-100' : 'bg-gray-100'
                }`}>
                <CheckCircle2 className="w-5 h-5 text-green-600" />
              </div>
              <div className="flex-1">
                <p className="text-gray-900 text-sm">{step.step}</p>
                <p className="text-xs text-gray-500">{step.duration || ''}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard
          title="Model Accuracy Trend"
          action={
            <button
              onClick={() => setShowComparison(true)}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              Compare Versions
            </button>
          }
        >
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={accuracyTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="version" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Line type="monotone" dataKey="mape" stroke="#3B82F6" strokeWidth={2} dot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-gray-900 mb-4">Last Run Metrics</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-4 bg-blue-50 rounded-xl">
              <span className="text-gray-700">MAPE (Accuracy)</span>
              <span className="text-2xl text-blue-900">{metrics?.mape ? `${metrics.mape}%` : '-'}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-green-50 rounded-xl">
              <span className="text-gray-700">RMSE</span>
              <span className="text-2xl text-green-900">{metrics?.rmse ?? '-'}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-purple-50 rounded-xl">
              <span className="text-gray-700">R² Score</span>
              <span className="text-2xl text-purple-900">{metrics?.r2 ?? '-'}</span>
            </div>
            <div className="flex justify-between items-center p-4 bg-amber-50 rounded-xl">
              <span className="text-gray-700">Coverage</span>
              <span className="text-2xl text-amber-900">{metrics?.coverage ? `${metrics.coverage}%` : '-'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Training History */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-gray-900 mb-4">Training History</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 text-gray-600">Date</th>
                <th className="text-left py-3 px-4 text-gray-600">Version</th>
                <th className="text-left py-3 px-4 text-gray-600">MAPE</th>
                <th className="text-left py-3 px-4 text-gray-600">RMSE</th>
                <th className="text-left py-3 px-4 text-gray-600">Duration</th>
                <th className="text-left py-3 px-4 text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {trainingHistory.length > 0 ? trainingHistory.map((run, i) => (
                <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-3 px-4 text-gray-700">{run.date}</td>
                  <td className="py-3 px-4 text-gray-900">{run.version}</td>
                  <td className="py-3 px-4 text-gray-700">{run.mape}%</td>
                  <td className="py-3 px-4 text-gray-700">{run.rmse}</td>
                  <td className="py-3 px-4 text-gray-700">{run.duration}</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-lg text-sm ${run.status === 'Production' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                      }`}>
                      {run.status}
                    </span>
                  </td>
                </tr>
              )) : (
                <tr><td colSpan={6} className="py-6 text-center text-gray-500">No training history available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Logs Viewer */}
      <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
        <h3 className="text-gray-900 mb-4">Recent Logs</h3>
        <div className="bg-gray-900 rounded-xl p-4 font-mono text-sm max-h-96 overflow-y-auto">
          {logs.length > 0 ? logs.map((log, i) => (
            <div key={i} className="flex gap-3 py-1">
              <span className="text-gray-500">{log.time}</span>
              <span className={log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARNING' ? 'text-yellow-400' : 'text-green-400'}>
                {log.level}
              </span>
              <span className="text-gray-300">{log.message}</span>
            </div>
          )) : (
            <div className="text-gray-400">No logs available</div>
          )}
        </div>
      </div>

      {/* Trigger Retrain Modal */}
      <Modal
        isOpen={showTrigger}
        onClose={() => setShowTrigger(false)}
        title="Trigger Model Retraining"
        size="md"
      >
        <div className="space-y-6">
          <div className="p-4 bg-blue-50 rounded-xl">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
              <div>
                <p className="text-blue-900 mb-1">New Training Session</p>
                <p className="text-blue-700 text-sm">
                  Select the model architecture and data range to start a new retraining run. Estimated duration will be shown after start.
                </p>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Data Range</label>
            <select value={dataRange} onChange={onDataRangeChange} className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option>Last 12 months</option>
              <option>Last 6 months</option>
              <option>Last 24 months</option>
              <option>All available data</option>
            </select>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Model Architecture</label>
            <select value={selectedModelType} onChange={onModelTypeChange} className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500">
              {modelTypes.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="flex items-center gap-3">
            <input type="checkbox" id="auto-deploy" className="w-4 h-4 text-blue-600" checked={autoDeploy} onChange={onAutoDeployChange} />
            <label htmlFor="auto-deploy" className="text-gray-700 text-sm">
              Auto-deploy if MAPE improves by &gt; 5%
            </label>
          </div>

          <div className="flex gap-3 justify-end">
            <button onClick={() => setShowTrigger(false)} className="px-6 py-2.5 border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors">
              Cancel
            </button>
            <button onClick={handleTriggerRetrain} className="px-6 py-2.5 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors flex items-center gap-2">
              <Play className="w-4 h-4" />
              Start Retraining
            </button>
          </div>
        </div>
      </Modal>

      {/* Version Comparison Modal */}
      <Modal
        isOpen={showComparison}
        onClose={() => setShowComparison(false)}
        title="Model Version Comparison"
        size="xl"
      >
        <div className="space-y-6">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={trainingHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="version" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip />
              <Legend />
              <Bar dataKey="mape" fill="#3B82F6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-gray-600">Version</th>
                  <th className="text-left py-3 px-4 text-gray-600">MAPE</th>
                  <th className="text-left py-3 px-4 text-gray-600">RMSE</th>
                  <th className="text-left py-3 px-4 text-gray-600">Improvement</th>
                </tr>
              </thead>
              <tbody>
                {trainingHistory.length > 0 ? trainingHistory.map((run, i) => (
                  <tr key={i} className="border-b border-gray-100">
                    <td className="py-3 px-4 text-gray-900">{run.version}</td>
                    <td className="py-3 px-4 text-gray-700">{run.mape}%</td>
                    <td className="py-3 px-4 text-gray-700">{run.rmse}</td>
                    <td className="py-3 px-4">
                      {i < trainingHistory.length - 1 && (
                        <span className="text-green-600">
                          {((trainingHistory[i + 1].mape - run.mape) / trainingHistory[i + 1].mape * 100).toFixed(1)}% better
                        </span>
                      )}
                    </td>
                  </tr>
                )) : (
                  <tr><td colSpan={4} className="py-6 text-center text-gray-500">No versions to compare</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </Modal>
    </div>
  );
}
