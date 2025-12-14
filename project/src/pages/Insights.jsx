import { useState } from 'react';
import { Brain, Send, FileText, TrendingUp, AlertCircle, Sparkles } from 'lucide-react';
import Drawer from '../components/Drawer';

const previousInsights = [
  {
    id: 1,
    title: 'Stockout Risk Analysis',
    date: '2024-05-15',
    preview: 'Analysis of high-risk SKUs and mitigation strategies...',
  },
  {
    id: 2,
    title: 'Promotional ROI Review',
    date: '2024-05-12',
    preview: 'Black Friday campaign performance exceeded expectations...',
  },
  {
    id: 3,
    title: 'Supplier Reliability Report',
    date: '2024-05-10',
    preview: 'Comparative analysis of supplier performance metrics...',
  },
  {
    id: 4,
    title: 'Demand Forecast Accuracy',
    date: '2024-05-08',
    preview: 'Q1 forecast accuracy improved by 12% compared to Q4...',
  },
];

const conversationHistory = [
  {
    role: 'assistant',
    content: "Hello! I'm your DemandCast AI assistant. I can help you analyze demand patterns, forecast accuracy, inventory optimization, and more. What would you like to explore today?",
    timestamp: '10:30 AM',
  },
];

export default function Insights() {
  const [messages, setMessages] = useState(conversationHistory);
  const [input, setInput] = useState('');
  const [showEvidence, setShowEvidence] = useState(false);
  const [selectedInsight, setSelectedInsight] = useState(null);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;

    const newMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages([...messages, newMessage]);
    setInput('');
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = {
        role: 'assistant',
        content: generateResponse(input),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        confidence: 0.94,
        evidence: [
          'Historical demand data (Jan-May 2024)',
          'Forecast model accuracy metrics',
          'Supplier performance records',
        ],
      };
      setMessages((prev) => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const generateResponse = (query) => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('stockout') || lowerQuery.includes('risk')) {
      return 'Based on current inventory levels and demand forecasts, I\'ve identified 5 SKUs at high risk of stockout within the next 7 days. SKU-001 has a 92% stockout probability with only 3 days of inventory remaining. I recommend expediting reorders for these items and considering safety stock increases. Would you like me to generate detailed mitigation recommendations?';
    } else if (lowerQuery.includes('forecast') || lowerQuery.includes('accuracy')) {
      return 'Your forecast accuracy has improved significantly. Current MAPE is 4.2%, which is 0.3% better than last month. The TFT model is performing particularly well for 7-day horizons (3.8% MAPE). Short-term demand sensing is capturing real-time signals effectively, reducing forecast error by 15% during promotional periods.';
    } else if (lowerQuery.includes('promo') || lowerQuery.includes('promotion')) {
      return 'Promotional campaign analysis shows strong performance. Black Friday achieved 140% uplift with a 3.2x ROI. Based on historical patterns and price elasticity analysis, I recommend a 15-20% discount for the upcoming Summer Sale to maximize incremental revenue while maintaining healthy margins.';
    } else if (lowerQuery.includes('supplier') || lowerQuery.includes('lead time')) {
      return 'Supplier A continues to be your most reliable partner with 96% on-time delivery and 7-day average lead time. However, Supplier C has shown increased variance (±3.4 days) over the past month. I recommend diversifying your supply chain and increasing safety stock for SKUs sourced exclusively from Supplier C.';
    } else {
      return 'I can help you with demand forecasting, inventory optimization, risk analysis, promotional planning, and supplier performance insights. Based on your current data, here are three key insights: 1) Your forecast accuracy is strong at 4.2% MAPE, 2) Five SKUs require immediate attention due to stockout risk, 3) Supplier reliability remains high across your network. What would you like to explore in more detail?';
    }
  };

  const quickPrompts = [
    'Analyze high-risk SKUs',
    'Review forecast accuracy',
    'Optimize inventory levels',
    'Supplier performance summary',
  ];

  const loadPreviousInsight = (insight) => {
    setSelectedInsight(insight);
    const insightMessage = {
      role: 'user',
      content: `Show me details for: ${insight.title}`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    
    const responseMessage = {
      role: 'assistant',
      content: `Loading insights from ${insight.date}...\n\n${insight.preview}\n\nThis analysis included comprehensive review of demand patterns, forecast accuracy metrics, and actionable recommendations. The key findings were validated against multiple data sources and showed high confidence levels.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      confidence: 0.92,
    };

    setMessages([...conversationHistory, insightMessage, responseMessage]);
  };

  return (
    <div className="flex h-[calc(100vh-4rem)] lg:h-screen">
      {/* Previous Insights Sidebar - Desktop */}
      <aside className="hidden lg:block w-80 bg-white border-r border-gray-200 overflow-y-auto">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-gray-900">Previous Insights</h2>
          <p className="text-sm text-gray-600 mt-1">Your analysis history</p>
        </div>
        <div className="p-4 space-y-2">
          {previousInsights.map((insight) => (
            <button
              key={insight.id}
              onClick={() => loadPreviousInsight(insight)}
              className={`w-full text-left p-4 rounded-xl hover:bg-gray-50 transition-colors ${
                selectedInsight?.id === insight.id ? 'bg-blue-50 border border-blue-200' : ''
              }`}
            >
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-gray-900 text-sm truncate">{insight.title}</p>
                  <p className="text-xs text-gray-500 mt-1">{insight.date}</p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4 lg:p-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-gray-900">AI Insights Engine</h1>
              <p className="text-sm text-gray-600">Powered by RAG + LLM</p>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-6 space-y-6">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-2xl ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white rounded-2xl rounded-tr-sm'
                    : 'bg-white rounded-2xl rounded-tl-sm shadow-sm border border-gray-100'
                } p-4`}
              >
                {message.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-4 h-4 text-blue-600" />
                    <span className="text-xs text-gray-600">AI Assistant</span>
                  </div>
                )}
                <p className={`${message.role === 'user' ? 'text-white' : 'text-gray-900'} whitespace-pre-line`}>
                  {message.content}
                </p>
                <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-100">
                  <span className={`text-xs ${message.role === 'user' ? 'text-blue-100' : 'text-gray-500'}`}>
                    {message.timestamp}
                  </span>
                  {message.role === 'assistant' && message.confidence && (
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3 text-green-600" />
                        <span className="text-xs text-gray-600">
                          {(message.confidence * 100).toFixed(0)}% confidence
                        </span>
                      </div>
                      {message.evidence && (
                        <button
                          onClick={() => setShowEvidence(message.evidence)}
                          className="text-xs text-blue-600 hover:text-blue-700 underline"
                        >
                          View Evidence
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl rounded-tl-sm shadow-sm border border-gray-100 p-4">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          {messages.length === 1 && (
            <div className="max-w-2xl mx-auto">
              <div className="text-center mb-6">
                <h3 className="text-gray-900 mb-2">Quick Actions</h3>
                <p className="text-sm text-gray-600">Get started with these common queries</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {quickPrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setInput(prompt);
                      setTimeout(() => handleSend(), 100);
                    }}
                    className="p-4 bg-white rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-sm transition-all text-left"
                  >
                    <p className="text-gray-900 text-sm">{prompt}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4 lg:p-6">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about forecasts, risks, inventory, or promotions..."
                className="flex-1 px-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-2 text-center">
              AI-powered insights with evidence-based recommendations
            </p>
          </div>
        </div>
      </div>

      {/* Evidence Drawer */}
      <Drawer
        isOpen={!!showEvidence}
        onClose={() => setShowEvidence(false)}
        title="Evidence & Citations"
        position="right"
      >
        <div className="space-y-4">
          <div className="p-4 bg-blue-50 rounded-xl">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-blue-900 mb-1">Data Sources</p>
                <p className="text-blue-700 text-sm">
                  This insight is based on validated data from multiple sources
                </p>
              </div>
            </div>
          </div>

          {showEvidence && showEvidence.map((evidence, i) => (
            <div key={i} className="p-4 bg-white border border-gray-200 rounded-xl">
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-gray-900">{evidence}</p>
                  <button className="text-sm text-blue-600 hover:text-blue-700 mt-2">
                    View source data →
                  </button>
                </div>
              </div>
            </div>
          ))}

          <div className="p-4 bg-gray-50 rounded-xl">
            <h4 className="text-gray-900 mb-3">SHAP Explanation</h4>
            <div className="space-y-2">
              {[
                { feature: 'Historical Demand Pattern', impact: 0.42 },
                { feature: 'Current Inventory Level', impact: 0.28 },
                { feature: 'Lead Time Variance', impact: 0.18 },
                { feature: 'Seasonal Factors', impact: 0.12 },
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-700">{item.feature}</span>
                    <span className="text-gray-900">{(item.impact * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${item.impact * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Drawer>
    </div>
  );
}
