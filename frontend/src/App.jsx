import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Mail,
  Trash2,
  Clock,
  ArrowUpCircle,
  Play,
  Pause,
  RotateCcw,
  Zap,
  Plus,
  ShieldAlert,
  Briefcase,
  LifeBuoy,
  ChevronRight,
  TrendingUp,
  Inbox,
  Settings,
  X,
  CheckCircle2,
  Brain,
  BarChart3,
  Target,
  Eye,
  Timer,
  AlertTriangle,
} from 'lucide-react';

// --- Config ---
const API_URL = import.meta.env.DEV ? 'http://localhost:8000' : '';

const URGENCY_LEVELS = {
  HIGH:   { label: 'High',   color: 'bg-red-500',     text: 'text-red-700',     bg: 'bg-red-50' },
  MEDIUM: { label: 'Medium', color: 'bg-amber-500',   text: 'text-amber-700',   bg: 'bg-amber-50' },
  LOW:    { label: 'Low',    color: 'bg-emerald-500', text: 'text-emerald-700', bg: 'bg-emerald-50' },
};

const EMAIL_TYPES = {
  SPAM:    { icon: ShieldAlert, label: 'Spam',    color: 'text-red-500' },
  WORK:    { icon: Briefcase,   label: 'Work',    color: 'text-blue-500' },
  SUPPORT: { icon: LifeBuoy,    label: 'Support', color: 'text-purple-500' },
};

const SENTIMENT_STYLES = {
  Aggressive:   { color: 'text-red-600',     bg: 'bg-red-100',     icon: '🔥' },
  Professional: { color: 'text-blue-600',    bg: 'bg-blue-100',    icon: '💼' },
  Casual:       { color: 'text-emerald-600', bg: 'bg-emerald-100', icon: '☕' },
};

// Safe lookups to prevent crashes on unknown keys
const getUrgency = (key) => URGENCY_LEVELS[key?.toUpperCase()] || URGENCY_LEVELS.MEDIUM;
const getEmailType = (key) => EMAIL_TYPES[key?.toUpperCase()] || EMAIL_TYPES.WORK;
const getSentiment = (key) => SENTIMENT_STYLES[key] || SENTIMENT_STYLES.Professional;

export default function App() {
  const [emails, setEmails] = useState([]);
  const [accounts, setAccounts] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [score, setScore] = useState(0);
  const [lastReward, setLastReward] = useState(null);
  const [isAuto, setIsAuto] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [simSpeed, setSimSpeed] = useState(5000);
  const [logs, setLogs] = useState([]);
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [classifierStats, setClassifierStats] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [errMessage, setErrMessage] = useState(null);

  // Form state
  const [newEmail, setNewEmail] = useState('');
  const [newPass, setNewPass] = useState('');

  const timerRef = useRef(null);
  const autoAgentRef = useRef(null);
  const emailsRef = useRef(emails);

  // Keep ref in sync for auto-agent closure
  useEffect(() => {
    emailsRef.current = emails;
  }, [emails]);

  // --- API Actions ---
  const fetchAccounts = async () => {
    try {
      const res = await fetch(`${API_URL}/accounts`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setAccounts(data.accounts || []);
    } catch (e) {
      console.error('[API] fetchAccounts:', e);
    }
  };

  const fetchClassifierStats = async () => {
    try {
      const res = await fetch(`${API_URL}/classifier-stats`);
      if (!res.ok) return;
      const data = await res.json();
      setClassifierStats(data);
    } catch (e) {
      console.error('[API] fetchClassifierStats:', e);
    }
  };

  const addAccount = async (e) => {
    e.preventDefault();
    setIsConnecting(true);
    setErrMessage(null);
    try {
      const res = await fetch(`${API_URL}/accounts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: newEmail, password: newPass }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || `HTTP ${res.status}`);
      }
      setNewEmail('');
      setNewPass('');
      setShowAddAccount(false);
      fetchAccounts();
    } catch (e) {
      console.error('[API] addAccount:', e);
      setErrMessage(e.message);
    } finally {
      setIsConnecting(false);
    }
  };

  const fetchLiveEmails = useCallback(async () => {
    if (isSyncing) return;
    setIsSyncing(true);
    try {
      const res = await fetch(`${API_URL}/live-inbox`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const serverEmails = data.emails || [];

      setEmails((prev) => {
        const currentIds = new Set(prev.map((e) => e.id));
        const newEmails = serverEmails
          .filter((e) => !currentIds.has(e.id))
          .map((e) => ({
            ...e,
            type: (e.type || 'WORK').toUpperCase(),
            urgency: (e.urgency || 'MEDIUM').toUpperCase(),
            sentiment: e.sentiment || 'Professional',
            discoveredAt: Date.now(),
          }));

        const merged = [...newEmails, ...prev];
        return merged
          .map((e) => ({
            ...e,
            waitingTime: Math.floor((Date.now() - (e.discoveredAt || e.createdAt)) / 1000),
          }))
          .slice(0, 30);
      });
    } catch (e) {
      console.error('[API] fetchLiveEmails:', e);
    } finally {
      setIsSyncing(false);
    }
  }, [isSyncing]);

  // --- RL Reward Logic ---
  const handleAction = useCallback(
    (id, actionType) => {
      const email = emailsRef.current.find((e) => e.id === id);
      if (!email) return;

      let reward = 0;
      let logMsg = '';

      const type = (email.type || 'WORK').toUpperCase();
      const urgency = (email.urgency || 'MEDIUM').toUpperCase();
      const sentiment = email.sentiment || 'Professional';
      const confidence = email.confidence || 0.5;
      const confidenceMultiplier = 0.5 + confidence; 

      if (type === 'SPAM') {
        if (actionType === 'DELETE') {
          reward += 2.0 * confidenceMultiplier;
          logMsg = 'Blocked Spam';
        } else {
          reward -= 2.5 * confidenceMultiplier;
          logMsg = 'Allowed Spam through';
        }
      } else {
        if (actionType === 'ESCALATE') {
          if (urgency === 'HIGH' || sentiment === 'Aggressive') {
            reward += 1.5;
            logMsg = `Smart escalation (${sentiment})`;
          } else {
            reward -= 1.0;
            logMsg = 'Unnecessary escalation';
          }
        } else if (actionType === 'OPEN') {
          reward += 1.0 * confidenceMultiplier;
          logMsg = `Processed ${type}`;
        } else if (actionType === 'DELETE') {
          reward -= 3.0;
          logMsg = 'Deleted real email!';
        } else if (actionType === 'DEFER') {
          if (urgency === 'HIGH') {
            reward -= 1.0;
            logMsg = 'Deferred urgent email';
          } else {
            reward -= 0.2;
            logMsg = 'Delayed response';
          }
        }
      }

      if (sentiment === 'Aggressive' && actionType !== 'ESCALATE' && actionType !== 'OPEN') {
        reward -= 1.0;
      }

      const waitPenalty = Math.min(0.1 * (email.waitingTime || 0), 2.0);
      reward -= waitPenalty;

      reward = Math.round(reward * 100) / 100;

      setScore((prev) => prev + reward);
      setLastReward({ value: reward, timestamp: Date.now() });
      setEmails((prev) => prev.filter((e) => e.id !== id));
      if (selectedId === id) setSelectedId(null);

      setLogs((prev) =>
        [
          {
            msg: logMsg,
            reward,
            action: actionType,
            type,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          },
          ...prev,
        ].slice(0, 12)
      );

      if (type === 'SPAM' && actionType === 'DELETE') {
        fetch(`${API_URL}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ predicted_spam: true, actual_spam: true }),
        }).catch(() => {});
      }
    },
    [selectedId]
  );

  // --- Simulation loops ---
  useEffect(() => {
    fetchAccounts();
    fetchClassifierStats();
  }, []);

  useEffect(() => {
    if (isRunning) {
      fetchLiveEmails();
      timerRef.current = setInterval(() => {
        fetchLiveEmails();
        setEmails((prev) =>
          prev.map((e) => ({
            ...e,
            waitingTime: Math.floor((Date.now() - (e.discoveredAt || e.createdAt)) / 1000),
          }))
        );
      }, simSpeed);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isRunning, simSpeed, fetchLiveEmails]);

  useEffect(() => {
    if (isAuto && isRunning) {
      autoAgentRef.current = setInterval(() => {
        const currentEmails = emailsRef.current;
        if (currentEmails.length === 0) return;

        const sorted = [...currentEmails].sort((a, b) => {
          const aSpam = (a.type || '').toUpperCase() === 'SPAM' ? 1 : 0;
          const bSpam = (b.type || '').toUpperCase() === 'SPAM' ? 1 : 0;
          if (aSpam !== bSpam) return bSpam - aSpam;

          const aUrg = (a.urgency || '').toUpperCase() === 'HIGH' ? 1 : 0;
          const bUrg = (b.urgency || '').toUpperCase() === 'HIGH' ? 1 : 0;
          if (aUrg !== bUrg) return bUrg - aUrg;

          return (b.waitingTime || 0) - (a.waitingTime || 0);
        });

        const target = sorted[0];
        const type = (target.type || 'WORK').toUpperCase();
        const sentiment = target.sentiment || 'Professional';
        const urgency = (target.urgency || 'MEDIUM').toUpperCase();

        let action = 'OPEN';
        if (type === 'SPAM') action = 'DELETE';
        else if (sentiment === 'Aggressive') action = 'ESCALATE';
        else if (type === 'SUPPORT' && urgency === 'HIGH') action = 'ESCALATE';
        else if ((target.waitingTime || 0) > 25) action = 'OPEN';
        else if (urgency === 'HIGH') action = 'OPEN';
        else action = 'DEFER';

        handleAction(target.id, action);
      }, 1000);
    } else {
      clearInterval(autoAgentRef.current);
    }
    return () => clearInterval(autoAgentRef.current);
  }, [isAuto, isRunning, handleAction]);

  useEffect(() => {
    if (isRunning) {
      const id = setInterval(fetchClassifierStats, 15000);
      return () => clearInterval(id);
    }
  }, [isRunning]);

  const selectedEmail = emails.find((e) => e.id === selectedId);

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-700">
      <div className="max-w-[1600px] mx-auto p-4 md:p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 lg:h-screen lg:overflow-hidden min-h-screen">
        <aside className="lg:col-span-3 flex flex-col gap-5 lg:overflow-y-auto pr-2 lg:pb-8">
          <div className="bg-white p-6 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-11 h-11 bg-gradient-to-br from-indigo-600 to-violet-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-indigo-200">
                <Zap size={20} fill="currentColor" />
              </div>
              <div>
                <h1 className="font-black text-xl tracking-tight text-slate-800">InboxIQ</h1>
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                  RL Engine v3.0
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-3">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`group flex items-center justify-between px-5 py-3.5 rounded-2xl font-bold transition-all ${
                  isRunning
                    ? 'bg-amber-50 text-amber-600 hover:bg-amber-100 border border-amber-200'
                    : 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-700 hover:to-violet-700 shadow-xl shadow-indigo-100 hover:-translate-y-0.5'
                }`}
              >
                <span className="flex items-center gap-3">
                  {isRunning ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}
                  {isRunning ? 'Pause System' : 'Launch System'}
                </span>
                {isRunning && (
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500" />
                  </span>
                )}
              </button>

              <button
                onClick={() => {
                  setEmails([]);
                  setScore(0);
                  setLogs([]);
                  setLastReward(null);
                }}
                className="flex items-center gap-3 px-5 py-3 rounded-2xl font-bold text-slate-500 hover:bg-slate-100 transition-all active:scale-95"
              >
                <RotateCcw size={18} />
                Reset Data
              </button>
            </div>
          </div>

          <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-5 rounded-3xl shadow-xl text-white">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest flex items-center gap-2">
                <Brain size={12} /> ML Intelligence
              </h3>
              <button
                onClick={() => {
                  setShowStats(!showStats);
                  fetchClassifierStats();
                }}
                className="text-[9px] font-bold px-2 py-0.5 rounded-full bg-white/10 text-slate-300 hover:bg-white/20 transition-colors"
              >
                {showStats ? 'Hide' : 'Details'}
              </button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-white/5 rounded-xl p-3">
                <p className="text-[9px] text-slate-400 font-bold mb-1">Accuracy</p>
                <p className="text-lg font-black text-emerald-400">
                  {classifierStats?.live_classifier?.accuracy
                    ? `${(classifierStats.live_classifier.accuracy * 100).toFixed(1)}%`
                    : '—'}
                </p>
              </div>
              <div className="bg-white/5 rounded-xl p-3">
                <p className="text-[9px] text-slate-400 font-bold mb-1">Classified</p>
                <p className="text-lg font-black text-blue-400">
                  {classifierStats?.live_classifier?.total_classified || 0}
                </p>
              </div>
            </div>
            {showStats && classifierStats?.live_classifier && (
              <div className="mt-3 pt-3 border-t border-white/10 space-y-1.5 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-slate-400">Model Trained</span>
                  <span className={classifierStats.live_classifier.model_trained ? 'text-emerald-400' : 'text-red-400'}>
                    {classifierStats.live_classifier.model_trained ? 'Yes' : 'No'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Vocabulary Size</span>
                  <span className="text-slate-200">{classifierStats.live_classifier.vocab_size?.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Avg Reward (L50)</span>
                  <span className="text-slate-200">{classifierStats.live_classifier.avg_reward_last_50}</span>
                </div>
              </div>
            )}
          </div>

          <div className="bg-white p-5 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-slate-100 flex-1 flex flex-col min-h-0">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-bold text-slate-700 flex items-center gap-2">
                <Settings size={16} className="text-slate-400" />
                Accounts
              </h2>
              <button
                onClick={() => setShowAddAccount(true)}
                className="p-1.5 bg-slate-50 text-indigo-600 rounded-lg hover:bg-indigo-50 transition-colors"
              >
                <Plus size={16} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 mb-4 scrollbar-hide">
              {accounts.length === 0 ? (
                <p className="text-xs text-slate-400 italic p-4 text-center border-2 border-dashed border-slate-100 rounded-2xl">
                  No accounts connected
                </p>
              ) : (
                accounts.map((acc) => (
                  <div
                    key={acc}
                    className="flex items-center justify-between p-3 bg-slate-50 rounded-2xl hover:bg-slate-100/80 transition-colors group"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-violet-400 flex items-center justify-center text-xs font-bold text-white">
                        {acc[0].toUpperCase()}
                      </div>
                      <span className="text-xs font-bold text-slate-600 truncate">{acc}</span>
                    </div>
                    <CheckCircle2
                      size={14}
                      className="text-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity"
                    />
                  </div>
                ))
              )}
            </div>

            <div className="bg-indigo-50/50 p-4 rounded-2xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                  Agent Mode
                </span>
                <span
                  className={`text-[10px] font-bold px-2 py-0.5 rounded-full transition-colors ${
                    isAuto ? 'bg-indigo-600 text-white' : 'bg-white text-indigo-400 border border-indigo-200'
                  }`}
                >
                  {isAuto ? 'Active' : 'Standby'}
                </span>
              </div>
              <button
                onClick={() => setIsAuto(!isAuto)}
                className={`w-full p-2.5 rounded-xl text-sm font-black transition-all ${
                  isAuto ? 'bg-white text-indigo-600 shadow-sm border border-indigo-100' : 'bg-indigo-100 text-indigo-600 hover:bg-indigo-200'
                }`}
              >
                {isAuto ? 'Switch to Manual' : 'Activate Auto Agent'}
              </button>
            </div>
          </div>
        </aside>

        <main className="lg:col-span-6 flex flex-col gap-5 lg:overflow-hidden min-h-[500px]">
          <div className="grid grid-cols-3 gap-4 shrink-0">
            <div className="bg-white p-5 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.02)] border border-slate-100 relative overflow-hidden group">
              <TrendingUp
                size={40}
                className="absolute -right-2 -bottom-2 text-indigo-50 group-hover:text-indigo-100 transition-colors"
              />
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">
                Total Rewards
              </p>
              <div className="flex items-baseline gap-2">
                <span className={`text-3xl font-black ${score >= 0 ? 'text-slate-800' : 'text-red-600'}`}>
                  {score}
                </span>
                {lastReward && (
                  <span
                    key={lastReward.timestamp}
                    className={`text-xs font-bold animate-bounce ${lastReward.value >= 0 ? 'text-emerald-500' : 'text-red-500'}`}
                  >
                    {lastReward.value > 0 ? '+' : ''}
                    {lastReward.value}
                  </span>
                )}
              </div>
            </div>

            <div className="bg-white p-5 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.02)] border border-slate-100 relative overflow-hidden group">
              <Inbox
                size={40}
                className="absolute -right-2 -bottom-2 text-blue-50 group-hover:text-blue-100 transition-colors"
              />
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Inbox</p>
              <div className="flex items-center gap-3">
                <span className="text-3xl font-black text-slate-800">{emails.length}</span>
                {isSyncing && (
                  <div className="h-4 w-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                )}
              </div>
            </div>

            <div className="bg-white p-5 rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.02)] border border-slate-100 relative overflow-hidden group">
              <BarChart3
                size={40}
                className="absolute -right-2 -bottom-2 text-violet-50 group-hover:text-violet-100 transition-colors"
              />
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Decisions</p>
              <span className="text-3xl font-black text-slate-800">{logs.length}</span>
            </div>
          </div>

          <div className="bg-white rounded-[32px] shadow-[0_8px_40px_rgb(0,0,0,0.04)] border border-slate-100 flex-1 flex flex-col min-h-0">
            <div className="px-8 py-5 border-b border-slate-50 flex items-center justify-between shrink-0">
              <h2 className="text-lg font-black text-slate-800 flex items-center gap-3">
                Live Inbox Feed
                <span className="px-2.5 py-1 bg-slate-100 text-slate-400 text-[10px] rounded-lg font-bold">
                  Last 24H
                </span>
              </h2>
              <select
                className="text-xs font-bold text-indigo-600 bg-indigo-50 border-none rounded-xl px-3 py-1.5 focus:ring-0 cursor-pointer"
                value={simSpeed}
                onChange={(e) => setSimSpeed(Number(e.target.value))}
              >
                <option value={8000}>Slow (8s)</option>
                <option value={5000}>Normal (5s)</option>
                <option value={2000}>Turbo (2s)</option>
              </select>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-2 scroll-smooth">
              {emails.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-300 gap-4 opacity-60">
                  <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center">
                    <Mail size={40} strokeWidth={1} />
                  </div>
                  <p className="font-bold text-sm">
                    {isRunning ? 'Scanning...' : 'Press Launch to start'}
                  </p>
                </div>
              ) : (
                <div className="space-y-2.5 pb-8">
                  {emails.map((email) => {
                    const typeInfo = getEmailType(email.type);
                    const urgInfo = getUrgency(email.urgency);
                    const sentInfo = getSentiment(email.sentiment);
                    const isSelected = selectedId === email.id;
                    const isOld = (email.waitingTime || 0) > 40;

                    return (
                      <div
                        key={email.id}
                        onClick={() => setSelectedId(email.id)}
                        className={`relative group flex items-center gap-4 p-4 rounded-2xl transition-all cursor-pointer border-2 ${
                          isSelected
                            ? 'bg-white border-indigo-500 shadow-xl shadow-indigo-50 -translate-x-1'
                            : 'bg-white border-transparent hover:bg-slate-50 hover:border-slate-100'
                        }`}
                      >
                        <div
                          className={`shrink-0 w-11 h-11 rounded-xl flex items-center justify-center transition-colors ${
                            isSelected
                              ? 'bg-indigo-600 text-white'
                              : email.type?.toUpperCase() === 'SPAM'
                              ? 'bg-red-50 text-red-400'
                              : 'bg-slate-50 text-slate-400'
                          }`}
                        >
                          {React.createElement(typeInfo.icon, { size: 22 })}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                            <span className="text-sm font-black text-slate-800 tracking-tight truncate max-w-[180px]">
                              {email.sender}
                            </span>
                            <span className={`text-[8px] font-black uppercase tracking-tighter px-1.5 py-0.5 rounded ${urgInfo.bg} ${urgInfo.text}`}>
                              {urgInfo.label}
                            </span>
                            <span className={`text-[8px] font-black uppercase tracking-tighter px-1.5 py-0.5 rounded ${sentInfo.bg} ${sentInfo.color} flex items-center gap-0.5`}>
                              {sentInfo.icon} {email.sentiment}
                            </span>
                          </div>
                          <p className="text-xs text-slate-500 line-clamp-1">{email.subject}</p>
                        </div>

                        <div className="text-right shrink-0 flex flex-col items-end gap-1">
                          <div className="flex items-center gap-1">
                            <Timer size={10} className="text-slate-300" />
                            <p className={`text-[10px] font-bold ${isOld ? 'text-red-500' : 'text-slate-400'}`}>
                              {email.waitingTime || 0}s
                            </p>
                          </div>
                          <div className={`w-1.5 h-1.5 rounded-full ${isOld ? 'bg-red-400 animate-pulse' : 'bg-slate-200'}`} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </main>

        <aside className="lg:col-span-3 flex flex-col gap-5 lg:overflow-hidden min-h-[500px]">
          <div className="bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6 rounded-[32px] shadow-2xl relative overflow-hidden shrink-0">
            <div className="relative z-10">
              <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                <Target size={12} /> Action Inspector
              </h3>
              {selectedEmail ? (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`text-[9px] font-black px-2 py-0.5 rounded ${selectedEmail.type?.toUpperCase() === 'SPAM' ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
                      {(selectedEmail.type || 'WORK').toUpperCase()}
                    </span>
                    <span className={`text-[9px] font-black px-2 py-0.5 rounded ${getUrgency(selectedEmail.urgency).text} bg-white/10`}>
                      {getUrgency(selectedEmail.urgency).label}
                    </span>
                  </div>
                  <h4 className="font-bold text-base leading-tight mb-1 line-clamp-2">{selectedEmail.subject}</h4>
                  <p className="text-slate-400 text-xs mb-1 truncate">{selectedEmail.sender}</p>

                  <div className="grid grid-cols-2 gap-2.5 mt-4">
                    <InspectorBtn onClick={() => handleAction(selectedEmail.id, 'OPEN')} icon={<Eye size={14} />} label="Open" color="bg-blue-500" />
                    <InspectorBtn onClick={() => handleAction(selectedEmail.id, 'DELETE')} icon={<Trash2 size={14} />} label="Trash" color="bg-red-500" />
                    <InspectorBtn onClick={() => handleAction(selectedEmail.id, 'DEFER')} icon={<Clock size={14} />} label="Defer" color="bg-slate-600" />
                    <InspectorBtn onClick={() => handleAction(selectedEmail.id, 'ESCALATE')} icon={<ArrowUpCircle size={14} />} label="Escalate" color="bg-purple-500" />
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-center text-slate-600 border-2 border-dashed border-slate-700 rounded-3xl">
                  <ChevronRight size={32} className="mb-2 opacity-20" />
                  <p className="text-xs font-medium">Select an email</p>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white p-5 rounded-[32px] shadow-[0_8px_30px_rgb(0,0,0,0.02)] border border-slate-100 flex-1 flex flex-col min-h-0 overflow-hidden">
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
              <BarChart3 size={12} /> Decision Log
            </h3>
            <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 scrollbar-hide">
              {logs.map((log, i) => (
                <div key={i} className={`flex items-start justify-between p-3.5 rounded-2xl border-l-4 ${log.reward >= 0 ? 'bg-emerald-50/50 border-emerald-300' : 'bg-red-50/50 border-red-300'}`}>
                  <div className="min-w-0">
                    <p className="text-xs font-black text-slate-700 truncate">{log.msg}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[9px] font-bold text-slate-400">{log.time}</span>
                      <span className="text-[8px] font-bold uppercase text-slate-300 bg-slate-100 px-1.5 py-0.5 rounded">{log.action}</span>
                    </div>
                  </div>
                  <span className={`text-xs font-black shrink-0 ml-2 ${log.reward >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                    {log.reward > 0 ? '+' : ''}{log.reward}
                  </span>
                </div>
              ))}
              {logs.length === 0 && <p className="text-[10px] text-slate-300 italic text-center p-8">No activity yet</p>}
            </div>
          </div>
        </aside>

        {showAddAccount && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
            <div className="bg-white w-full max-w-md rounded-[32px] p-8 shadow-2xl">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-black text-slate-800">Connect Account</h3>
                <button onClick={() => setShowAddAccount(false)} className="p-2 hover:bg-slate-100 rounded-full">
                  <X size={20} />
                </button>
              </div>

              <form onSubmit={addAccount} className="space-y-5">
                <div>
                  <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Gmail Address</label>
                  <input required type="text" value={newEmail} onChange={(e) => setNewEmail(e.target.value)} className="w-full px-5 py-3.5 bg-slate-50 border-2 border-transparent rounded-2xl focus:bg-white focus:border-indigo-500 outline-none transition-all font-bold" placeholder="you@gmail.com or demo" />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest">App Password</label>
                    <a href="https://myaccount.google.com/apppasswords" target="_blank" rel="noopener noreferrer" className="text-[9px] font-bold text-indigo-500 hover:underline">Get one here</a>
                  </div>
                  <input required type="password" value={newPass} onChange={(e) => setNewPass(e.target.value)} className="w-full px-5 py-3.5 bg-slate-50 border-2 border-transparent rounded-2xl focus:bg-white focus:border-indigo-500 outline-none transition-all font-bold" placeholder="xxxx xxxx xxxx xxxx" />
                  {errMessage && <p className="text-[10px] text-red-500 mt-2 font-bold flex items-center gap-1"><AlertTriangle size={10} /> {errMessage}</p>}
                </div>
                
                <div className="flex gap-3">
                  <button onClick={(e) => {
                    e.preventDefault();
                    setNewEmail('demo');
                    setNewPass('demo');
                    setTimeout(() => document.getElementById('connect-btn').click(), 10);
                  }} className="flex-1 py-4 bg-slate-100 text-slate-600 font-black rounded-2xl hover:bg-slate-200 transition-all text-sm">
                    Try Demo
                  </button>
                  <button id="connect-btn" disabled={isConnecting} className={`flex-[2] text-white font-black py-4 rounded-2xl shadow-xl transition-all flex items-center justify-center gap-3 ${isConnecting ? 'bg-slate-400' : 'bg-gradient-to-r from-indigo-600 to-violet-600'}`}>
                    {isConnecting && <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
                    {isConnecting ? 'Validating...' : 'Connect Account'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>

      <style dangerouslySetInnerHTML={{ __html: `.scrollbar-hide::-webkit-scrollbar { display: none; } .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }` }} />
    </div>
  );
}

function InspectorBtn({ onClick, icon, label, color }) {
  return (
    <button onClick={onClick} className={`w-full py-3 rounded-xl font-black text-xs text-white transition-all active:scale-95 flex items-center justify-center gap-2 ${color} hover:opacity-90`}>
      {icon} {label}
    </button>
  );
}
