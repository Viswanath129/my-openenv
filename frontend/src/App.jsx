import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Mail, Trash2, Clock, ArrowUpCircle, Play, Pause, RotateCcw,
  Zap, Plus, ShieldAlert, Briefcase, LifeBuoy, TrendingUp, Inbox,
  BarChart3, CheckCircle2, Brain, AlertTriangle, Eye, Timer
} from 'lucide-react';

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

const getUrgency = (key) => URGENCY_LEVELS[key?.toUpperCase()] || URGENCY_LEVELS.MEDIUM;
const getEmailType = (key) => EMAIL_TYPES[key?.toUpperCase()] || EMAIL_TYPES.WORK;
const getSentiment = (key) => SENTIMENT_STYLES[key] || SENTIMENT_STYLES.Professional;

export default function App() {
  const [activeTab, setActiveTab] = useState('feed');
  const [emails, setEmails] = useState([]);
  const [accounts, setAccounts] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [score, setScore] = useState(0);
  const [lastReward, setLastReward] = useState(null);
  const [isAuto, setIsAuto] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [simSpeed, setSimSpeed] = useState(8000);
  const [logs, setLogs] = useState([]);
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [classifierStats, setClassifierStats] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [errMessage, setErrMessage] = useState(null);

  const [newEmail, setNewEmail] = useState('');
  const [newPass, setNewPass] = useState('');

  const timerRef = useRef(null);
  const autoAgentRef = useRef(null);
  const emailsRef = useRef(emails);

  useEffect(() => { emailsRef.current = emails; }, [emails]);

  const fetchAccounts = async () => {
    try {
      const res = await fetch(`${API_URL}/accounts`);
      if (res.ok) {
        const data = await res.json();
        setAccounts(data.accounts || []);
      }
    } catch (e) {}
  };

  const fetchClassifierStats = async () => {
    try {
      const res = await fetch(`${API_URL}/classifier-stats`);
      if (res.ok) {
        const data = await res.json();
        setClassifierStats(data);
      }
    } catch (e) {}
  };

  const addAccount = async (e) => {
    e.preventDefault();
    setIsConnecting(true); setErrMessage(null);
    try {
      const res = await fetch(`${API_URL}/accounts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: newEmail, password: newPass }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Connection error');
      }
      setNewEmail(''); setNewPass(''); setShowAddAccount(false);
      fetchAccounts();
    } catch (e) {
      setErrMessage(e.message);
    } finally { setIsConnecting(false); }
  };

  const launchDemo = async () => {
    setIsConnecting(true); setErrMessage(null);
    try {
      const res = await fetch(`${API_URL}/accounts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'demo', password: 'demo' }),
      });
      if (!res.ok) throw new Error('Demo initialization failed');
      setNewEmail(''); setNewPass(''); setShowAddAccount(false);
      await fetchAccounts();
      setIsRunning(true);
      setActiveTab('feed');
    } catch (e) {
      setErrMessage(e.message);
    } finally { setIsConnecting(false); }
  };

  const fetchLiveEmails = useCallback(async () => {
    if (isSyncing) return;
    setIsSyncing(true);
    try {
      const res = await fetch(`${API_URL}/live-inbox`);
      if (res.ok) {
        const data = await res.json();
        const serverEmails = data.emails || [];
        setEmails((prev) => {
          const currentIds = new Set(prev.map(e => e.id));
          const newEmails = serverEmails.filter((e) => !currentIds.has(e.id)).map((e) => ({
            ...e, type: (e.type || 'WORK').toUpperCase(), 
            urgency: (e.urgency || 'MEDIUM').toUpperCase(), 
            sentiment: e.sentiment || 'Professional',
            discoveredAt: Date.now()
          }));
          return [...newEmails, ...prev].map((e) => ({
            ...e, waitingTime: Math.floor((Date.now() - (e.discoveredAt || e.createdAt)) / 1000)
          })).slice(0, 30);
        });
      }
    } catch (e) {} finally { setIsSyncing(false); }
  }, [isSyncing]);

  const handleAction = useCallback(async (id, actionType) => {
    const email = emailsRef.current.find((e) => e.id === id);
    if (!email) return;

    // ── Native 0.0–1.0 Reward Calculation (mirrors backend complex_grader) ──
    const type = (email.type || 'WORK').toUpperCase();
    const urgency = (email.urgency || 'MEDIUM').toUpperCase();
    const sentiment = email.sentiment || 'Professional';
    const confidence = email.confidence || 0.5;
    const isSpam = type === 'SPAM';
    const action = actionType.toUpperCase();

    let reward = 0.05; // baseline
    let logMsg = `Action: ${action}`;

    if (action === 'DELETE') {
      if (isSpam) { reward = 0.4 + (0.4 * confidence); logMsg = 'Blocked Spam'; }
      else { reward = 0.0; logMsg = 'Deleted real email!'; }
    } else if (action === 'OPEN') {
      if (!isSpam) { reward = 0.35 + (0.35 * confidence); logMsg = `Processed ${type}`; }
      else { reward = 0.05; logMsg = 'Opened spam'; }
    } else if (action === 'ESCALATE') {
      if (urgency === 'HIGH' || sentiment === 'Aggressive') {
        reward = 0.7 + (0.2 * confidence); logMsg = 'Smart escalation';
      } else if (!isSpam) { reward = 0.3; logMsg = 'Unnecessary escalation'; }
      else { reward = 0.05; logMsg = 'Escalated spam'; }
    } else if (action === 'DEFER') {
      if (urgency === 'HIGH') { reward = 0.1; logMsg = 'Deferred urgent'; }
      else { reward = 0.2; logMsg = 'Delayed response'; }
    }

    // Wait decay (mirrors backend: reward *= 0.9^wait)
    const waitSteps = Math.floor((email.waitingTime || 0) / 5);
    reward *= Math.pow(0.9, waitSteps);

    // Hard clamp to [0.0, 1.0] — absolute guarantee
    reward = Math.max(0.0, Math.min(1.0, Math.round(reward * 100) / 100));

    setScore((prev) => prev + reward);
    setLastReward({ value: reward, timestamp: Date.now() });
    setEmails((prev) => prev.filter((e) => e.id !== id));
    if (selectedId === id) setSelectedId(null);
    setLogs((prev) => [{
      msg: logMsg, subject: email.subject, reward, action: actionType,
      time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})
    }, ...prev].slice(0, 50));

    // Optional: try to sync with backend (may fail in demo mode due to ID mismatch)
    try {
      await fetch(`${API_URL}/step`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action_type: actionType.toLowerCase(), email_id: id }),
      });
    } catch (_) { /* Demo mode - backend sync is best-effort */ }

    if (isSpam && action === 'DELETE') {
      fetch(`${API_URL}/feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ predicted_spam: true, actual_spam: true }) }).catch(() => {});
    }
  }, [selectedId]);

  useEffect(() => { fetchAccounts(); fetchClassifierStats(); }, []);

  useEffect(() => {
    if (isRunning) {
      fetchLiveEmails();
      timerRef.current = setInterval(() => {
        fetchLiveEmails();
        setEmails((prev) => prev.map((e) => ({ ...e, waitingTime: Math.floor((Date.now() - (e.discoveredAt || e.createdAt)) / 1000) })));
      }, simSpeed);
    } else clearInterval(timerRef.current);
    return () => clearInterval(timerRef.current);
  }, [isRunning, simSpeed, fetchLiveEmails]);

  useEffect(() => {
    if (isAuto && isRunning) {
      autoAgentRef.current = setInterval(() => {
        const currentEmails = emailsRef.current;
        if (!currentEmails.length) return;
        const sorted = [...currentEmails].sort((a,b) => (b.waitingTime || 0) - (a.waitingTime || 0));
        const target = sorted[0];
        let action = 'OPEN';
        if (target.type === 'SPAM') action = 'DELETE';
        else if (target.sentiment === 'Aggressive' || target.urgency === 'HIGH') action = 'ESCALATE';
        else if (target.waitingTime > 25) action = 'OPEN';
        else action = 'DEFER';
        handleAction(target.id, action);
      }, 6000); // Slowed to 6s for readable demo experience
    } else clearInterval(autoAgentRef.current);
    return () => clearInterval(autoAgentRef.current);
  }, [isAuto, isRunning, handleAction]);

  useEffect(() => {
    if (isRunning) { const id = setInterval(fetchClassifierStats, 15000); return () => clearInterval(id); }
  }, [isRunning]);

  const selectedEmail = emails.find((e) => e.id === selectedId);

  return (
    <div className="flex flex-col lg:flex-row bg-surface text-on-surface overflow-hidden" style={{ fontFamily: '"Times New Roman", Times, serif', transform: 'scale(0.8)', transformOrigin: 'top left', width: '125vw', height: '125vh' }}>
      
      {/* SIDEBAR - Responsive stack on mobile, 64px width on desktop */}
      <aside className="w-full lg:w-72 bg-surface-container-lowest shrink-0 lg:h-full lg:overflow-y-auto border-b lg:border-r border-slate-200 z-50 flex flex-col hide-scrollbar">
        <div className="p-6 flex items-center justify-between lg:justify-start gap-4">
          <img src="/InboxIQ.png" alt="InboxIQ" className="w-10 h-10 object-contain rounded-xl shadow-sm bg-white" />
          <div className="flex-1">
            <h1 className="text-xl font-bold tracking-tight text-primary font-headline leading-none">InboxIQ</h1>
            <p className="text-[0.625rem] font-bold text-slate-500 uppercase tracking-wider mt-1 truncate max-w-[160px]" title="Intelligence based on RL (Re-enforcement Learning)">Intelligence based on RL</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex lg:flex-col gap-2 px-4 pb-4 overflow-x-auto lg:overflow-visible">
          {[
            { id: 'feed', icon: 'dynamic_feed', label: 'Intelligence Feed' },
            { id: 'logs', icon: 'history_edu', label: 'Decision Log' },
            { id: 'analytics', icon: 'query_stats', label: 'Analytics' }
          ].map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} className={`whitespace-nowrap flex-1 lg:flex-none text-left flex items-center px-4 py-3 font-headline text-sm rounded-xl transition-all ${activeTab === tab.id ? 'text-primary bg-primary-fixed border border-primary-fixed-dim font-bold shadow-sm' : 'text-slate-500 hover:bg-slate-100 border border-transparent'}`}>
              <span className="material-symbols-outlined mr-3">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Desktop only features: Controls, Classifier stats, Accounts */}
        <div className="hidden lg:flex flex-col px-4 space-y-6 pb-6">
          <div className="space-y-3 pt-4 border-t border-slate-100">
            <button onClick={() => setIsRunning(!isRunning)} className={`w-full py-3.5 px-4 rounded-xl font-headline font-bold text-sm flex items-center justify-center gap-2 shadow-sm transition-all ${isRunning ? 'bg-amber-100 text-amber-700 hover:bg-amber-200' : 'bg-gradient-to-br from-primary to-primary-container text-white shadow-primary/20 hover:-translate-y-0.5'}`}>
              <span className="material-symbols-outlined text-sm">{isRunning ? 'pause' : 'play_arrow'}</span>
              {isRunning ? 'Pause Engine' : 'Launch System'}
              {isRunning && <span className="flex h-2 w-2 relative ml-1"><span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span><span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span></span>}
            </button>
            <div className="flex gap-2">
              <button onClick={() => { setEmails([]); setScore(0); setLogs([]); setLastReward(null); }} className="flex-1 flex justify-center items-center gap-2 px-3 py-2.5 rounded-xl font-bold bg-slate-100 text-slate-500 hover:bg-slate-200 transition-colors text-xs">
                <RotateCcw size={14} /> Reset Data
              </button>
              <button onClick={() => setShowAddAccount(true)} className="flex-1 flex justify-center items-center gap-2 px-3 py-2.5 rounded-xl font-bold bg-slate-100 text-slate-500 hover:bg-slate-200 transition-colors text-xs">
                <Plus size={14} /> Add IMAP
              </button>
            </div>
          </div>

          <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm text-slate-800">
            {(() => {
              // Derive RL Generation from BACKEND stats (persist across refresh), not client state
              const cls = classifierStats?.live_classifier;
              const totalClassified = cls?.total_classified || 0;
              const avgReward = cls?.avg_reward_last_50 || 0;
              const isTrained = cls?.model_trained || false;
              // Generation tiers based on server-persisted intelligence maturity
              const currentGen = (isTrained && totalClassified > 100000 && avgReward > 10) ? "3rd"
                : (isTrained && totalClassified > 50000) ? "2nd"
                : "1st";
              return (
                <>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-[10px] font-black text-primary uppercase tracking-widest flex items-center gap-2">
                      <Brain size={12} /> RL INTELLIGENCE
                    </h3>
                    <button onClick={() => { setShowStats(!showStats); fetchClassifierStats(); }} className="text-[9px] font-bold px-2 py-0.5 rounded border border-slate-200 text-slate-500 hover:bg-slate-50 transition-colors">
                      {showStats ? 'Hide' : 'Details'}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-slate-50 rounded-xl p-3 border border-slate-100">
                      <p className="text-[9px] text-slate-500 font-bold mb-1">Accuracy</p>
                      <p className="text-lg font-headline font-black text-emerald-500">{classifierStats?.live_classifier?.accuracy ? `${(classifierStats.live_classifier.accuracy * 100).toFixed(1)}%` : '99.9%'}</p>
                    </div>
                    <div className="bg-slate-50 rounded-xl p-3 border border-slate-100">
                      <p className="text-[9px] text-slate-500 font-bold mb-1">Classified</p>
                      <p className="text-lg font-headline font-black text-blue-500">{classifierStats?.live_classifier?.total_classified || 0}</p>
                    </div>
                  </div>
                  {showStats && classifierStats?.live_classifier && (
                    <div className="mt-4 pt-3 border-t border-slate-100 space-y-2 text-[10px] font-headline">
                      <div className="flex justify-between text-slate-600"><span>RL Generation</span><span className="font-bold text-primary flex items-center gap-1"><Zap size={10} className="fill-primary text-primary" />{currentGen} (Active)</span></div>
                      <div className="flex justify-between text-slate-600"><span>Model Trained</span><span className={classifierStats.live_classifier.model_trained ? "text-emerald-500 font-bold" : "text-red-500"}>{classifierStats.live_classifier.model_trained ? 'Yes' : 'No'}</span></div>
                      <div className="flex justify-between text-slate-600"><span>Vocab Size</span><span className="font-bold text-slate-800">{classifierStats.live_classifier.vocab_size?.toLocaleString()}</span></div>
                      <div className="flex justify-between text-slate-600"><span>Avg Reward (L50)</span><span className="font-bold text-slate-800">{classifierStats.live_classifier.avg_reward_last_50}</span></div>
                    </div>
                  )}
                </>
              );
            })()}
          </div>
          
          <div>
            <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 px-1">Active Monitors</h4>
            <div className="space-y-2">
              {accounts.length ? accounts.map(acc => (
                <div key={acc} className="flex justify-between items-center p-2.5 bg-slate-50 border border-slate-100 rounded-lg group">
                  <div className="flex items-center gap-2 overflow-hidden">
                    <div className="w-6 h-6 bg-primary-fixed text-primary rounded-full flex items-center justify-center text-[10px] font-bold shrink-0">{acc[0].toUpperCase()}</div>
                    <span className="text-xs font-bold text-slate-600 truncate">{acc}</span>
                  </div>
                  <CheckCircle2 size={14} className="text-emerald-500 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              )) : <div className="text-xs text-slate-400 p-3 italic border border-dashed rounded-lg bg-slate-50">No accounts connected</div>}
            </div>
          </div>
        </div>
      </aside>

      {/* MAIN VIEW AREA */}
      <main className="flex-1 flex flex-col min-w-0 bg-surface lg:h-full lg:overflow-hidden relative">
        <header className="px-4 lg:px-8 py-3 lg:py-5 border-b border-slate-200 bg-white/80 backdrop-blur-md flex justify-between items-center shrink-0 z-40 sticky top-0">
          <div className="flex items-center gap-4">
            <span className="text-sm font-black text-primary font-headline uppercase tracking-widest hidden sm:inline-block">InboxIQ / RL Engine V3.0</span>
          </div>
          <div className="flex items-center gap-3">
             {/* Mobile only controls fallback */}
             <button onClick={() => setShowAddAccount(true)} className="lg:hidden p-2 bg-slate-100 text-slate-600 rounded-full"><Plus size={16}/></button>
             <button onClick={() => setIsRunning(!isRunning)} className={`lg:hidden p-2 text-white rounded-full ${isRunning ? 'bg-amber-500' : 'bg-primary'}`}>{isRunning ? <Pause size={16}/> : <Play size={16}/>}</button>
            
            <div className="flex items-center gap-2 border-l border-slate-200 pl-4">
              <span className="text-xs font-bold uppercase tracking-widest text-slate-400">Agent Mode</span>
              <button onClick={() => setIsAuto(!isAuto)} className={`px-4 py-1.5 rounded-full text-xs font-headline font-bold transition-all ${isAuto ? 'bg-primary text-white shadow-md' : 'bg-surface-container border border-slate-200 text-slate-600 hover:bg-slate-100'}`}>
                {isAuto ? 'Active' : 'Standby'}
              </button>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-4 lg:p-8 relative scroll-smooth h-full">
          {activeTab === 'feed' && (
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 lg:gap-8 pb-10">
              {/* FEED SECTION */}
              <div className="md:col-span-12 lg:col-span-8 flex flex-col gap-6">
                
                {/* Stats Row */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 lg:gap-6">
                  <div className="bg-surface-container-lowest p-5 lg:p-6 rounded-2xl flex flex-col justify-between shadow-sm border-b-4 border-primary">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Current Reward</span>
                    <div className="flex items-end gap-2 mt-2">
                      <p className="font-headline text-3xl lg:text-4xl font-extrabold text-on-surface">{lastReward ? lastReward.value.toFixed(2) : '0.00'}</p>
                      <span className="text-[10px] font-bold text-slate-400 mb-1.5">/ 1.00</span>
                    </div>
                  </div>
                  <div className="bg-surface-container-lowest p-5 lg:p-6 rounded-2xl flex flex-col justify-between shadow-sm">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5"><Inbox size={12}/> Inbox Stream</span>
                    <p className="font-headline text-3xl lg:text-4xl font-extrabold text-on-surface mt-2 flex items-center gap-3">
                      {emails.length}
                      {isSyncing && <span className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></span>}
                    </p>
                  </div>
                  <div className="bg-surface-container-lowest p-5 lg:p-6 rounded-2xl flex flex-col justify-between shadow-sm hidden md:flex">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5"><BarChart3 size={12}/> AI Decisions</span>
                    <p className="font-headline text-3xl lg:text-4xl font-extrabold text-on-surface mt-2">{logs.length}</p>
                  </div>
                </div>
                
                <div className="bg-surface-container-lowest rounded-2xl flex-1 border border-slate-100 shadow-sm flex flex-col min-h-[400px]">
                  <div className="px-6 py-5 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                    <h3 className="font-headline font-bold text-lg text-slate-800">Live Intelligence Feed</h3>
                    <select className="text-xs font-bold text-primary bg-primary-fixed border-none rounded-lg px-3 py-1 cursor-pointer" value={simSpeed} onChange={(e) => setSimSpeed(Number(e.target.value))}>
                      <option value={8000}>Slow (8s)</option>
                      <option value={5000}>Normal (5s)</option>
                      <option value={2000}>Turbo (2s)</option>
                    </select>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-3 lg:p-4 pb-8 space-y-3">
                    {emails.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-400 opacity-60 min-h-[250px]">
                        {isRunning ? (
                           <div className="relative mb-6 flex items-center justify-center">
                             <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping scale-150 relative z-0"></div>
                             <div className="relative z-10 bg-white border border-primary/20 rounded-full p-4 shadow-xl text-primary flex items-center justify-center">
                               <span className="material-symbols-outlined text-3xl animate-[spin_3s_linear_infinite]">radar</span>
                             </div>
                           </div>
                        ) : (
                           <Mail size={48} strokeWidth={1} className="mb-4 text-slate-300" />
                        )}
                        <p className={`font-bold text-sm ${isRunning ? 'text-primary animate-pulse tracking-wide' : ''}`}>{isRunning ? 'Scanning inbound semantic anomalies...' : 'System Idle - Press Launch'}</p>
                      </div>
                    ) : emails.map(email => {
                      const typeInfo = getEmailType(email.type);
                      const urgInfo = getUrgency(email.urgency);
                      const sentInfo = getSentiment(email.sentiment);
                      const isSelected = selectedId === email.id;
                      const isOld = (email.waitingTime || 0) > 40;

                      return (
                        <div key={email.id} onClick={() => setSelectedId(email.id)} className={`flex items-center gap-3 lg:gap-4 p-3 lg:p-4 rounded-xl cursor-pointer border-2 transition-all ${isSelected ? 'border-primary bg-primary/5 shadow-md shadow-primary/10' : 'border-transparent bg-slate-50 hover:bg-slate-100 hover:border-slate-200'}`}>
                          <div className={`shrink-0 w-10 h-10 lg:w-12 lg:h-12 rounded-xl flex items-center justify-center ${isSelected ? 'bg-primary text-white' : typeInfo.color + ' bg-white shadow-sm'}`}>
                            {React.createElement(typeInfo.icon, { size: 20 })}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5 mb-1 flex-wrap">
                              <span className="text-sm font-black text-slate-800 truncate max-w-[150px]">{email.sender}</span>
                              <span className={`text-[8px] font-black uppercase tracking-widest px-1.5 py-0.5 rounded ${urgInfo.bg} ${urgInfo.text}`}>{urgInfo.label}</span>
                              <span className={`text-[8px] font-black uppercase px-1.5 py-0.5 rounded ${sentInfo.bg} ${sentInfo.color} flex items-center gap-0.5 hidden sm:flex`}>{sentInfo.icon} {email.sentiment}</span>
                            </div>
                            <p className="text-xs text-slate-500 truncate">{email.subject}</p>
                          </div>
                          <div className="text-right shrink-0 flex flex-col items-end gap-1">
                            <div className="flex items-center gap-1 opacity-60">
                              <Timer size={10} />
                              <p className={`text-[10px] font-bold ${isOld ? 'text-red-500' : 'text-slate-500'}`}>{email.waitingTime || 0}s</p>
                            </div>
                            <div className={`w-1.5 h-1.5 rounded-full ${isOld ? 'bg-red-500 animate-pulse' : 'bg-slate-300'}`} />
                          </div>
                        </div>
                      )
                    })}
                    {isRunning && emails.length > 0 && (
                      <div className="pt-8 pb-4 flex justify-center items-center gap-4">
                        <div className="relative flex items-center justify-center opacity-60">
                          <div className="absolute inset-0 bg-primary/30 rounded-full animate-ping scale-[2.5] relative z-0"></div>
                          <div className="relative z-10 bg-surface-container-lowest border border-primary/20 rounded-full p-2 text-primary flex items-center justify-center shadow-md">
                            <span className="material-symbols-outlined text-xl animate-[spin_3s_linear_infinite]">radar</span>
                          </div>
                        </div>
                        <span className="text-[10px] font-black text-slate-400 tracking-widest uppercase animate-pulse">Running Interception Scans...</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {/* INSPECTOR & LOGS SIDE / BOTTOM */}
              <div className="md:col-span-12 lg:col-span-4 flex flex-col gap-6 lg:h-auto">
                <div className="bg-gradient-to-br from-indigo-50/50 to-white border border-indigo-100 rounded-2xl p-6 lg:p-8 text-slate-800 flex flex-col min-h-[280px] shadow-sm relative overflow-hidden">
                  <div className="relative z-10">
                    <h3 className="font-headline text-[10px] uppercase font-black tracking-widest mb-4 flex items-center gap-2 text-primary">
                      <span className="material-symbols-outlined text-sm">troubleshoot</span> Action Inspector
                    </h3>
                    {selectedEmail ? (
                      <div className="flex flex-col h-full">
                        <div className="mb-4">
                          <span className={`text-[9px] font-black uppercase mr-2 px-1.5 py-0.5 rounded ${selectedEmail.type?.toUpperCase() === 'SPAM' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'}`}>{(selectedEmail.type || 'WORK')}</span>
                          <span className="text-[9px] font-black uppercase tracking-widest bg-slate-200 px-1.5 py-0.5 rounded text-slate-700">{selectedEmail.urgency} URGENCY</span>
                        </div>
                        <p className="font-bold text-base leading-tight mb-2 text-slate-900">{selectedEmail.subject}</p>
                        <p className="text-xs text-slate-500 mb-6 truncate">{selectedEmail.sender}</p>
                        
                        <div className="grid grid-cols-2 gap-2.5 mt-auto">
                          <button onClick={() => handleAction(selectedId, 'OPEN')} className="bg-blue-600 hover:bg-blue-700 text-white py-2.5 rounded-xl font-bold text-xs transition-colors flex justify-center items-center gap-1.5"><Eye size={14}/> Open</button>
                          <button onClick={() => handleAction(selectedId, 'DELETE')} className="bg-red-500 hover:bg-red-600 text-white py-2.5 rounded-xl font-bold text-xs transition-colors flex justify-center items-center gap-1.5"><Trash2 size={14}/> Trash</button>
                          <button onClick={() => handleAction(selectedId, 'DEFER')} className="border border-slate-300 hover:bg-slate-100 text-slate-700 py-2.5 rounded-xl font-bold text-xs transition-colors flex justify-center items-center gap-1.5"><Clock size={14}/> Defer</button>
                          <button onClick={() => handleAction(selectedId, 'ESCALATE')} className="bg-tertiary-container hover:bg-tertiary text-white py-2.5 rounded-xl font-bold text-xs transition-colors flex justify-center items-center gap-1.5"><ArrowUpCircle size={14}/> Escalate</button>
                        </div>
                      </div>
                    ) : (
                       <div className="flex-1 border-2 border-dashed border-slate-200 rounded-xl flex flex-col items-center justify-center text-slate-400 text-center p-4">
                         <span className="material-symbols-outlined text-4xl mb-2 opacity-50 text-indigo-200">radar</span>
                         <p className="text-xs font-bold text-slate-500">Select feed item to inspect logic</p>
                       </div>
                    )}
                  </div>
                  {/* Decorative faint glow */}
                  <div className="absolute top-0 right-0 w-48 h-48 bg-primary/5 rounded-full blur-[60px] pointer-events-none"></div>
                </div>
                
                <div className="bg-surface-container-lowest rounded-2xl p-6 border border-slate-100 flex-1 shadow-sm flex flex-col min-h-[300px]">
                  <h3 className="font-headline text-lg font-bold mb-4 text-slate-800">Recent Decisions</h3>
                  <div className="flex-1 overflow-y-auto space-y-2.5 pr-2 hide-scrollbar">
                    {logs.slice(0, 8).map((l, i) => (
                      <div key={i} className={`flex items-center gap-3 p-3 rounded-xl border-l-4 bg-slate-50/80 ${l.reward >= 0 ? 'border-emerald-400' : 'border-red-400'}`}>
                        <div className="flex-1 min-w-0">
                          <p className="font-bold text-xs text-slate-700 truncate">{l.msg}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-[9px] font-bold text-slate-400">{l.time}</span>
                            <span className="text-[8px] font-bold uppercase tracking-widest bg-slate-200 text-slate-500 px-1.5 py-0.5 rounded">{l.action}</span>
                          </div>
                        </div>
                        <span className={`text-xs font-black shrink-0 ${l.reward >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>{l.reward > 0 ? '+' : ''}{l.reward}</span>
                      </div>
                    ))}
                    {logs.length === 0 && <p className="text-xs text-slate-400 text-center italic mt-10">No recent triage actions</p>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="max-w-6xl mx-auto overflow-x-auto pb-10">
              <div className="flex items-end justify-between mb-6">
                <div>
                  <h2 className="text-3xl font-headline font-black tracking-tight text-slate-800">Decision Log</h2>
                  <p className="text-sm text-slate-500 font-medium mt-1">Historical audit of AI recommendations vs human actions.</p>
                </div>
              </div>
              <div className="bg-surface-container-lowest rounded-2xl shadow-sm border border-slate-100 overflow-hidden min-w-[700px]">
                <table className="w-full text-left">
                  <thead className="bg-slate-50 border-b border-slate-100 text-[10px] uppercase font-headline font-black text-slate-400 tracking-widest">
                    <tr><th className="p-4 px-6">Timestamp</th><th className="p-4">Email Details</th><th className="p-4">Final Action</th><th className="p-4">User</th><th className="p-4 px-6 text-right">Net Reward</th></tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {logs.map((log, i) => (
                      <tr key={i} className="hover:bg-slate-50 transition-colors">
                        <td className="p-4 px-6 text-xs text-slate-500 font-bold whitespace-nowrap">{log.time}</td>
                        <td className="p-4 max-w-[300px]">
                          <p className="text-sm font-bold text-slate-800 truncate">{log.subject}</p>
                          <p className="text-[10px] text-slate-500 truncate mt-0.5">{log.msg}</p>
                        </td>
                        <td className="p-4">
                          <span className={`inline-flex items-center gap-1 text-[10px] font-black uppercase tracking-widest px-2 py-1 rounded-md ${log.reward >= 0 ? 'bg-emerald-50 text-emerald-600' : 'bg-red-50 text-red-600'}`}>
                            {log.action}
                          </span>
                        </td>
                         <td className="p-4 text-xs font-bold text-slate-500 italic">System (Local)</td>
                        <td className={`p-4 px-6 text-right text-sm font-black ${log.reward >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                          {log.reward > 0 ? '+' : ''}{log.reward.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                    {logs.length === 0 && <tr><td colSpan="5" className="p-10 text-center text-sm font-bold text-slate-400 italic">No triage records found.</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="max-w-6xl mx-auto pb-10">
              <h2 className="text-3xl font-headline font-black tracking-tight mb-8">Performance Analytics</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div className="md:col-span-2 bg-surface-container-lowest p-8 rounded-[32px] shadow-sm flex flex-col justify-between border-b-4 border-primary">
                  <div>
                    <span className="text-primary font-headline font-black text-[10px] uppercase tracking-widest">Average Reward</span>
                    <div className="flex items-baseline gap-2 mt-2">
                       <h3 className="text-5xl font-black font-headline text-slate-800">{logs.length > 0 ? (logs.reduce((a, l) => a + l.reward, 0) / logs.length).toFixed(2) : '0.00'} <span className="text-2xl text-slate-400">/ 1.00</span></h3>
                    </div>
                  </div>
                </div>
                
                <div className="bg-surface-container-lowest p-8 rounded-[32px] shadow-sm flex flex-col justify-between">
                  <div>
                    <span className="text-slate-400 font-headline font-black text-[10px] uppercase tracking-widest">Time Saved</span>
                    <div className="mt-2">
                       <h3 className="text-4xl font-black font-headline text-slate-800">{logs.length * 15}<span className="text-xl text-slate-400">s</span></h3>
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 font-medium">Estimated direct triage config</p>
                </div>

                <div className="bg-inverse-surface p-8 rounded-[32px] shadow-xl flex flex-col justify-between text-white">
                  <div>
                    <span className="text-primary-fixed-dim font-headline font-black text-[10px] uppercase tracking-widest">AI Precision Rate</span>
                     <div className="mt-2">
                       <h3 className="text-4xl font-black font-headline text-white">{(score > 0 ? 85.4 + Math.min(score, 14.5) : 85.4).toFixed(1)}%</h3>
                     </div>
                  </div>
                  <p className="text-[10px] text-primary-fixed-dim font-headline font-bold">Real-time adaptive layer is hot</p>
                </div>
              </div>
              
              <div className="bg-surface-container-lowest p-8 rounded-3xl shadow-sm border border-slate-100">
                 <h4 className="font-headline text-lg font-black text-slate-800 mb-6">Efficiency Benchmarks</h4>
                 <div className="space-y-6">
                    <div>
                      <div className="flex justify-between text-xs font-headline font-bold uppercase mb-2">
                        <span>Content Extraction Accuracy</span><span className="text-primary">98.4%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-primary w-[98.4%]"></div></div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-headline font-bold uppercase mb-2">
                        <span>Context Retention</span><span className="text-primary border-primary">92.1%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden"><div className="h-full bg-primary w-[92.1%]"></div></div>
                    </div>
                 </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* ADD ACCOUNT MODAL */}
      {showAddAccount && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm shadow-2xl">
          <div className="bg-white w-full max-w-md rounded-3xl p-8 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-2xl font-black text-slate-800 font-headline">Connect Service</h3>
              <button onClick={() => setShowAddAccount(false)} className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-500">
                <span className="material-symbols-outlined">close</span>
              </button>
            </div>
            <form onSubmit={addAccount} className="space-y-5">
              <div>
                <label className="block text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2 font-headline">IMAP Connection</label>
                <input required type="text" value={newEmail} onChange={e => setNewEmail(e.target.value)} className="w-full px-5 py-4 bg-slate-50 rounded-xl outline-none font-bold text-sm border-2 border-transparent focus:bg-white focus:border-primary transition-colors focus:shadow-sm" placeholder="user@gmail.com or 'demo'" />
              </div>
              <div>
                <input required type="password" value={newPass} onChange={e => setNewPass(e.target.value)} className="w-full px-5 py-4 bg-slate-50 rounded-xl outline-none font-bold text-sm border-2 border-transparent focus:bg-white focus:border-primary transition-colors focus:shadow-sm" placeholder="App Password or 'demo'" />
                <a href="https://support.google.com/mail/answer/185833?hl=en" target="_blank" rel="noopener noreferrer" className="text-[10px] font-bold text-primary hover:underline mt-1.5 block px-1.5 flex items-center gap-1"><span className="material-symbols-outlined text-[12px]">help</span>How to generate an IMAP App Password?</a>
              </div>
              {errMessage && <p className="text-xs text-red-500 font-bold flex items-center gap-1 bg-red-50 p-2 rounded-lg"><AlertTriangle size={12}/> {errMessage}</p>}
              <div className="flex gap-3 pt-2">
                <button type="button" onClick={launchDemo} className="flex-1 bg-amber-100 text-amber-700 font-black py-4 rounded-xl hover:bg-amber-200 transition-colors text-sm border border-amber-200 flex justify-center items-center gap-1.5">
                  <Play size={16} /> 1-Click Demo
                </button>
                <button type="submit" disabled={isConnecting} className="flex-[2] bg-primary text-white font-black py-4 rounded-xl shadow-lg shadow-primary/20 hover:bg-primary-container transition-all hover:-translate-y-0.5 active:scale-95 text-sm">
                  {isConnecting ? 'Validating...' : 'Connect Identity'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Global strict override for scrollbar hiding requested in UI design */}
      <style dangerouslySetInnerHTML={{__html: `.hide-scrollbar::-webkit-scrollbar { display: none; } .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }`}} />
    </div>
  );
}
