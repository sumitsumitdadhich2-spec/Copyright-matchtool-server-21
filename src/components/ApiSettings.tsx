import React, { useState, useEffect } from 'react';
import {
  KeyRound, ChevronDown, ChevronUp, CheckCircle2, AlertCircle,
  RefreshCw, Eye, EyeOff, Save, Zap,
} from 'lucide-react';

interface SettingsState {
  geminiConfigured: boolean;
  geminiKeyMasked: string;
  gpuEmbedServiceUrl: string;
}

export default function ApiSettings() {
  const [open, setOpen] = useState(false);
  const [loaded, setLoaded] = useState<SettingsState | null>(null);

  const [geminiKey, setGeminiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [gpuUrl, setGpuUrl] = useState('');

  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const [testMsg, setTestMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const refresh = async () => {
    try {
      const res = await fetch('/api/settings');
      if (!res.ok) return;
      const data: SettingsState = await res.json();
      setLoaded(data);
      setGpuUrl(data.gpuEmbedServiceUrl || '');
    } catch { /* server may be offline; panel stays editable */ }
  };

  useEffect(() => { refresh(); }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg(null);
    setTestMsg(null);
    try {
      const body: Record<string, string> = { gpuEmbedServiceUrl: gpuUrl };
      // Only send the key if the user typed something — empty field means "keep existing"
      if (geminiKey.trim()) body.geminiApiKey = geminiKey.trim();
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);
      setLoaded(data);
      setGeminiKey('');
      setSaveMsg({ ok: true, text: 'Saved — applies to all new match jobs immediately.' });
    } catch (e: any) {
      setSaveMsg({ ok: false, text: e?.message || 'Failed to save' });
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestMsg(null);
    try {
      const res = await fetch('/api/settings/test-gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geminiKey.trim() ? { geminiApiKey: geminiKey.trim() } : {}),
      });
      const data = await res.json();
      setTestMsg(data.ok
        ? { ok: true, text: 'Gemini key is valid and working.' }
        : { ok: false, text: data.error || 'Key test failed' });
    } catch (e: any) {
      setTestMsg({ ok: false, text: e?.message || 'Network error' });
    } finally {
      setTesting(false);
    }
  };

  const geminiOn = loaded?.geminiConfigured ?? false;
  const gpuOn = !!(loaded?.gpuEmbedServiceUrl);

  return (
    <section className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen(s => !s)}
        className="w-full flex items-center justify-between px-6 py-4 hover:bg-slate-800/50 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-3">
          <div className="bg-amber-500/10 p-2 rounded-lg border border-amber-500/20">
            <KeyRound className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-left">
            <h2 className="text-base font-semibold text-white">API &amp; GPU Connection</h2>
            <p className="text-xs text-slate-500 flex items-center gap-3">
              <span className={`flex items-center gap-1 ${geminiOn ? 'text-green-400' : 'text-slate-500'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${geminiOn ? 'bg-green-400' : 'bg-slate-600'}`} />
                Gemini {geminiOn ? `active (${loaded?.geminiKeyMasked})` : 'not set'}
              </span>
              <span className={`flex items-center gap-1 ${gpuOn ? 'text-green-400' : 'text-slate-500'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${gpuOn ? 'bg-green-400' : 'bg-slate-600'}`} />
                GPU service {gpuOn ? 'set' : 'not set'}
              </span>
            </p>
          </div>
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
      </button>

      {open && (
        <div className="px-6 pb-6 pt-2 border-t border-slate-800 space-y-5">
          {/* Gemini key */}
          <div className="space-y-1.5">
            <label htmlFor="gemini-key" className="text-sm font-medium text-slate-300">
              Gemini API Key
            </label>
            <p className="text-xs text-slate-500 leading-relaxed">
              Used for VLM segment verification. Get a free key at{' '}
              <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer" className="text-amber-400 hover:underline">
                aistudio.google.com/apikey
              </a>. {geminiOn && 'A key is already saved — leave blank to keep it.'}
            </p>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  id="gemini-key"
                  type={showKey ? 'text' : 'password'}
                  value={geminiKey}
                  onChange={e => setGeminiKey(e.target.value)}
                  placeholder={geminiOn ? `Saved: ${loaded?.geminiKeyMasked}` : 'AIza…'}
                  autoComplete="off"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 pr-10 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-amber-500/60"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(s => !s)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 cursor-pointer"
                  aria-label={showKey ? 'Hide key' : 'Show key'}
                >
                  {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <button
                onClick={handleTest}
                disabled={testing || (!geminiKey.trim() && !geminiOn)}
                className="flex items-center gap-1.5 px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 rounded-lg text-sm font-medium transition cursor-pointer whitespace-nowrap"
              >
                {testing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Zap className="w-3.5 h-3.5" />}
                Test
              </button>
            </div>
            {testMsg && (
              <p className={`flex items-center gap-1.5 text-xs ${testMsg.ok ? 'text-green-400' : 'text-red-400'}`}>
                {testMsg.ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                {testMsg.text}
              </p>
            )}
          </div>

          {/* GPU embed service URL */}
          <div className="space-y-1.5">
            <label htmlFor="gpu-url" className="text-sm font-medium text-slate-300">
              GPU Embedding Service URL (ngrok)
            </label>
            <p className="text-xs text-slate-500 leading-relaxed">
              Public ngrok URL from your Colab notebook. Free ngrok URLs change on every
              Colab restart — update this field each time instead of restarting the server.
            </p>
            <input
              id="gpu-url"
              type="url"
              value={gpuUrl}
              onChange={e => setGpuUrl(e.target.value)}
              placeholder="https://xxxx.ngrok-free.app"
              autoComplete="off"
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-amber-500/60"
            />
          </div>

          {/* Save */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-medium transition cursor-pointer"
            >
              {saving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              Save Settings
            </button>
            {saveMsg && (
              <p className={`flex items-center gap-1.5 text-xs ${saveMsg.ok ? 'text-green-400' : 'text-red-400'}`}>
                {saveMsg.ok ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                {saveMsg.text}
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
