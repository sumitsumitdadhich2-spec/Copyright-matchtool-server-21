// Persistent session storage: saves jobIds to localStorage so they survive
// page refreshes and accidental navigation.

export interface CachedJob {
  jobId: string;
  fileName: string;
  fileSize: number;
  totalFrames: number;
  savedAt: number;
}

const STORAGE_KEY = 'nexus_sessions_v1';

interface Sessions {
  reference?: CachedJob;
  target?: CachedJob;
  /** Active/last background /api/match job id, so it survives reloads and
   *  tab backgrounding the same way reference/target fingerprint jobs do. */
  match?: string;
}

function load(): Sessions {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

function save(s: Sessions) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch { /* ignore quota errors */ }
}

export function saveJobSession(role: 'reference' | 'target', job: CachedJob) {
  const s = load();
  s[role] = job;
  save(s);
}

export function getJobSession(role: 'reference' | 'target'): CachedJob | null {
  return load()[role] ?? null;
}

export function clearJobSession(role: 'reference' | 'target') {
  const s = load();
  delete s[role];
  save(s);
}

export function getAllSessions(): Sessions {
  return load();
}

export function saveMatchJobId(matchJobId: string) {
  const s = load();
  s.match = matchJobId;
  save(s);
}

export function getMatchJobId(): string | null {
  return load().match ?? null;
}

export function clearMatchJobId() {
  const s = load();
  delete s.match;
  save(s);
}
