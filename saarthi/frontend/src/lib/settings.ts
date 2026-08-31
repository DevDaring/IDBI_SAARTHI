// Lightweight localStorage-backed settings (no context needed).
const CONSENSUS_KEY = 'saarthi.consensusJudge'

export function getConsensus(): boolean {
  try {
    return localStorage.getItem(CONSENSUS_KEY) === 'true'
  } catch {
    return false
  }
}

export function setConsensus(value: boolean): void {
  try {
    localStorage.setItem(CONSENSUS_KEY, value ? 'true' : 'false')
  } catch {
    /* ignore storage failures */
  }
}
