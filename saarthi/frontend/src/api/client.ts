import axios from 'axios'
import type {
  UploadResult,
  MapResult,
  RunRequest,
  RunResult,
  StatusResult,
  PortfolioResult,
  LoanResult,
  ModelsResult,
  HealthResult,
} from '../types'

export const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

/** Extract a human-friendly message from an axios/unknown error. */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const data = err.response?.data as
      | { error?: string; message?: string }
      | undefined
    return (
      data?.error ||
      data?.message ||
      err.message ||
      'Network error — is the backend running on :5000?'
    )
  }
  if (err instanceof Error) return err.message
  return 'Unexpected error'
}

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<UploadResult>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function mapColumns(uploadId: string): Promise<MapResult> {
  const { data } = await api.post<MapResult>('/map', { upload_id: uploadId })
  return data
}

export async function runAnalysis(body: RunRequest): Promise<RunResult> {
  const { data } = await api.post<RunResult>('/run', body)
  return data
}

export async function getStatus(jobId: string): Promise<StatusResult> {
  const { data } = await api.get<StatusResult>(
    `/status/${encodeURIComponent(jobId)}`,
  )
  return data
}

export async function getResults(jobId: string): Promise<PortfolioResult> {
  const { data } = await api.get<PortfolioResult>(
    `/results/${encodeURIComponent(jobId)}`,
  )
  return data
}

export async function getLoan(
  jobId: string,
  loanId: string,
): Promise<LoanResult> {
  const { data } = await api.get<LoanResult>(
    `/loan/${encodeURIComponent(jobId)}/${encodeURIComponent(loanId)}`,
  )
  return data
}

export async function getModels(): Promise<ModelsResult> {
  const { data } = await api.get<ModelsResult>('/models')
  return data
}

export async function getHealth(): Promise<HealthResult> {
  const { data } = await api.get<HealthResult>('/health')
  return data
}
