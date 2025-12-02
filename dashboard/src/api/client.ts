const API_BASE = '/api'

interface ApiResponse<T> {
  data?: T
  error?: string
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })
    
    const data = await response.json()
    
    if (!response.ok) {
      return { error: data.detail || 'Request failed' }
    }
    
    return { data }
  } catch (error) {
    return { error: error instanceof Error ? error.message : 'Network error' }
  }
}

// Agent API
export const agentApi = {
  start: (loadCheckpoint = true, checkpointPath?: string) =>
    request<{ success: boolean; message: string }>('/agent/start', {
      method: 'POST',
      body: JSON.stringify({ load_checkpoint: loadCheckpoint, checkpoint_path: checkpointPath }),
    }),
  
  stop: () =>
    request<{ success: boolean; message: string }>('/agent/stop', {
      method: 'POST',
    }),
  
  pause: () =>
    request<{ success: boolean; message: string }>('/agent/pause', {
      method: 'POST',
    }),
  
  resume: () =>
    request<{ success: boolean; message: string }>('/agent/resume', {
      method: 'POST',
    }),
  
  getStatus: () =>
    request<any>('/agent/status'),
}

// Training API
export const trainingApi = {
  startPretrain: (csvPath: string, totalTimesteps = 1000000) =>
    request<{ success: boolean; message: string }>('/training/pretrain', {
      method: 'POST',
      body: JSON.stringify({ csv_path: csvPath, total_timesteps: totalTimesteps }),
    }),
  
  stop: () =>
    request<{ success: boolean; message: string }>('/training/stop', {
      method: 'POST',
    }),
  
  pause: () =>
    request<{ success: boolean; message: string }>('/training/pause', {
      method: 'POST',
    }),
  
  resume: () =>
    request<{ success: boolean; message: string }>('/training/resume', {
      method: 'POST',
    }),
  
  getProgress: () =>
    request<any>('/training/progress'),
}

// Data API
export const dataApi = {
  getCandles: (limit = 100, offset = 0) =>
    request<{ candles: any[]; total: number }>(`/data/candles?limit=${limit}&offset=${offset}`),
  
  getTrades: (limit = 100, offset = 0) =>
    request<{ trades: any[]; total: number; summary: any }>(`/data/trades?limit=${limit}&offset=${offset}`),
  
  getMetrics: () =>
    request<any>('/data/metrics'),
  
  getCsvFiles: () =>
    request<{ files: any[]; directory: string }>('/data/csv-files'),
}

// Checkpoint API
export const checkpointApi = {
  list: () =>
    request<{ checkpoints: any[]; best_checkpoint: string | null; latest_checkpoint: string | null }>('/checkpoint/list'),
  
  load: (path: string, loadOptimizer = true) =>
    request<{ success: boolean; message: string; path: string }>('/checkpoint/load', {
      method: 'POST',
      body: JSON.stringify({ path, load_optimizer: loadOptimizer }),
    }),
  
  save: (name?: string) =>
    request<{ success: boolean; message: string; path: string }>('/checkpoint/save', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),
  
  loadBest: () =>
    request<{ success: boolean; message: string; path: string }>('/checkpoint/load-best', {
      method: 'POST',
    }),
  
  loadLatest: () =>
    request<{ success: boolean; message: string; path: string }>('/checkpoint/load-latest', {
      method: 'POST',
    }),
}

