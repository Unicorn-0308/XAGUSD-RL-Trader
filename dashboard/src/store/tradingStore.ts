import { create } from 'zustand'

export interface Candle {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface Position {
  position_id: string
  side: 'long' | 'short'
  entry_price: number
  volume: number
  unrealized_pnl: number
  open_time: string
}

export interface AgentStatus {
  status: 'stopped' | 'running' | 'paused' | 'error'
  started_at: string | null
  total_steps: number
  total_trades: number
  total_pnl: number
  current_position: Position | null
  last_prediction: number[] | null
  last_action: string | null
  error_message: string | null
}

export interface TrainingStatus {
  status: 'idle' | 'pretraining' | 'online_training' | 'paused' | 'error'
  progress: number
  timesteps: number
  episodes: number
  best_reward: number
  started_at: string | null
  error_message: string | null
}

export interface Metrics {
  total_pnl: number
  win_rate: number
  total_trades: number
  winning_trades: number
  losing_trades: number
  avg_win: number
  avg_loss: number
  profit_factor: number
}

interface TradingStore {
  // Connection
  isConnected: boolean
  setConnected: (connected: boolean) => void
  
  // Agent status
  agentStatus: AgentStatus
  setAgentStatus: (status: Partial<AgentStatus>) => void
  
  // Training status
  trainingStatus: TrainingStatus
  setTrainingStatus: (status: Partial<TrainingStatus>) => void
  
  // Candles
  candles: Candle[]
  addCandle: (candle: Candle) => void
  setCandles: (candles: Candle[]) => void
  
  // Predictions
  predictions: Candle[]
  addPrediction: (prediction: Candle) => void
  
  // Metrics
  metrics: Metrics
  setMetrics: (metrics: Metrics) => void
  
  // Logs
  logs: string[]
  addLog: (log: string) => void
  clearLogs: () => void
}

const initialAgentStatus: AgentStatus = {
  status: 'stopped',
  started_at: null,
  total_steps: 0,
  total_trades: 0,
  total_pnl: 0,
  current_position: null,
  last_prediction: null,
  last_action: null,
  error_message: null,
}

const initialTrainingStatus: TrainingStatus = {
  status: 'idle',
  progress: 0,
  timesteps: 0,
  episodes: 0,
  best_reward: 0,
  started_at: null,
  error_message: null,
}

const initialMetrics: Metrics = {
  total_pnl: 0,
  win_rate: 0,
  total_trades: 0,
  winning_trades: 0,
  losing_trades: 0,
  avg_win: 0,
  avg_loss: 0,
  profit_factor: 0,
}

export const useTradingStore = create<TradingStore>((set) => ({
  // Connection
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),
  
  // Agent status
  agentStatus: initialAgentStatus,
  setAgentStatus: (status) =>
    set((state) => ({
      agentStatus: { ...state.agentStatus, ...status },
    })),
  
  // Training status
  trainingStatus: initialTrainingStatus,
  setTrainingStatus: (status) =>
    set((state) => ({
      trainingStatus: { ...state.trainingStatus, ...status },
    })),
  
  // Candles
  candles: [],
  addCandle: (candle) =>
    set((state) => ({
      candles: [...state.candles.slice(-499), candle],
    })),
  setCandles: (candles) => set({ candles }),
  
  // Predictions
  predictions: [],
  addPrediction: (prediction) =>
    set((state) => ({
      predictions: [...state.predictions.slice(-99), prediction],
    })),
  
  // Metrics
  metrics: initialMetrics,
  setMetrics: (metrics) => set({ metrics }),
  
  // Logs
  logs: [],
  addLog: (log) =>
    set((state) => ({
      logs: [...state.logs.slice(-99), `[${new Date().toLocaleTimeString()}] ${log}`],
    })),
  clearLogs: () => set({ logs: [] }),
}))

