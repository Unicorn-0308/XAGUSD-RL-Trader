import { useState } from 'react'
import { Play, Square, Pause, PlayCircle } from 'lucide-react'
import { useTradingStore } from '../../store/tradingStore'
import { agentApi } from '../../api/client'
import { cn, formatCurrency } from '../../lib/utils'

export function AgentControl() {
  const { agentStatus, setAgentStatus, addLog } = useTradingStore()
  const [loading, setLoading] = useState(false)
  
  const handleStart = async () => {
    setLoading(true)
    const { data, error } = await agentApi.start()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setAgentStatus({ status: 'running' })
      addLog(data?.message || 'Agent started')
    }
    setLoading(false)
  }
  
  const handleStop = async () => {
    setLoading(true)
    const { data, error } = await agentApi.stop()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setAgentStatus({ status: 'stopped' })
      addLog(data?.message || 'Agent stopped')
    }
    setLoading(false)
  }
  
  const handlePause = async () => {
    setLoading(true)
    const { data, error } = await agentApi.pause()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setAgentStatus({ status: 'paused' })
      addLog(data?.message || 'Agent paused')
    }
    setLoading(false)
  }
  
  const handleResume = async () => {
    setLoading(true)
    const { data, error } = await agentApi.resume()
    if (error) {
      addLog(`Error: ${error}`)
    } else {
      setAgentStatus({ status: 'running' })
      addLog(data?.message || 'Agent resumed')
    }
    setLoading(false)
  }
  
  return (
    <div 
      className="card"
      style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
    >
      <h3 
        className="text-lg font-semibold mb-4"
        style={{ color: 'var(--text-primary)' }}
      >
        Agent Control
      </h3>
      
      {/* Status */}
      <div className="flex items-center gap-2 mb-4">
        <span 
          className={cn(
            "status-dot",
            agentStatus.status === 'running' && "running",
            agentStatus.status === 'stopped' && "stopped",
            agentStatus.status === 'paused' && "paused",
            agentStatus.status === 'error' && "error",
          )}
        />
        <span 
          className="capitalize font-medium"
          style={{ color: 'var(--text-primary)' }}
        >
          {agentStatus.status}
        </span>
      </div>
      
      {/* Error message */}
      {agentStatus.error_message && (
        <div 
          className="mb-4 p-2 rounded text-sm"
          style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--loss)' }}
        >
          {agentStatus.error_message}
        </div>
      )}
      
      {/* Controls */}
      <div className="flex gap-2 mb-4">
        {agentStatus.status === 'stopped' && (
          <button
            onClick={handleStart}
            disabled={loading}
            className="btn-success flex items-center gap-2"
          >
            <Play size={16} />
            Start
          </button>
        )}
        
        {agentStatus.status === 'running' && (
          <>
            <button
              onClick={handlePause}
              disabled={loading}
              className="btn-secondary flex items-center gap-2"
            >
              <Pause size={16} />
              Pause
            </button>
            <button
              onClick={handleStop}
              disabled={loading}
              className="btn-danger flex items-center gap-2"
            >
              <Square size={16} />
              Stop
            </button>
          </>
        )}
        
        {agentStatus.status === 'paused' && (
          <>
            <button
              onClick={handleResume}
              disabled={loading}
              className="btn-success flex items-center gap-2"
            >
              <PlayCircle size={16} />
              Resume
            </button>
            <button
              onClick={handleStop}
              disabled={loading}
              className="btn-danger flex items-center gap-2"
            >
              <Square size={16} />
              Stop
            </button>
          </>
        )}
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p 
            className="text-xs uppercase"
            style={{ color: 'var(--text-muted)' }}
          >
            Total Steps
          </p>
          <p 
            className="text-lg font-semibold"
            style={{ color: 'var(--text-primary)' }}
          >
            {agentStatus.total_steps.toLocaleString()}
          </p>
        </div>
        <div>
          <p 
            className="text-xs uppercase"
            style={{ color: 'var(--text-muted)' }}
          >
            Total Trades
          </p>
          <p 
            className="text-lg font-semibold"
            style={{ color: 'var(--text-primary)' }}
          >
            {agentStatus.total_trades}
          </p>
        </div>
        <div>
          <p 
            className="text-xs uppercase"
            style={{ color: 'var(--text-muted)' }}
          >
            Total PnL
          </p>
          <p 
            className={cn(
              "text-lg font-semibold",
              agentStatus.total_pnl >= 0 ? "number-positive" : "number-negative"
            )}
          >
            {formatCurrency(agentStatus.total_pnl)}
          </p>
        </div>
        <div>
          <p 
            className="text-xs uppercase"
            style={{ color: 'var(--text-muted)' }}
          >
            Last Action
          </p>
          <p 
            className="text-lg font-semibold"
            style={{ color: 'var(--text-primary)' }}
          >
            {agentStatus.last_action || '-'}
          </p>
        </div>
      </div>
    </div>
  )
}

