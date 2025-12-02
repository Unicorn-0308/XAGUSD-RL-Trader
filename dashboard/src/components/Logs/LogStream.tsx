import { useEffect, useRef } from 'react'
import { Terminal, Trash2 } from 'lucide-react'
import { useTradingStore } from '../../store/tradingStore'

export function LogStream() {
  const { logs, clearLogs } = useTradingStore()
  const containerRef = useRef<HTMLDivElement>(null)
  
  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])
  
  return (
    <div 
      className="card h-full flex flex-col"
      style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Terminal size={16} style={{ color: 'var(--text-secondary)' }} />
          <h3 
            className="text-sm font-semibold"
            style={{ color: 'var(--text-primary)' }}
          >
            Activity Log
          </h3>
        </div>
        <button
          onClick={clearLogs}
          className="p-1 rounded hover:bg-opacity-10 hover:bg-white"
          style={{ color: 'var(--text-muted)' }}
        >
          <Trash2 size={14} />
        </button>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto font-mono text-xs space-y-1"
        style={{ 
          backgroundColor: 'var(--bg-primary)',
          padding: '8px',
          borderRadius: '4px',
          maxHeight: '200px',
        }}
      >
        {logs.length === 0 ? (
          <p style={{ color: 'var(--text-muted)' }}>No logs yet...</p>
        ) : (
          logs.map((log, index) => (
            <p 
              key={index}
              style={{ color: 'var(--text-secondary)' }}
            >
              {log}
            </p>
          ))
        )}
      </div>
    </div>
  )
}

