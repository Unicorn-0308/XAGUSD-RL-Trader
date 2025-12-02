import { useState, useEffect } from 'react'
import { RefreshCw, Download } from 'lucide-react'
import { dataApi } from '../api/client'
import { cn, formatCurrency, formatTimestamp } from '../lib/utils'

interface Trade {
  position_id: string
  side: string
  entry_price: number
  exit_price: number | null
  volume: number
  open_time: string
  close_time: string | null
  realized_pnl: number | null
  close_reason: string | null
}

interface TradeSummary {
  total_trades: number
  winning_trades: number
  losing_trades: number
  total_pnl: number
  win_rate: number
}

export function History() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [summary, setSummary] = useState<TradeSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(0)
  const [total, setTotal] = useState(0)
  const pageSize = 20
  
  const fetchTrades = async () => {
    setLoading(true)
    const { data } = await dataApi.getTrades(pageSize, page * pageSize)
    if (data) {
      setTrades(data.trades)
      setTotal(data.total)
      setSummary(data.summary)
    }
    setLoading(false)
  }
  
  useEffect(() => {
    fetchTrades()
  }, [page])
  
  const exportCsv = () => {
    if (trades.length === 0) return
    
    const headers = ['ID', 'Side', 'Entry Price', 'Exit Price', 'Volume', 'Open Time', 'Close Time', 'PnL', 'Close Reason']
    const rows = trades.map(t => [
      t.position_id,
      t.side,
      t.entry_price,
      t.exit_price ?? '',
      t.volume,
      t.open_time,
      t.close_time ?? '',
      t.realized_pnl ?? '',
      t.close_reason ?? '',
    ])
    
    const csv = [headers, ...rows].map(row => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `trades_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
  }
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <h2 
          className="text-2xl font-bold font-display"
          style={{ color: 'var(--text-primary)' }}
        >
          Trade History
        </h2>
        
        <div className="flex gap-2">
          <button
            onClick={fetchTrades}
            disabled={loading}
            className="btn-secondary flex items-center gap-2"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={exportCsv}
            disabled={trades.length === 0}
            className="btn-secondary flex items-center gap-2"
          >
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </div>
      
      {/* Summary */}
      {summary && (
        <div 
          className="card"
          style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
        >
          <div className="grid grid-cols-5 gap-4">
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Total Trades
              </p>
              <p 
                className="text-xl font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {summary.total_trades}
              </p>
            </div>
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Win Rate
              </p>
              <p 
                className="text-xl font-semibold"
                style={{ color: 'var(--text-primary)' }}
              >
                {(summary.win_rate * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Wins
              </p>
              <p className="text-xl font-semibold number-positive">
                {summary.winning_trades}
              </p>
            </div>
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Losses
              </p>
              <p className="text-xl font-semibold number-negative">
                {summary.losing_trades}
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
                  "text-xl font-semibold",
                  summary.total_pnl >= 0 ? "number-positive" : "number-negative"
                )}
              >
                {formatCurrency(summary.total_pnl)}
              </p>
            </div>
          </div>
        </div>
      )}
      
      {/* Trades table */}
      <div 
        className="card"
        style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                <th className="text-left py-3 px-2" style={{ color: 'var(--text-secondary)' }}>ID</th>
                <th className="text-left py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Side</th>
                <th className="text-right py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Entry</th>
                <th className="text-right py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Exit</th>
                <th className="text-right py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Volume</th>
                <th className="text-left py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Opened</th>
                <th className="text-left py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Closed</th>
                <th className="text-right py-3 px-2" style={{ color: 'var(--text-secondary)' }}>PnL</th>
                <th className="text-left py-3 px-2" style={{ color: 'var(--text-secondary)' }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 ? (
                <tr>
                  <td 
                    colSpan={9} 
                    className="text-center py-8"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    No trades found
                  </td>
                </tr>
              ) : (
                trades.map((trade) => (
                  <tr 
                    key={trade.position_id}
                    className="hover:bg-opacity-5 hover:bg-white"
                    style={{ borderBottom: '1px solid var(--border-color)' }}
                  >
                    <td className="py-2 px-2 font-mono text-xs" style={{ color: 'var(--text-primary)' }}>
                      {trade.position_id}
                    </td>
                    <td className="py-2 px-2">
                      <span 
                        className={cn(
                          "px-2 py-0.5 rounded text-xs font-medium uppercase",
                          trade.side === 'long' 
                            ? "bg-profit/20 text-profit" 
                            : "bg-loss/20 text-loss"
                        )}
                      >
                        {trade.side}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right font-mono" style={{ color: 'var(--text-primary)' }}>
                      ${trade.entry_price.toFixed(3)}
                    </td>
                    <td className="py-2 px-2 text-right font-mono" style={{ color: 'var(--text-primary)' }}>
                      {trade.exit_price ? `$${trade.exit_price.toFixed(3)}` : '-'}
                    </td>
                    <td className="py-2 px-2 text-right" style={{ color: 'var(--text-secondary)' }}>
                      {trade.volume}
                    </td>
                    <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {formatTimestamp(trade.open_time)}
                    </td>
                    <td className="py-2 px-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {trade.close_time ? formatTimestamp(trade.close_time) : '-'}
                    </td>
                    <td 
                      className={cn(
                        "py-2 px-2 text-right font-semibold",
                        trade.realized_pnl !== null && trade.realized_pnl >= 0 
                          ? "number-positive" 
                          : "number-negative"
                      )}
                    >
                      {trade.realized_pnl !== null ? formatCurrency(trade.realized_pnl) : '-'}
                    </td>
                    <td className="py-2 px-2 text-xs capitalize" style={{ color: 'var(--text-secondary)' }}>
                      {trade.close_reason?.replace('_', ' ') || '-'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {/* Pagination */}
        {total > pageSize && (
          <div 
            className="flex items-center justify-between mt-4 pt-4 border-t"
            style={{ borderColor: 'var(--border-color)' }}
          >
            <p 
              className="text-sm"
              style={{ color: 'var(--text-secondary)' }}
            >
              Showing {page * pageSize + 1} - {Math.min((page + 1) * pageSize, total)} of {total}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn-secondary text-sm"
              >
                Previous
              </button>
              <button
                onClick={() => setPage(p => p + 1)}
                disabled={(page + 1) * pageSize >= total}
                className="btn-secondary text-sm"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

