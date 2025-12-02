import { ArrowUpCircle, ArrowDownCircle, MinusCircle } from 'lucide-react'
import { useTradingStore } from '../../store/tradingStore'
import { cn, formatCurrency, formatTimestamp } from '../../lib/utils'

export function PositionCard() {
  const { agentStatus } = useTradingStore()
  const position = agentStatus.current_position
  
  return (
    <div 
      className="card"
      style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
    >
      <h3 
        className="text-lg font-semibold mb-4"
        style={{ color: 'var(--text-primary)' }}
      >
        Current Position
      </h3>
      
      {position ? (
        <div className="space-y-4">
          {/* Position type */}
          <div className="flex items-center gap-3">
            {position.side === 'long' ? (
              <ArrowUpCircle size={24} className="text-profit" />
            ) : (
              <ArrowDownCircle size={24} className="text-loss" />
            )}
            <div>
              <p 
                className="text-lg font-semibold capitalize"
                style={{ color: 'var(--text-primary)' }}
              >
                {position.side}
              </p>
              <p 
                className="text-sm"
                style={{ color: 'var(--text-secondary)' }}
              >
                {position.volume} lots
              </p>
            </div>
          </div>
          
          {/* Details */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Entry Price
              </p>
              <p 
                className="text-lg font-mono"
                style={{ color: 'var(--text-primary)' }}
              >
                ${position.entry_price.toFixed(3)}
              </p>
            </div>
            <div>
              <p 
                className="text-xs uppercase"
                style={{ color: 'var(--text-muted)' }}
              >
                Unrealized PnL
              </p>
              <p 
                className={cn(
                  "text-lg font-semibold",
                  position.unrealized_pnl >= 0 ? "number-positive" : "number-negative"
                )}
              >
                {formatCurrency(position.unrealized_pnl)}
              </p>
            </div>
          </div>
          
          {/* Open time */}
          <div 
            className="text-xs pt-2 border-t"
            style={{ 
              color: 'var(--text-muted)',
              borderColor: 'var(--border-color)',
            }}
          >
            Opened: {formatTimestamp(position.open_time)}
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8">
          <MinusCircle 
            size={32} 
            style={{ color: 'var(--text-muted)' }}
          />
          <p 
            className="mt-2"
            style={{ color: 'var(--text-secondary)' }}
          >
            No open position
          </p>
        </div>
      )}
    </div>
  )
}

