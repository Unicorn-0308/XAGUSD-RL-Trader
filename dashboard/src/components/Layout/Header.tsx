import { Moon, Sun, Wifi, WifiOff } from 'lucide-react'
import { useThemeStore } from '../../store/themeStore'
import { useTradingStore } from '../../store/tradingStore'
import { cn } from '../../lib/utils'

export function Header() {
  const { theme, toggleTheme } = useThemeStore()
  const { isConnected, agentStatus } = useTradingStore()
  
  return (
    <header 
      className="h-14 border-b flex items-center justify-between px-6"
      style={{ 
        backgroundColor: 'var(--bg-secondary)', 
        borderColor: 'var(--border-color)' 
      }}
    >
      <div className="flex items-center gap-4">
        <h1 
          className="text-xl font-bold font-display"
          style={{ color: 'var(--text-primary)' }}
        >
          XAGUSD <span className="text-accent-blue">RL Trader</span>
        </h1>
        
        {/* Status indicator */}
        <div className="flex items-center gap-2">
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
            className="text-sm capitalize"
            style={{ color: 'var(--text-secondary)' }}
          >
            {agentStatus.status}
          </span>
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        {/* Connection status */}
        <div 
          className="flex items-center gap-2 text-sm"
          style={{ color: isConnected ? 'var(--profit)' : 'var(--loss)' }}
        >
          {isConnected ? (
            <>
              <Wifi size={16} />
              <span>Connected</span>
            </>
          ) : (
            <>
              <WifiOff size={16} />
              <span>Disconnected</span>
            </>
          )}
        </div>
        
        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-md hover:bg-opacity-20 hover:bg-white transition-colors"
          style={{ color: 'var(--text-secondary)' }}
        >
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </button>
      </div>
    </header>
  )
}

