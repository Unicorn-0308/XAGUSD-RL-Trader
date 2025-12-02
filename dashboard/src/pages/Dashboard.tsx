import { PriceChart } from '../components/Charts/PriceChart'
import { AgentControl } from '../components/Controls/AgentControl'
import { MetricsPanel } from '../components/Metrics/MetricsPanel'
import { PositionCard } from '../components/Position/PositionCard'
import { LogStream } from '../components/Logs/LogStream'

export function Dashboard() {
  return (
    <div className="space-y-6 animate-fade-in">
      <h2 
        className="text-2xl font-bold font-display"
        style={{ color: 'var(--text-primary)' }}
      >
        Dashboard
      </h2>
      
      {/* Main grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Chart - spans 8 columns */}
        <div className="col-span-8">
          <PriceChart height={400} />
        </div>
        
        {/* Right sidebar - spans 4 columns */}
        <div className="col-span-4 space-y-6">
          <AgentControl />
          <PositionCard />
        </div>
      </div>
      
      {/* Bottom row */}
      <div className="grid grid-cols-12 gap-6">
        {/* Metrics - spans 8 columns */}
        <div className="col-span-8">
          <MetricsPanel />
        </div>
        
        {/* Logs - spans 4 columns */}
        <div className="col-span-4">
          <LogStream />
        </div>
      </div>
    </div>
  )
}

