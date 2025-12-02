import { useEffect, useRef } from 'react'
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts'
import { useThemeStore } from '../../store/themeStore'
import { useTradingStore, Candle } from '../../store/tradingStore'

interface PriceChartProps {
  height?: number
}

export function PriceChart({ height = 400 }: PriceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  
  const { theme } = useThemeStore()
  const { candles } = useTradingStore()
  
  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return
    
    const isDark = theme === 'dark'
    
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: isDark ? '#0a0e14' : '#f8fafc' },
        textColor: isDark ? '#8b949e' : '#475569',
      },
      grid: {
        vertLines: { color: isDark ? '#1c2430' : '#e2e8f0' },
        horzLines: { color: isDark ? '#1c2430' : '#e2e8f0' },
      },
      width: chartContainerRef.current.clientWidth,
      height,
      crosshair: {
        mode: 1,
        vertLine: {
          color: isDark ? '#3b82f6' : '#2563eb',
          width: 1,
          style: 2,
        },
        horzLine: {
          color: isDark ? '#3b82f6' : '#2563eb',
          width: 1,
          style: 2,
        },
      },
      timeScale: {
        borderColor: isDark ? '#2d3748' : '#e2e8f0',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: isDark ? '#2d3748' : '#e2e8f0',
      },
    })
    
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#10b981',
      wickDownColor: '#ef4444',
      wickUpColor: '#10b981',
    })
    
    chartRef.current = chart
    candleSeriesRef.current = candleSeries
    
    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    
    window.addEventListener('resize', handleResize)
    
    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [theme, height])
  
  // Update data
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return
    
    const chartData: CandlestickData[] = candles.map((candle: Candle) => ({
      time: (new Date(candle.timestamp).getTime() / 1000) as Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }))
    
    candleSeriesRef.current.setData(chartData)
    
    // Scroll to latest
    chartRef.current?.timeScale().scrollToRealTime()
  }, [candles])
  
  return (
    <div 
      className="card"
      style={{ backgroundColor: 'var(--bg-card)', borderColor: 'var(--border-color)' }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 
          className="text-lg font-semibold"
          style={{ color: 'var(--text-primary)' }}
        >
          XAGUSD Price
        </h3>
        {candles.length > 0 && (
          <span 
            className="text-sm"
            style={{ color: 'var(--text-secondary)' }}
          >
            Last: ${candles[candles.length - 1]?.close.toFixed(3)}
          </span>
        )}
      </div>
      <div ref={chartContainerRef} />
    </div>
  )
}

