import { useState, useEffect, useCallback } from 'react'
import MapView from './components/MapView'
import Sidebar from './components/Sidebar'
import Header from './components/Header'

const API_BASE = (import.meta.env.VITE_API_URL || '') + '/api'

function App() {
  const [boats, setBoats] = useState([])
  const [alerts, setAlerts] = useState([])
  const [stats, setStats] = useState(null)
  const [selectedBoat, setSelectedBoat] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const [boatsRes, alertsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/boats/`),
        fetch(`${API_BASE}/alerts/active`),
        fetch(`${API_BASE}/dashboard/stats`),
      ])
      
      if (boatsRes.ok) setBoats(await boatsRes.json())
      if (alertsRes.ok) setAlerts(await alertsRes.json())
      if (statsRes.ok) setStats(await statsRes.json())
    } catch (err) {
      console.error('Error fetching data:', err)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [fetchData])

  const sosBoats = boats.filter(b => b.status === 'sos')

  return (
    <div className="app-container">
      <Header stats={stats} sosCount={sosBoats.length} />
      <div className="main-content">
        <div className="map-container">
          <MapView 
            boats={boats} 
            selectedBoat={selectedBoat}
            onSelectBoat={setSelectedBoat}
          />
          {sosBoats.length > 0 && (
            <div className="sos-indicator">
              🚨 {sosBoats.length} SOS نشط
            </div>
          )}
          {stats && (
            <div className="dashboard-overlay">
              <div className="dash-card">
                <div className="dash-card-label">رحلات اليوم</div>
                <div className="dash-card-value">{stats.total_trips_today}</div>
              </div>
              <div className="dash-card">
                <div className="dash-card-label">المسافة الإجمالية</div>
                <div className="dash-card-value">{stats.total_distance_today.toFixed(1)} كم</div>
              </div>
              <div className="dash-card">
                <div className="dash-card-label">تنبيهات نشطة</div>
                <div className="dash-card-value" style={{color: stats.active_alerts > 0 ? '#ef4444' : '#22c55e'}}>
                  {stats.active_alerts}
                </div>
              </div>
            </div>
          )}
        </div>
        <Sidebar 
          boats={boats}
          alerts={alerts}
          selectedBoat={selectedBoat}
          onSelectBoat={setSelectedBoat}
        />
      </div>
    </div>
  )
}

export default App
