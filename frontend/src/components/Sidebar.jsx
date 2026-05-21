import { useState } from 'react'

function Sidebar({ boats, alerts, selectedBoat, onSelectBoat }) {
  const [activeTab, setActiveTab] = useState('boats')

  return (
    <aside className="sidebar">
      <div className="sidebar-tabs">
        <button 
          className={`sidebar-tab ${activeTab === 'boats' ? 'active' : ''}`}
          onClick={() => setActiveTab('boats')}
        >
          القوارب ({boats.length})
        </button>
        <button 
          className={`sidebar-tab ${activeTab === 'alerts' ? 'active' : ''}`}
          onClick={() => setActiveTab('alerts')}
        >
          التنبيهات ({alerts.length})
        </button>
      </div>
      <div className="sidebar-content">
        {activeTab === 'boats' && (
          <BoatList boats={boats} selectedBoat={selectedBoat} onSelectBoat={onSelectBoat} />
        )}
        {activeTab === 'alerts' && (
          <AlertList alerts={alerts} />
        )}
      </div>
    </aside>
  )
}

function BoatList({ boats, selectedBoat, onSelectBoat }) {
  const sortedBoats = [...boats].sort((a, b) => {
    const priority = { sos: 0, active: 1, docked: 2, offline: 3 }
    return (priority[a.status] ?? 4) - (priority[b.status] ?? 4)
  })

  return (
    <div>
      {sortedBoats.map((boat) => (
        <div
          key={boat.id}
          className={`boat-card ${selectedBoat?.id === boat.id ? 'selected' : ''}`}
          onClick={() => onSelectBoat(boat)}
        >
          <div className="boat-card-header">
            <span className="boat-name">{boat.name}</span>
            <span className={`boat-status ${boat.status}`}>
              {getStatusArabic(boat.status)}
            </span>
          </div>
          <div className="boat-info">
            <span>⚓ {boat.port || '-'}</span>
            <span>🔋 {boat.battery_level}%</span>
            <span>💨 {boat.speed} عقدة</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function AlertList({ alerts }) {
  if (alerts.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
        ✅ لا توجد تنبيهات نشطة
      </div>
    )
  }

  return (
    <div>
      {alerts.map((alert) => (
        <div key={alert.id} className={`alert-card ${alert.alert_type}`}>
          <div className="alert-header">
            <span className="alert-type">
              {getAlertIcon(alert.alert_type)} {getAlertTypeArabic(alert.alert_type)}
            </span>
            <span className="alert-time">
              {new Date(alert.created_at).toLocaleTimeString('ar-MA')}
            </span>
          </div>
          <div className="alert-message">{alert.message}</div>
        </div>
      ))}
    </div>
  )
}

function getStatusArabic(status) {
  const map = { active: 'نشط', docked: 'راسي', sos: 'SOS', offline: 'غير متصل' }
  return map[status] || status
}

function getAlertTypeArabic(type) {
  const map = {
    sos: 'استغاثة',
    weather: 'طقس',
    speed: 'سرعة',
    boundary: 'حدود',
    low_battery: 'بطارية',
  }
  return map[type] || type
}

function getAlertIcon(type) {
  const map = { sos: '🚨', weather: '⛈️', speed: '⚡', boundary: '🚧', low_battery: '🔋' }
  return map[type] || '⚠️'
}

export default Sidebar
