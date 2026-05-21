function Header({ stats, sosCount }) {
  return (
    <header className="header">
      <div className="header-logo">
        <h1>🚢 SmartSea Morocco</h1>
        <span>نظام تتبع قوارب الصيد</span>
      </div>
      {stats && (
        <div className="header-stats">
          <div className="stat-badge active">
            <span>نشط</span>
            <span className="count">{stats.active_boats}</span>
          </div>
          <div className="stat-badge docked">
            <span>راسي</span>
            <span className="count">{stats.docked_boats}</span>
          </div>
          <div className="stat-badge sos">
            <span>SOS</span>
            <span className="count">{sosCount}</span>
          </div>
        </div>
      )}
    </header>
  )
}

export default Header
