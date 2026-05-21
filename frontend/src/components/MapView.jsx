import { useEffect, useRef } from 'react'
import L from 'leaflet'

const STATUS_COLORS = {
  active: '#22c55e',
  docked: '#0ea5e9',
  sos: '#ef4444',
  offline: '#64748b',
}

function MapView({ boats, selectedBoat, onSelectBoat }) {
  const mapRef = useRef(null)
  const mapInstance = useRef(null)
  const markersRef = useRef({})

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return

    mapInstance.current = L.map(mapRef.current, {
      center: [28.5, -12.0],
      zoom: 6,
      zoomControl: true,
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19,
    }).addTo(mapInstance.current)

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove()
        mapInstance.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (!mapInstance.current) return

    boats.forEach((boat) => {
      const color = STATUS_COLORS[boat.status] || '#64748b'
      const isSOS = boat.status === 'sos'
      const size = isSOS ? 16 : 12

      const icon = L.divIcon({
        className: 'custom-marker',
        html: `
          <div style="
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 0 ${isSOS ? '12px' : '6px'} ${color};
            ${isSOS ? 'animation: pulse 1s infinite;' : ''}
            cursor: pointer;
          "></div>
        `,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      })

      if (markersRef.current[boat.id]) {
        markersRef.current[boat.id].setLatLng([boat.latitude, boat.longitude])
        markersRef.current[boat.id].setIcon(icon)
      } else {
        const marker = L.marker([boat.latitude, boat.longitude], { icon })
          .addTo(mapInstance.current)
          .bindPopup(`
            <div style="direction: rtl; text-align: right; font-family: sans-serif;">
              <strong>${boat.name}</strong><br/>
              <span style="color: ${color}; font-weight: bold;">${getStatusArabic(boat.status)}</span><br/>
              السرعة: ${boat.speed} عقدة<br/>
              البطارية: ${boat.battery_level}%<br/>
              الميناء: ${boat.port || '-'}
            </div>
          `)

        marker.on('click', () => onSelectBoat(boat))
        markersRef.current[boat.id] = marker
      }
    })

    // Remove markers for boats that no longer exist
    Object.keys(markersRef.current).forEach((id) => {
      if (!boats.find((b) => b.id === parseInt(id))) {
        markersRef.current[id].remove()
        delete markersRef.current[id]
      }
    })
  }, [boats, onSelectBoat])

  useEffect(() => {
    if (!mapInstance.current || !selectedBoat) return
    mapInstance.current.flyTo([selectedBoat.latitude, selectedBoat.longitude], 10, {
      duration: 1,
    })
  }, [selectedBoat])

  return <div ref={mapRef} style={{ height: '100%', width: '100%' }} />
}

function getStatusArabic(status) {
  const map = {
    active: 'نشط',
    docked: 'راسي',
    sos: '🚨 استغاثة',
    offline: 'غير متصل',
  }
  return map[status] || status
}

export default MapView
