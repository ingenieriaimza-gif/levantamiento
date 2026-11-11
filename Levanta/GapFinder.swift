import ARKit

/// Busca partes de pisos y paredes que la cámara ya vio (ARKit las detecta como planos) pero que el LiDAR todavía
/// no leyó, y dice dónde están ("Falta el piso a tu izquierda, a 2 metros").
/// Todo se llama en la cola de escaneo.
final class GapFinder {
    private var planes: [UUID: ARPlaneAnchor] = [:]
    private var lastCheck: TimeInterval = 0
    private var announced: [String: TimeInterval] = [:]
    private var hint: (text: String, spoken: String, until: TimeInterval)?

    func reset() {
        planes = [:]; lastCheck = 0; announced = [:]; hint = nil
    }

    func add(_ anchors: [ARAnchor]) {
        for a in anchors { if let p = a as? ARPlaneAnchor { planes[p.identifier] = p } }
    }

    func remove(_ anchors: [ARAnchor]) {
        for a in anchors { planes.removeValue(forKey: a.identifier) }
    }

    /// Aviso vigente (dura unos segundos), o nil.
    func current(_ t: TimeInterval) -> (text: String, spoken: String)? {
        guard let h = hint, t < h.until else { return nil }
        return (h.text, h.spoken)
    }

    private static func inside(_ poly: [SIMD2<Float>], _ p: SIMD2<Float>) -> Bool {
        var c = false
        var j = poly.count - 1
        for i in 0..<poly.count {
            let a = poly[i], b = poly[j]
            if (a.y > p.y) != (b.y > p.y), p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x { c.toggle() }
            j = i
        }
        return c
    }

    /// Cada ~1.5 s revisa las celdas de 20 cm de los planos cercanos y elige la faltante más cercana.
    func check(frame: ARFrame, occupied: (SIMD3<Float>) -> Bool) {
        let t = frame.timestamp
        guard t - lastCheck >= 1.5, !planes.isEmpty else { return }
        lastCheck = t
        let camT = frame.camera.transform
        let cam = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        var best: (dist: Float, p: SIMD3<Float>, label: String, count: Int)?
        let step: Float = 0.2
        for (_, a) in planes {
            let poly = a.geometry.boundaryVertices.map { SIMD2<Float>($0.x, $0.z) }
            guard poly.count >= 3 else { continue }
            let minX = poly.map { $0.x }.min()!, maxX = poly.map { $0.x }.max()!
            let minZ = poly.map { $0.y }.min()!, maxZ = poly.map { $0.y }.max()!
            guard (maxX - minX) * (maxZ - minZ) > 0.2 else { continue }
            let n = SIMD3<Float>(a.transform.columns.1.x, a.transform.columns.1.y, a.transform.columns.1.z)
            let label: String
            switch a.classification {
            case .floor: label = "el piso"
            case .wall: label = "la pared"
            case .ceiling: label = "el techo"
            default: label = a.alignment == .horizontal ? "una superficie plana" : "una pared"
            }
            var missing: (dist: Float, p: SIMD3<Float>)?
            var missCount = 0
            var x = minX + step / 2
            while x < maxX {
                var z = minZ + step / 2
                while z < maxZ {
                    defer { z += step }
                    guard Self.inside(poly, SIMD2<Float>(x, z)) else { continue }
                    let w4 = a.transform * SIMD4<Float>(x, 0, z, 1)
                    let p = SIMD3<Float>(w4.x, w4.y, w4.z)
                    let d = simd_distance(p, cam)
                    guard d > 0.35, d < 3.0 else { continue }
                    // la celda está cubierta si en la mayoría de 9 puntos hay lectura a ±3 cm del plano
                    var hits = 0
                    let tu = simd_normalize(simd_cross(n, abs(n.y) > 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)))
                    let tv = simd_cross(n, tu)
                    for i in -1...1 { for j in -1...1 {
                        let q = p + tu * (Float(i) * 0.06) + tv * (Float(j) * 0.06)
                        var any = false
                        for k in -2...2 where occupied(q + n * (Float(k) * 0.015)) { any = true; break }
                        if any { hits += 1 }
                    } }
                    if hits < 4 {
                        missCount += 1
                        if missing == nil || d < missing!.dist { missing = (d, p) }
                    }
                }
                x += step
            }
            // un hueco real: al menos 4 celdas (0.16 m²) sin lectura
            if missCount >= 4, let m = missing, best == nil || m.dist < best!.dist {
                best = (m.dist, m.p, label, missCount)
            }
        }
        guard let b = best else { hint = nil; return }
        let key = "\(b.label)-\(Int((b.p.x / 0.8).rounded()))-\(Int((b.p.z / 0.8).rounded()))"
        if let at = announced[key], t - at < 40 { return }     // ya se avisó de este hueco hace poco
        announced[key] = t
        let f = SIMD3<Float>(-camT.columns.2.x, 0, -camT.columns.2.z)
        let r = SIMD3<Float>(camT.columns.0.x, 0, camT.columns.0.z)
        let d = b.p - cam
        let az = atan2(simd_dot(d, r), simd_dot(d, f)) * 180 / .pi
        let side: String
        if abs(az) < 35 { side = "al frente" }
        else if abs(az) > 135 { side = "detrás de ti" }
        else if az > 0 { side = "a tu derecha" }
        else { side = "a tu izquierda" }
        let meters = max(1, Int(simd_length(SIMD3<Float>(d.x, 0, d.z)).rounded()))
        let dist = meters == 1 ? "a un metro" : "a \(meters) metros"
        hint = ("Falta \(b.label) \(side) · \(meters) m", "Falta \(b.label) \(side), \(dist)", t + 6)
    }
}
