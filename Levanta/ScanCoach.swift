import ARKit
import AVFoundation
import UIKit
import SwiftUI

/// Lo que se midió en un cuadro (lo llena `integrate`).
struct FrameStats {
    var total = 0                 // puntos leídos en la vista
    var good = 0                  // de esos, los que ya están en verde (buena o excelente)
    var depths: [Float] = []      // muestra de distancias de la vista
    var zoneIDs: [String] = []
    var zoneHits: [Int] = []      // lecturas que cayeron dentro de cada pieza
    var zoneConfirmed: [Int] = [] // de esas, en puntos de 2 mm ya confirmados
    var zoneDepth: [Float] = []   // suma de distancias dentro de cada pieza
}

/// Indicación para el usuario mientras escanea.
struct ScanGuide: Equatable {
    enum Kind { case closer, farther, ok, ready, gap }
    var kind: Kind
    var text: String
    var spoken: String = ""
}

/// Indica acércate / aléjate y avisa cuando una pieza o una parte del espacio ya quedó lista.
/// Todo se llama en la cola de escaneo; `onGuide` se llama en el hilo principal.
final class ScanCoach {
    /// Distancias ideales: piezas enfocadas 30–60 cm (el LiDAR no lee a menos de ~25 cm);
    /// el espacio en general a menos de 1.5 m (ahí sale verde = exacto).
    static let zoneMin: Float = 0.28, zoneMax: Float = 0.65
    static let spaceMin: Float = 0.25, spaceMax: Float = 1.5
    /// Lados desde los que hay que ver una pieza para darla por lista (de 8 alrededor + arriba + abajo).
    static let sidesNeeded = 5

    private struct ZoneState {
        var sectors = Set<Int>()
        var conf: Float = 0
        var lastNew: TimeInterval = 0
        var goodTime: Float = 0
        var lastT: TimeInterval = 0
        var ready = false
        var readyAt: TimeInterval = 0
    }
    private var states: [String: ZoneState] = [:]
    private var last = FrameStats()
    private var goodEMA: Float = 0
    private var spaceReady = false
    private var lastGuideTime: TimeInterval = 0

    /// Aviso de un piso o pared que falta por leer (lo pone el motor; se usa cuando no hay nada más urgente).
    var gapHint: (text: String, spoken: String)?
    var onGuide: ((ScanGuide?, [String: Int]) -> Void)?

    func reset() {
        states = [:]; last = FrameStats(); goodEMA = 0; spaceReady = false; lastGuideTime = 0
    }

    // MARK: Avance

    func ingest(_ s: FrameStats, frame: ARFrame, zones: [FocusZone]) {
        last = s
        let t = frame.timestamp
        if s.total > 300 {
            goodEMA = goodEMA * 0.7 + 0.3 * Float(s.good) / Float(s.total)
        }
        let cam = Self.position(frame)
        for (k, id) in s.zoneIDs.enumerated() {
            guard let z = zones.first(where: { $0.id == id }) else { continue }
            var st = states[id] ?? ZoneState()
            let dt = st.lastT > 0 ? Float(min(0.3, t - st.lastT)) : 0
            st.lastT = t
            let hits = s.zoneHits[k]
            if hits >= 150 {
                let md = s.zoneDepth[k] / Float(hits)
                if md >= Self.zoneMin - 0.03 && md <= Self.zoneMax + 0.1 {
                    if st.sectors.insert(Self.sector(of: cam - z.center)).inserted { st.lastNew = t }
                    st.conf = st.conf * 0.8 + 0.2 * Float(s.zoneConfirmed[k]) / Float(hits)
                    st.goodTime += dt
                }
            }
            if !st.ready {
                let n = st.sectors.count
                // vista de suficientes lados, o (pieza contra la pared) ya no hay lados nuevos que ver
                if (n >= Self.sidesNeeded && st.conf >= 0.6) ||
                    (n >= 3 && st.conf >= 0.6 && t - st.lastNew > 10 && st.goodTime > 15) {
                    st.ready = true
                    st.readyAt = t
                }
            }
            states[id] = st
        }
    }

    private func progress(_ zones: [FocusZone]) -> [String: Int] {
        var out: [String: Int] = [:]
        for z in zones {
            guard let st = states[z.id] else { out[z.id] = 0; continue }
            if st.ready { out[z.id] = 100; continue }
            let f = 0.75 * min(1, Float(st.sectors.count) / Float(Self.sidesNeeded)) + 0.25 * min(1, st.conf / 0.6)
            out[z.id] = min(99, Int(f * 100))
        }
        return out
    }

    // MARK: Indicación

    func guide(frame: ARFrame, zones: [FocusZone], centerDepth: Float?) {
        let t = frame.timestamp
        guard t - lastGuideTime >= 0.4 else { return }
        lastGuideTime = t
        let prog = progress(zones)
        var g: ScanGuide?

        // 1) Una pieza enfocada en la vista y todavía sin terminar
        var best: (hits: Int, zone: FocusZone, depth: Float)?
        for (k, id) in last.zoneIDs.enumerated() where last.zoneHits[k] >= 150 {
            guard let z = zones.first(where: { $0.id == id }), !(states[id]?.ready ?? false) else { continue }
            if best == nil || last.zoneHits[k] > best!.hits {
                best = (last.zoneHits[k], z, last.zoneDepth[k] / Float(last.zoneHits[k]))
            }
        }
        let justReady = zones.first { z in
            guard let st = states[z.id], st.ready else { return false }
            return t - st.readyAt < 5
        }
        if let jr = justReady {
            g = ScanGuide(kind: .ready, text: "\(jr.name) lista ✓ · toca otra pieza o sigue escaneando",
                          spoken: "\(jr.name) lista. Puedes tocar otra pieza o seguir escaneando.")
        } else if let b = best {
            let p = prog[b.zone.id] ?? 0
            if b.depth < Self.zoneMin {
                g = ScanGuide(kind: .farther, text: "\(b.zone.name): muy cerca (\(Self.cm(b.depth))) · aléjate a 30–60 cm",
                              spoken: "Aléjate un poco")
            } else if b.depth > Self.zoneMax {
                g = ScanGuide(kind: .closer, text: "\(b.zone.name): acércate (\(Self.cm(b.depth)) → 30–60 cm)",
                              spoken: "Acércate a la pieza")
            } else {
                g = ScanGuide(kind: .ok, text: "\(b.zone.name) \(p) % · buena distancia · \(sideHint(b.zone, frame))")
            }
        } else {
            // 2) El espacio en general
            if !spaceReady && goodEMA >= 0.85 { spaceReady = true } else if spaceReady && goodEMA < 0.65 { spaceReady = false }
            let md = Self.median(last.depths)
            let near = centerDepth ?? md
            if let d = near, d < Self.spaceMin {
                g = ScanGuide(kind: .farther, text: "Muy cerca (\(Self.cm(d))) · aléjate, el LiDAR lee desde 25 cm",
                              spoken: "Aléjate")
            } else if spaceReady {
                g = ScanGuide(kind: .ready, text: "Esta parte ya está lista ✓ · pasa a otra área o toca una pieza",
                              spoken: "Esta parte está lista")
            } else if let d = md, d > Self.spaceMax {
                g = ScanGuide(kind: .closer, text: String(format: "Acércate: estás a %.1f m · a menos de 1.5 m sale exacto", d),
                              spoken: "Acércate")
            } else if last.total > 0 {
                let pct = Int(min(0.99, goodEMA / 0.85) * 100)
                g = ScanGuide(kind: .ok, text: zones.isEmpty
                              ? "Esta vista \(pct) % · muévete despacio · toca una pieza para más detalle"
                              : "Esta vista \(pct) % · buena distancia, muévete despacio")
            }
        }
        if (g == nil || g?.kind == .ok), let gh = gapHint {
            g = ScanGuide(kind: .gap, text: gh.text, spoken: gh.spoken)
        }
        DispatchQueue.main.async { self.onGuide?(g, prog) }
    }

    /// Hacia qué lado rodear la pieza para ver el lado que falta.
    private func sideHint(_ z: FocusZone, _ frame: ARFrame) -> String {
        guard let st = states[z.id] else { return "rodéala despacio" }
        let cam = Self.position(frame)
        let v = cam - z.center
        let myAz = atan2(v.z, v.x)
        let missing = (0..<8).filter { !st.sectors.contains($0) }
        if missing.isEmpty || st.sectors.filter({ $0 < 8 }).count >= Self.sidesNeeded {
            return st.sectors.contains(100) ? "sigue rodeándola" : "mírala también desde arriba"
        }
        // lado faltante más cercano
        func azOf(_ b: Int) -> Float { (Float(b) + 0.5) / 8 * 2 * .pi - .pi }
        func diff(_ a: Float, _ b: Float) -> Float { var d = a - b; while d > .pi { d -= 2 * .pi }; while d < -.pi { d += 2 * .pi }; return d }
        let target = missing.min { abs(diff(azOf($0), myAz)) < abs(diff(azOf($1), myAz)) }!
        let r = max(0.4, simd_length(SIMD2<Float>(v.x, v.z)))
        let a = azOf(target)
        let goal = z.center + SIMD3<Float>(cos(a) * r, v.y, sin(a) * r)
        let T = frame.camera.transform
        let right = SIMD3<Float>(T.columns.0.x, T.columns.0.y, T.columns.0.z)
        return simd_dot(goal - cam, right) >= 0 ? "rodéala hacia la derecha →" : "← rodéala hacia la izquierda"
    }

    // MARK: Utilidades

    /// Lado desde el que se ve la pieza: 8 alrededor, uno arriba (100) y 4 abajo (200…).
    static func sector(of v: SIMD3<Float>) -> Int {
        let l = simd_length(v)
        guard l > 1e-4 else { return 0 }
        let el = asin(max(-1, min(1, v.y / l)))
        if el > 0.6 { return 100 }
        let b = Int(((atan2(v.z, v.x) + .pi) / (2 * .pi) * 8).rounded(.down)) % 8
        if el < -0.45 { return 200 + b / 2 }
        return b
    }

    static func position(_ f: ARFrame) -> SIMD3<Float> {
        let T = f.camera.transform
        return SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
    }

    static func median(_ a: [Float]) -> Float? {
        guard !a.isEmpty else { return nil }
        let s = a.sorted()
        return s[s.count / 2]
    }

    static func cm(_ d: Float) -> String { String(format: "%.0f cm", d * 100) }
}

/// Dice en voz alta los cambios importantes (acércate, aléjate, lista) para no tener que ver la pantalla.
final class GuideVoice {
    private let synth = AVSpeechSynthesizer()
    private var lastSpoken = ""
    private var lastSpokenAt = Date.distantPast
    private var pendingKind: ScanGuide.Kind?
    private var pendingCount = 0
    private var lastReadyText = ""

    func reset() { lastSpoken = ""; pendingKind = nil; pendingCount = 0; lastReadyText = "" }

    func handle(_ g: ScanGuide?, enabled: Bool) {
        guard let g else { pendingKind = nil; return }
        // solo cuando la indicación se mantiene (evita hablar en cada cambio pequeño)
        if g.kind == pendingKind { pendingCount += 1 } else { pendingKind = g.kind; pendingCount = 1 }
        if g.kind == .ok { lastSpoken = ""; return }
        if g.kind == .ready {
            guard g.text != lastReadyText else { return }
            lastReadyText = g.text
            UINotificationFeedbackGenerator().notificationOccurred(.success)
            say(g.spoken, enabled)
            return
        }
        lastReadyText = ""
        guard pendingCount >= 2, !g.spoken.isEmpty else { return }
        // se repite si sigues fuera de distancia, pero no más de cada 8 s
        if g.spoken != lastSpoken || Date().timeIntervalSince(lastSpokenAt) > 8 {
            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
            say(g.spoken, enabled)
        }
    }

    private func say(_ text: String, _ enabled: Bool) {
        lastSpoken = text
        lastSpokenAt = Date()
        guard enabled, !text.isEmpty else { return }
        try? AVAudioSession.sharedInstance().setCategory(.playback, options: [.duckOthers, .mixWithOthers])
        try? AVAudioSession.sharedInstance().setActive(true)
        if synth.isSpeaking { synth.stopSpeaking(at: .word) }
        let u = AVSpeechUtterance(string: text)
        u.voice = AVSpeechSynthesisVoice(language: "es-MX") ?? AVSpeechSynthesisVoice(language: "es-ES")
        u.rate = 0.52
        synth.speak(u)
    }
}
