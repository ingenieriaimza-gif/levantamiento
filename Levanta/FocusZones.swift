import ARKit
import SceneKit
import SwiftUI
import Vision

/// Zonas de detalle: tocas una pieza mientras escaneas, la IA del iPhone (Vision, recorte de objetos)
/// la separa del fondo, el LiDAR da su caja en 3D, y dentro de esa caja se guardan puntos cada 2 mm,
/// se toman más fotos y la PC la reconstruye con más detalle.
struct FocusZone: Identifiable, Equatable {
    let id: String
    var name: String
    var center: SIMD3<Float>
    var half: SIMD3<Float>          // medio tamaño de la caja (m), alineada a los ejes del mundo

    func contains(_ p: SIMD3<Float>, margin: Float = 0) -> Bool {
        let d = simd_abs(p - center)
        return d.x <= half.x + margin && d.y <= half.y + margin && d.z <= half.z + margin
    }

    /// Distancia de un punto a la caja (0 si está dentro).
    func distance(to p: SIMD3<Float>) -> Float {
        let d = simd_max(simd_abs(p - center) - half, SIMD3<Float>(repeating: 0))
        return simd_length(d)
    }

    var json: [String: Any] {
        ["id": id, "nombre": name, "centro": [center.x, center.y, center.z], "medio": [half.x, half.y, half.z]]
    }
}

final class FocusController: ObservableObject {
    @Published private(set) var zones: [FocusZone] = []
    @Published var message: String?
    @Published private(set) var busy = false

    /// Se llama al cambiar la lista (para mandarla a la PC y guardarla con el escaneo).
    var onChange: (([FocusZone]) -> Void)?
    /// Copia para la cola de escaneo (se lee en cada cuadro).
    private(set) var snapshot: [FocusZone] = []
    private let lock = NSLock()

    let root = SCNNode()
    private var counter = 0
    private let color = UIColor(red: 0.96, green: 0.66, blue: 0.13, alpha: 1)

    func current() -> [FocusZone] {
        lock.lock(); defer { lock.unlock() }
        return snapshot
    }

    private func publish(_ z: [FocusZone]) {
        lock.lock(); snapshot = z; lock.unlock()
        zones = z
        redraw()
        onChange?(z)
    }

    func remove(_ id: String) {
        publish(zones.filter { $0.id != id })
    }

    func clear() {
        counter = 0
        publish([])
    }

    // MARK: Seleccionar

    /// Toque en la pantalla: identifica el objeto bajo el dedo y crea su zona.
    func select(at tap: CGPoint, viewSize: CGSize, frame: ARFrame, orientation: UIInterfaceOrientation) {
        guard !busy else { return }
        guard let depth = frame.sceneDepth ?? frame.smoothedSceneDepth else {
            message = "Espera a que el LiDAR vea la pieza"
            return
        }
        busy = true
        message = "Identificando la pieza…"
        // punto tocado en coordenadas normalizadas de la imagen del sensor (0–1, arriba a la izquierda)
        let toImage = frame.displayTransform(for: orientation, viewportSize: viewSize).inverted()
        let norm = CGPoint(x: tap.x / viewSize.width, y: tap.y / viewSize.height).applying(toImage)
        let image = frame.capturedImage
        let K = frame.camera.intrinsics
        let T = frame.camera.transform
        let imgW = Float(CVPixelBufferGetWidth(image)), imgH = Float(CVPixelBufferGetHeight(image))
        let depthMap = depth.depthMap
        let confMap = depth.confidenceMap

        DispatchQueue.global(qos: .userInitiated).async {
            let zone = Self.segment(image: image, tap: norm, depthMap: depthMap, confMap: confMap,
                                    K: K, T: T, imgW: imgW, imgH: imgH)
            DispatchQueue.main.async {
                self.busy = false
                guard var z = zone.zone else {
                    self.message = "No se encontró una pieza ahí; toca más cerca del objeto"
                    return
                }
                self.counter += 1
                z.name = "Pieza \(self.counter)"
                self.publish(self.zones + [z])
                let s = z.half * 200
                self.message = String(format: "%@ enfocada (%.0f × %.0f × %.0f cm)%@. Acércate a 30–60 cm y rodéala.",
                                      z.name, s.x, s.z, s.y, zone.byAI ? "" : " · zona alrededor del punto")
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        }
    }

    /// Recorte del objeto (Vision) + LiDAR -> caja 3D. Si no hay objeto separable, caja de 40 cm alrededor del punto.
    private static func segment(image: CVPixelBuffer, tap: CGPoint, depthMap: CVPixelBuffer, confMap: CVPixelBuffer?,
                                K: simd_float3x3, T: simd_float4x4, imgW: Float, imgH: Float)
        -> (zone: FocusZone?, byAI: Bool) {
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        if let c = confMap { CVPixelBufferLockBaseAddress(c, .readOnly) }
        defer { if let c = confMap { CVPixelBufferUnlockBaseAddress(c, .readOnly) } }
        guard let dBase = CVPixelBufferGetBaseAddress(depthMap) else { return (nil, false) }
        let dw = CVPixelBufferGetWidth(depthMap), dh = CVPixelBufferGetHeight(depthMap)
        let dRow = CVPixelBufferGetBytesPerRow(depthMap)
        let cBase = confMap.flatMap { CVPixelBufferGetBaseAddress($0) }
        let cRow = confMap.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
        let sx = Float(dw) / imgW, sy = Float(dh) / imgH
        let fx = K[0][0] * sx, fy = K[1][1] * sy
        let cx = (K[2][0] + 0.5) * sx - 0.5, cy = (K[2][1] + 0.5) * sy - 0.5

        func depthAt(_ u: Int, _ v: Int) -> Float? {
            guard u >= 0, v >= 0, u < dw, v < dh else { return nil }
            let d = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)[u]
            if let cb = cBase, cb.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)[u] < 1 { return nil }
            return d.isFinite && d > 0.1 && d < 4 ? d : nil
        }
        func world(_ u: Int, _ v: Int, _ d: Float) -> SIMD3<Float> {
            let pc = SIMD4<Float>((Float(u) - cx) / fx * d, -(Float(v) - cy) / fy * d, -d, 1)
            let w = T * pc
            return SIMD3<Float>(w.x, w.y, w.z)
        }
        let tu = Int((Float(tap.x) * Float(dw)).rounded(.down)), tv = Int((Float(tap.y) * Float(dh)).rounded(.down))
        // profundidad en el punto tocado (o la más cercana alrededor)
        var dTap: Float?
        search: for r in 0...6 {
            for dv in -r...r { for du in -r...r {
                if let d = depthAt(tu + du, tv + dv) { dTap = d; break search }
            } }
        }
        guard let d0 = dTap else { return (nil, false) }

        // 1) IA de Apple: recorte de la pieza bajo el dedo
        var pts: [SIMD3<Float>] = []
        let req = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .up)
        if (try? handler.perform([req])) != nil, let obs = req.results?.first {
            let mask = obs.instanceMask
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            if let mBase = CVPixelBufferGetBaseAddress(mask) {
                let mw = CVPixelBufferGetWidth(mask), mh = CVPixelBufferGetHeight(mask)
                let mRow = CVPixelBufferGetBytesPerRow(mask)
                func label(_ x: Float, _ y: Float) -> UInt8 {
                    let mx = min(mw - 1, max(0, Int(x * Float(mw)))), my = min(mh - 1, max(0, Int(y * Float(mh))))
                    return mBase.advanced(by: my * mRow).assumingMemoryBound(to: UInt8.self)[mx]
                }
                var lab = label(Float(tap.x), Float(tap.y))
                if lab == 0 {        // tocó justo en el borde: buscar la pieza más cercana (hasta 4 % de la imagen)
                    outer: for r in stride(from: Float(0.005), through: 0.04, by: 0.005) {
                        for a in stride(from: Float(0), to: 2 * .pi, by: .pi / 8) {
                            let l = label(Float(tap.x) + r * cos(a), Float(tap.y) + r * sin(a))
                            if l != 0 { lab = l; break outer }
                        }
                    }
                }
                if lab != 0 {
                    for v in 0..<dh { for u in 0..<dw {
                        guard label((Float(u) + 0.5) / Float(dw), (Float(v) + 0.5) / Float(dh)) == lab,
                              let d = depthAt(u, v), abs(d - d0) < 0.6 else { continue }
                        pts.append(world(u, v, d))
                    } }
                }
            }
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        }
        if pts.count >= 40 {
            // caja robusta: percentiles 2–98 % por eje y 3 cm de margen
            func pct(_ a: [Float], _ f: Float) -> Float { a.sorted()[min(a.count - 1, Int(Float(a.count - 1) * f))] }
            let xs = pts.map { $0.x }, ys = pts.map { $0.y }, zs = pts.map { $0.z }
            let lo = SIMD3<Float>(pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02))
            let hi = SIMD3<Float>(pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98))
            let half = (hi - lo) / 2 + 0.03
            // si la "pieza" es enorme (una pared, el piso) no sirve como zona de detalle
            if half.x < 1.2, half.y < 1.2, half.z < 1.2 {
                return (FocusZone(id: UUID().uuidString, name: "", center: (lo + hi) / 2,
                                  half: simd_max(half, SIMD3<Float>(repeating: 0.05))), true)
            }
        }
        // 2) Sin pieza separable: zona de 40 cm alrededor del punto tocado
        let p = world(min(max(tu, 0), dw - 1), min(max(tv, 0), dh - 1), d0)
        return (FocusZone(id: UUID().uuidString, name: "", center: p, half: SIMD3<Float>(repeating: 0.2)), false)
    }

    // MARK: Dibujo (caja en la cámara)

    private func redraw() {
        root.childNodes.forEach { $0.removeFromParentNode() }
        for z in zones {
            let n = SCNNode()
            let c = z.center, h = z.half
            var corners: [SCNVector3] = []
            for sx in [-1, 1] as [Float] { for sy in [-1, 1] as [Float] { for sz in [-1, 1] as [Float] {
                corners.append(SCNVector3(c.x + sx * h.x, c.y + sy * h.y, c.z + sz * h.z))
            } } }
            let edges: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7]
            let g = SCNGeometry(sources: [SCNGeometrySource(vertices: corners)],
                                elements: [SCNGeometryElement(indices: edges, primitiveType: .line)])
            g.firstMaterial = MeasureController.overlayMaterial(color)
            let box = SCNNode(geometry: g)
            box.renderingOrder = 90
            n.addChildNode(box)
            let text = SCNText(string: z.name, extrusionDepth: 0)
            text.font = UIFont.systemFont(ofSize: 10, weight: .bold)
            text.flatness = 0.1
            text.firstMaterial = MeasureController.overlayMaterial(color)
            let tn = SCNNode(geometry: text)
            let (a, b) = text.boundingBox
            tn.pivot = SCNMatrix4MakeTranslation((a.x + b.x) / 2, a.y, 0)
            tn.scale = SCNVector3(0.003, 0.003, 0.003)
            tn.position = SCNVector3(c.x, c.y + h.y + 0.02, c.z)
            tn.constraints = [SCNBillboardConstraint()]
            tn.renderingOrder = 91
            n.addChildNode(tn)
            root.addChildNode(n)
        }
    }
}

/// Indicación (acércate / aléjate / lista), fichas de las piezas con su avance y ✕ para quitarlas.
struct FocusBar: View {
    @ObservedObject var focus: FocusController
    let guide: ScanGuide?
    let progress: [String: Int]
    @Binding var voice: Bool
    let scanning: Bool

    private let amber = Color(red: 0.96, green: 0.66, blue: 0.13)
    private let green = Color(red: 0.20, green: 0.78, blue: 0.42)

    var body: some View {
        VStack(spacing: 6) {
            if let m = focus.message {
                pill(m, icon: "scope", bg: AnyShapeStyle(.ultraThinMaterial), fg: .white)
            } else if let g = guide {
                HStack(spacing: 8) {
                    switch g.kind {
                    case .closer: pill(g.text, icon: "arrow.up.forward.circle.fill", bg: AnyShapeStyle(amber.opacity(0.95)), fg: .black, big: true)
                    case .farther: pill(g.text, icon: "arrow.down.backward.circle.fill", bg: AnyShapeStyle(amber.opacity(0.95)), fg: .black, big: true)
                    case .ready: pill(g.text, icon: "checkmark.seal.fill", bg: AnyShapeStyle(green.opacity(0.95)), fg: .black, big: true)
                    case .ok: pill(g.text, icon: "hand.thumbsup.fill", bg: AnyShapeStyle(.ultraThinMaterial), fg: .white)
                    }
                    voiceButton
                }
            } else if scanning {
                HStack(spacing: 8) {
                    pill("Toca una pieza para enfocarla en detalle", icon: "hand.tap.fill",
                         bg: AnyShapeStyle(.ultraThinMaterial), fg: .white)
                    voiceButton
                }
            }
            if !focus.zones.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(focus.zones) { z in
                            let p = progress[z.id] ?? 0
                            let done = p >= 100
                            HStack(spacing: 6) {
                                Image(systemName: done ? "checkmark.circle.fill" : "scope")
                                Text(done ? "\(z.name) lista" : "\(z.name) · \(p) %")
                                Button { focus.remove(z.id) } label: { Image(systemName: "xmark.circle.fill") }
                                    .accessibilityLabel("Quitar \(z.name)")
                            }
                            .font(.caption.weight(.semibold).monospacedDigit())
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background((done ? green : amber).opacity(0.9), in: Capsule())
                            .foregroundStyle(.black)
                        }
                    }
                    .padding(.horizontal, 16)
                }
            }
        }
        .foregroundStyle(.white)
        .animation(.easeInOut(duration: 0.2), value: guide?.kind)
    }

    private var voiceButton: some View {
        Button { voice.toggle() } label: {
            Image(systemName: voice ? "speaker.wave.2.fill" : "speaker.slash.fill")
                .font(.footnote.weight(.bold))
                .frame(width: 34, height: 34)
                .background(.ultraThinMaterial, in: Circle())
        }
        .accessibilityLabel(voice ? "Silenciar indicaciones" : "Decir indicaciones en voz alta")
    }

    private func pill(_ text: String, icon: String, bg: AnyShapeStyle, fg: Color, big: Bool = false) -> some View {
        Label(text, systemImage: icon)
            .font(big ? .subheadline.weight(.bold) : .footnote.weight(.semibold))
            .multilineTextAlignment(.center)
            .padding(.horizontal, 14).padding(.vertical, big ? 10 : 8)
            .background(bg, in: Capsule())
            .foregroundStyle(fg)
    }
}
