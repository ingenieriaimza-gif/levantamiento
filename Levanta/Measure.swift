import SceneKit
import SwiftUI
import UIKit

/// Medición en "Revisar 3D": se toca un punto de la nube, luego otro, y queda la distancia.
/// Las medidas están en las mismas coordenadas del modelo, así que también aparecen en la PC.
final class MeasureController: ObservableObject {

    struct Measure: Identifiable {
        let id: String
        var name: String
        var a: SIMD3<Float>
        var b: SIMD3<Float>
        var distance: Float { simd_distance(a, b) }
        var horizontal: Float { simd_length(SIMD2<Float>(a.x - b.x, a.z - b.z)) }
        var vertical: Float { abs(a.y - b.y) }
    }

    @Published private(set) var measures: [Measure] = []
    @Published private(set) var pending: SIMD3<Float>?
    @Published var active = false {
        didSet { if !active { pending = nil; redraw() } }
    }
    /// Aviso corto cuando el toque no cae sobre puntos.
    @Published var missed = false

    /// Se llama cada vez que cambia la lista (para enviarla a la PC).
    var onChange: (([Measure]) -> Void)?

    /// Nodo con líneas y etiquetas; se agrega a la escena de la revisión 3D.
    let root = SCNNode()
    /// Tamaño de marcas y etiquetas según el tamaño del escaneo.
    var markScale: Float = 1 { didSet { redraw() } }

    private var counter = 0
    private let measureColor = UIColor(red: 0.96, green: 0.66, blue: 0.13, alpha: 1)

    /// Carga medidas guardadas (sin avisar a la PC).
    func load(_ items: [[String: Any]]) {
        func v(_ x: Any?) -> SIMD3<Float>? {
            guard let a = x as? [NSNumber], a.count == 3 else { return nil }
            return SIMD3<Float>(a[0].floatValue, a[1].floatValue, a[2].floatValue)
        }
        measures = items.compactMap { m in
            guard let a = v(m["a"]), let b = v(m["b"]) else { return nil }
            return Measure(id: m["id"] as? String ?? UUID().uuidString, name: m["nombre"] as? String ?? "iPhone", a: a, b: b)
        }
        counter = measures.count
        pending = nil
        redraw()
    }

    // MARK: Puntos

    func tap(_ p: SIMD3<Float>) {
        missed = false
        if let a = pending {
            counter += 1
            measures.append(Measure(id: "ip\(Int(Date().timeIntervalSince1970 * 1000))", name: "iPhone \(counter)",
                                    a: a, b: p))
            pending = nil
            onChange?(measures)
        } else {
            pending = p
        }
        redraw()
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func undo() {
        if pending != nil {
            pending = nil
        } else if !measures.isEmpty {
            measures.removeLast()
            onChange?(measures)
        }
        redraw()
    }

    func clear() {
        let had = !measures.isEmpty
        measures.removeAll()
        pending = nil
        redraw()
        if had { onChange?(measures) }
    }

    // MARK: Dibujo

    private func redraw() {
        root.childNodes.forEach { $0.removeFromParentNode() }
        for m in measures {
            let n = SCNNode()
            Self.makeSegment(n, a: m.a, b: m.b, color: measureColor, label: Self.format(m.distance), scale: markScale)
            root.addChildNode(n)
        }
        if let p = pending {
            let s = SCNSphere(radius: CGFloat(0.004 * markScale))
            s.firstMaterial = Self.overlayMaterial(.white)
            let n = SCNNode(geometry: s)
            n.simdPosition = p
            n.renderingOrder = 100
            root.addChildNode(n)
        }
    }

    static func format(_ meters: Float) -> String {
        if meters >= 1 { return String(format: "%.3f m", meters) }
        return String(format: "%.1f mm", meters * 1000)
    }

    static func overlayMaterial(_ color: UIColor) -> SCNMaterial {
        let m = SCNMaterial()
        m.diffuse.contents = color
        m.lightingModel = .constant
        m.readsFromDepthBuffer = false
        m.writesToDepthBuffer = false
        return m
    }

    /// Línea (cilindro delgado) entre dos puntos, con esferas en los extremos y etiqueta opcional.
    static func makeSegment(_ node: SCNNode, a: SIMD3<Float>, b: SIMD3<Float>, color: UIColor,
                            label: String?, scale: Float = 1) {
        node.childNodes.forEach { $0.removeFromParentNode() }
        node.geometry = nil
        let d = b - a
        let len = simd_length(d)
        guard len > 0.0005 else { return }
        let cyl = SCNCylinder(radius: CGFloat(0.001 * scale), height: CGFloat(len))
        cyl.firstMaterial = overlayMaterial(color)
        let line = SCNNode(geometry: cyl)
        line.simdPosition = (a + b) / 2
        line.simdOrientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: d / len)
        line.renderingOrder = 100
        node.addChildNode(line)
        for p in [a, b] {
            let s = SCNSphere(radius: CGFloat(0.003 * scale))
            s.firstMaterial = overlayMaterial(color)
            let sn = SCNNode(geometry: s)
            sn.simdPosition = p
            sn.renderingOrder = 100
            node.addChildNode(sn)
        }
        guard let label else { return }
        let text = SCNText(string: label, extrusionDepth: 0)
        text.font = UIFont.monospacedDigitSystemFont(ofSize: 10, weight: .bold)
        text.flatness = 0.1
        text.firstMaterial = overlayMaterial(.white)
        let tn = SCNNode(geometry: text)
        let (minB, maxB) = text.boundingBox
        let w = maxB.x - minB.x, h = maxB.y - minB.y
        tn.pivot = SCNMatrix4MakeTranslation(minB.x + w / 2, minB.y + h / 2, 0)
        tn.renderingOrder = 102
        let bg = SCNPlane(width: CGFloat(w) + 6, height: CGFloat(h) + 4)
        bg.cornerRadius = 3
        bg.firstMaterial = overlayMaterial(UIColor(white: 0.05, alpha: 0.85))
        let bgNode = SCNNode(geometry: bg)
        bgNode.renderingOrder = 101
        let holder = SCNNode()
        holder.addChildNode(bgNode)
        holder.addChildNode(tn)
        let k = 0.0025 * scale
        holder.scale = SCNVector3(k, k, k)
        holder.simdPosition = (a + b) / 2 + SIMD3<Float>(0, 0.02 * scale, 0)
        holder.constraints = [SCNBillboardConstraint()]
        node.addChildNode(holder)
    }
}

// MARK: - Elegir el punto tocado

enum PointPicker {
    /// Punto de la nube bajo el dedo: el más cercano a la cámara dentro de un radio en pantalla,
    /// afinado con el promedio de los puntos vecinos (≤ 6 mm) para quitar ruido.
    static func pick(at tap: CGPoint, in view: SCNView, points: [SIMD3<Float>]) -> SIMD3<Float>? {
        guard !points.isEmpty, let pov = view.pointOfView, let cam = pov.camera else { return nil }
        let size = view.bounds.size
        guard size.width > 0, size.height > 0 else { return nil }
        let inv = simd_inverse(pov.simdWorldTransform)
        let fovRad = Float(cam.fieldOfView) * .pi / 180
        let vertical = cam.projectionDirection == .vertical
        let half = Float(vertical ? size.height : size.width) / 2
        let f = half / tan(fovRad / 2)
        let cx = Float(size.width) / 2, cy = Float(size.height) / 2
        let tx = Float(tap.x), ty = Float(tap.y)
        let zNear = Float(cam.zNear)

        func search(radius: Float) -> (Int, Float)? {
            let r2 = radius * radius
            var best = -1
            var bestDepth = Float.greatestFiniteMagnitude
            for i in 0..<points.count {
                let p = points[i]
                let c = inv * SIMD4<Float>(p.x, p.y, p.z, 1)
                let depth = -c.z
                guard depth > zNear, depth < bestDepth else { continue }
                let sx = cx + f * c.x / depth
                let sy = cy - f * c.y / depth
                let dx = sx - tx, dy = sy - ty
                if dx * dx + dy * dy <= r2 { best = i; bestDepth = depth }
            }
            return best >= 0 ? (best, bestDepth) : nil
        }
        // con radio chico primero (precisión); si no hay puntos, un poco más amplio
        guard let (idx, _) = search(radius: 10) ?? search(radius: 24) else { return nil }
        let seed = points[idx]
        var sum = SIMD3<Float>(repeating: 0)
        var n: Float = 0
        for p in points where simd_distance_squared(p, seed) <= 0.006 * 0.006 {
            sum += p
            n += 1
        }
        return n > 0 ? sum / n : seed
    }
}

// MARK: - Interfaz de medición (dentro de Revisar 3D)

struct MeasurePanel: View {
    @ObservedObject var measure: MeasureController
    var onDone: (() -> Void)? = nil

    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.caption).foregroundStyle(measure.missed ? Color.orange : Color.secondary)
                    Text(readout)
                        .font(.system(size: 30, weight: .bold, design: .monospaced))
                        .contentTransition(.numericText())
                }
                Spacer()
                if let last = measure.measures.last, measure.pending == nil {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("horizontal \(MeasureController.format(last.horizontal))")
                        Text("vertical \(MeasureController.format(last.vertical))")
                    }
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 18) {
                smallButton("arrow.uturn.backward", "Deshacer") { measure.undo() }
                    .disabled(measure.pending == nil && measure.measures.isEmpty)
                smallButton("trash", "Borrar") { measure.clear() }
                    .disabled(measure.measures.isEmpty)
                Spacer()
                smallButton("checkmark", "Listo") {
                    if let onDone { onDone() } else { measure.active = false }
                }
            }

            if !measure.measures.isEmpty {
                ScrollView {
                    VStack(spacing: 6) {
                        ForEach(measure.measures.reversed()) { m in
                            HStack {
                                Text(m.name).font(.caption)
                                Spacer()
                                Text(MeasureController.format(m.distance)).font(.caption.monospaced().weight(.semibold))
                            }
                        }
                    }
                }
                .frame(maxHeight: 70)
            }
            Text("Toca dos veces para acercarte a un punto. Un toque pone cada punto. Lo que midas aparece también en la PC.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .foregroundStyle(.white)
    }

    private var title: String {
        if measure.missed { return "Ahí no hay puntos: toca sobre la nube" }
        return measure.pending == nil ? "Toca el primer punto" : "Toca el segundo punto"
    }

    private var readout: String {
        if let last = measure.measures.last, measure.pending == nil { return MeasureController.format(last.distance) }
        return "—"
    }

    private func smallButton(_ icon: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                    .frame(width: 48, height: 48)
                    .background(Color.white.opacity(0.12), in: Circle())
                Text(label).font(.caption2)
            }
        }
    }
}
