import ARKit
import SceneKit
import SwiftUI

/// Medición en realidad aumentada: la mira central toma puntos 3D con el LiDAR (cualquier superficie),
/// dibuja líneas y distancias sobre la cámara y comparte las medidas con la PC.
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

    @Published private(set) var active = false
    @Published private(set) var measures: [Measure] = []
    @Published private(set) var pending: SIMD3<Float>?
    @Published private(set) var liveDistance: Float?
    @Published private(set) var hasHit = false

    /// Se llama cada vez que cambia la lista (para enviarla a la PC).
    var onChange: (([Measure]) -> Void)?

    let root = SCNNode()
    private let cursor = SCNNode()
    private let rubber = SCNNode()
    private weak var view: ARSCNView?
    private var current: SIMD3<Float>?
    private var timer: Timer?
    private var counter = 0

    private let measureColor = UIColor(red: 0.96, green: 0.66, blue: 0.13, alpha: 1)

    func attach(_ v: ARSCNView) {
        view = v
        v.scene.rootNode.addChildNode(root)
        let ring = SCNTorus(ringRadius: 0.012, pipeRadius: 0.0015)
        ring.firstMaterial = Self.overlayMaterial(.white)
        let ringNode = SCNNode(geometry: ring)
        cursor.addChildNode(ringNode)
        cursor.isHidden = true
        v.scene.rootNode.addChildNode(cursor)
        v.scene.rootNode.addChildNode(rubber)
    }

    func toggle() { setActive(!active) }

    func setActive(_ on: Bool) {
        active = on
        pending = nil
        liveDistance = nil
        timer?.invalidate()
        timer = nil
        if on {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 15.0, repeats: true) { [weak self] _ in self?.tick() }
        } else {
            cursor.isHidden = true
            rubber.geometry = nil
            rubber.childNodes.forEach { $0.removeFromParentNode() }
        }
    }

    // MARK: Puntos

    func addPoint() {
        guard let c = current else { return }
        if let p = pending {
            counter += 1
            measures.append(Measure(id: "ip\(Int(Date().timeIntervalSince1970 * 1000))", name: "iPhone \(counter)",
                                    a: p, b: c))
            pending = nil
            liveDistance = nil
            redraw()
            onChange?(measures)
        } else {
            pending = c
        }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }

    func undo() {
        if pending != nil {
            pending = nil
            liveDistance = nil
        } else if !measures.isEmpty {
            measures.removeLast()
            redraw()
            onChange?(measures)
        }
    }

    func clear() {
        measures.removeAll()
        pending = nil
        liveDistance = nil
        redraw()
        onChange?(measures)
    }

    // MARK: Mira (15 veces por segundo)

    private func tick() {
        guard let v = view else { return }
        let center = CGPoint(x: v.bounds.midX, y: v.bounds.midY)
        var hit: SIMD3<Float>?
        var normal = SIMD3<Float>(0, 1, 0)
        if let q = v.raycastQuery(from: center, allowing: .estimatedPlane, alignment: .any),
           let r = v.session.raycast(q).first {
            let t = r.worldTransform
            hit = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
            normal = SIMD3<Float>(t.columns.1.x, t.columns.1.y, t.columns.1.z)
        }
        current = hit
        hasHit = hit != nil
        if let h = hit {
            cursor.isHidden = false
            cursor.simdPosition = h
            cursor.simdOrientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: simd_normalize(normal))
        } else {
            cursor.isHidden = true
        }
        if let p = pending, let h = hit {
            liveDistance = simd_distance(p, h)
            Self.makeSegment(rubber, a: p, b: h, color: UIColor.white.withAlphaComponent(0.85), label: nil)
        } else {
            liveDistance = nil
            rubber.geometry = nil
            rubber.childNodes.forEach { $0.removeFromParentNode() }
        }
    }

    // MARK: Dibujo

    private func redraw() {
        root.childNodes.forEach { $0.removeFromParentNode() }
        for m in measures {
            let n = SCNNode()
            Self.makeSegment(n, a: m.a, b: m.b, color: measureColor, label: Self.format(m.distance))
            root.addChildNode(n)
        }
    }

    static func format(_ meters: Float) -> String {
        if meters >= 1 { return String(format: "%.3f m", meters) }
        return String(format: "%.0f mm", meters * 1000)
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
    static func makeSegment(_ node: SCNNode, a: SIMD3<Float>, b: SIMD3<Float>, color: UIColor, label: String?) {
        node.childNodes.forEach { $0.removeFromParentNode() }
        node.geometry = nil
        let d = b - a
        let len = simd_length(d)
        guard len > 0.0005 else { return }
        let cyl = SCNCylinder(radius: 0.0012, height: CGFloat(len))
        cyl.firstMaterial = overlayMaterial(color)
        let line = SCNNode(geometry: cyl)
        line.simdPosition = (a + b) / 2
        line.simdOrientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: d / len)
        line.renderingOrder = 100
        node.addChildNode(line)
        for p in [a, b] {
            let s = SCNSphere(radius: 0.004)
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
        holder.scale = SCNVector3(0.0035, 0.0035, 0.0035)
        holder.simdPosition = (a + b) / 2 + SIMD3<Float>(0, 0.025, 0)
        holder.constraints = [SCNBillboardConstraint()]
        node.addChildNode(holder)
    }
}

// MARK: - Interfaz de medición

struct MeasureToggle: View {
    @ObservedObject var measure: MeasureController

    var body: some View {
        Button {
            measure.toggle()
        } label: {
            Image(systemName: "ruler")
                .frame(width: 40, height: 40)
                .background(measure.active ? AnyShapeStyle(Color(red: 0.96, green: 0.66, blue: 0.13))
                                           : AnyShapeStyle(.ultraThinMaterial), in: Circle())
                .foregroundStyle(measure.active ? Color.black : Color.white)
        }
        .accessibilityLabel(measure.active ? "Salir de medir" : "Medir")
    }
}

struct MeasurePanel: View {
    @ObservedObject var measure: MeasureController

    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.caption).foregroundStyle(.secondary)
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
                Button {
                    measure.addPoint()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 30, weight: .bold))
                        .frame(width: 72, height: 72)
                        .background(Color(red: 0.96, green: 0.66, blue: 0.13), in: Circle())
                        .foregroundStyle(.black)
                }
                .disabled(!measure.hasHit)
                .accessibilityLabel(measure.pending == nil ? "Primer punto" : "Segundo punto")
                smallButton("trash", "Borrar") { measure.clear() }
                    .disabled(measure.measures.isEmpty)
            }
            .frame(maxWidth: .infinity)

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
            Text("Apunta la mira al punto y toca +. Lo que midas aparece también en la PC.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .foregroundStyle(.white)
    }

    private var title: String {
        if !measure.hasHit { return "Apunta a una superficie" }
        return measure.pending == nil ? "Primer punto" : "Distancia"
    }

    private var readout: String {
        if let d = measure.liveDistance { return MeasureController.format(d) }
        if let last = measure.measures.last { return MeasureController.format(last.distance) }
        return "—"
    }

    private func smallButton(_ icon: String, _ label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                    .frame(width: 52, height: 52)
                    .background(Color.white.opacity(0.12), in: Circle())
                Text(label).font(.caption2)
            }
        }
    }
}

/// Elige entre el panel de escaneo y el de medición según el modo.
struct BottomArea<ScanBar: View>: View {
    @ObservedObject var measure: MeasureController
    let scanBar: ScanBar

    var body: some View {
        if measure.active {
            MeasurePanel(measure: measure)
        } else {
            scanBar
        }
    }
}
