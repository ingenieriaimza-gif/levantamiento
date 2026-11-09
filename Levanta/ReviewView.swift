import SwiftUI
import SceneKit

/// Revisión 3D de la nube fuera de la cámara: se gira con un dedo, se acerca con dos.
struct ReviewView: View {
    @ObservedObject var engine: ScanEngine
    @Environment(\.dismiss) private var dismiss

    @State private var mode: ScanEngine.ColorMode = .coverage
    @State private var geometry: SCNGeometry?
    @State private var stats: ScanEngine.CloudStats?
    @State private var loading = true
    @State private var view: ReviewCamera = .perspective

    enum ReviewCamera { case perspective, top, front }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let geometry, let stats {
                CloudSceneView(geometry: geometry, center: stats.center, size: stats.size, camera: view)
                    .ignoresSafeArea()
            } else if !loading {
                Text("No hay puntos todavía.")
                    .foregroundStyle(.secondary)
            }

            if loading { ProgressView().tint(.white) }

            VStack(spacing: 10) {
                HStack {
                    Picker("Color", selection: $mode) {
                        ForEach(ScanEngine.ColorMode.allCases) { m in Text(m.rawValue).tag(m) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 240)
                    Spacer()
                    Button("Cerrar") { dismiss() }
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                }

                Spacer()

                if let stats, stats.points > 0 {
                    statsPanel(stats)
                }

                HStack(spacing: 10) {
                    cameraButton("Perspectiva", .perspective)
                    cameraButton("Planta", .top)
                    cameraButton("Frente", .front)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .foregroundStyle(.white)
        }
        .onAppear(perform: load)
        .onChange(of: mode) { _, _ in load() }
    }

    private func load() {
        loading = true
        engine.makeReviewData(mode: mode) { geo, st in
            geometry = geo
            stats = st
            loading = false
        }
    }

    private func statsPanel(_ s: ScanEngine.CloudStats) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Ancho × Largo × Alto").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("\(s.points) puntos").font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Text(String(format: "%.2f × %.2f × %.2f m", s.size.x, s.size.z, s.size.y))
                .font(.system(size: 20, weight: .semibold, design: .monospaced))
            if mode == .coverage {
                HStack(spacing: 14) {
                    legend(Color(red: 0.95, green: 0.24, blue: 0.20), "1 lectura")
                    legend(Color(red: 0.98, green: 0.70, blue: 0.15), "2–3")
                    legend(Color(red: 0.25, green: 0.85, blue: 0.40), "4 o más")
                }
                .font(.caption)
                let total = max(1, s.points + s.weak)
                Text("Puntos débiles: \(Int((Double(s.weak) / Double(total) * 100).rounded())) %. Las zonas rojas o vacías conviene repasarlas.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private func legend(_ color: Color, _ text: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 9, height: 9)
            Text(text)
        }
    }

    private func cameraButton(_ title: String, _ cam: ReviewCamera) -> some View {
        Button(title) { view = cam }
            .font(.footnote.weight(.semibold))
            .padding(.horizontal, 14).padding(.vertical, 9)
            .background(view == cam ? Color.white.opacity(0.28) : Color.white.opacity(0.1), in: Capsule())
    }
}

/// Vista SceneKit con control de cámara táctil.
struct CloudSceneView: UIViewRepresentable {
    let geometry: SCNGeometry
    let center: SIMD3<Float>
    let size: SIMD3<Float>
    let camera: ReviewView.ReviewCamera

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero)
        v.backgroundColor = UIColor(white: 0.06, alpha: 1)
        v.allowsCameraControl = true
        v.antialiasingMode = .none
        v.scene = SCNScene()
        let cloud = SCNNode(geometry: geometry)
        cloud.name = "cloud"
        v.scene?.rootNode.addChildNode(cloud)
        let camNode = SCNNode()
        camNode.camera = SCNCamera()
        camNode.camera?.zNear = 0.01
        camNode.camera?.zFar = 500
        camNode.name = "cam"
        v.scene?.rootNode.addChildNode(camNode)
        v.pointOfView = camNode
        context.coordinator.lastCamera = nil
        apply(to: v, coordinator: context.coordinator)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {
        if let cloud = v.scene?.rootNode.childNode(withName: "cloud", recursively: false),
           cloud.geometry !== geometry {
            cloud.geometry = geometry
        }
        apply(to: v, coordinator: context.coordinator)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var lastCamera: ReviewView.ReviewCamera?
    }

    private func apply(to v: SCNView, coordinator: Coordinator) {
        guard coordinator.lastCamera != camera else { return }
        coordinator.lastCamera = camera
        guard let camNode = v.scene?.rootNode.childNode(withName: "cam", recursively: false) else { return }
        let c = SCNVector3(center.x, center.y, center.z)
        let span = max(0.5, max(size.x, max(size.y, size.z)))
        let dist = span * 1.6
        switch camera {
        case .perspective:
            camNode.position = SCNVector3(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.8)
        case .top:
            camNode.position = SCNVector3(center.x, center.y + dist * 1.2, center.z + 0.001)
        case .front:
            camNode.position = SCNVector3(center.x, center.y, center.z + dist * 1.2)
        }
        if camera == .top {
            camNode.position = SCNVector3(center.x, center.y + dist * 1.2, center.z)
            camNode.look(at: c, up: SCNVector3(0, 0, -1), localFront: SCNVector3(0, 0, -1))
        } else {
            camNode.look(at: c)
        }
        v.pointOfView = camNode
        v.defaultCameraController.target = c
    }
}
