import SwiftUI
import SceneKit
import WebKit

/// Revisar 3D: muestra el modelo terminado de la PC (con fotos), o el modelo en vivo de la PC,
/// o la nube del iPhone si no hay PC. Se gira con un dedo, se acerca con dos, doble toque para
/// acercarse a un punto, y con "Medir" se toca un punto y luego otro.
struct ReviewView: View {
    @ObservedObject var engine: ScanEngine
    @ObservedObject var link: PCLink
    @Environment(\.dismiss) private var dismiss
    @StateObject private var finalMeasure = MeasureController()

    enum Source: Hashable {
        case photo(PCLink.PCModel)      // vista foto-realista entrenada en la PC (solo para ver)
        case final(PCLink.PCModel)
        case live
        case cloud
    }
    enum ReviewCamera { case perspective, top, front }

    @State private var source: Source?
    @State private var models: [PCLink.PCModel] = []
    @State private var content: SCNNode?
    @State private var center = SIMD3<Float>(repeating: 0)
    @State private var size = SIMD3<Float>(repeating: 1)
    @State private var triangles = 0
    @State private var pickPoints: [SIMD3<Float>] = []
    @State private var loading: String? = "Buscando el modelo…"
    @State private var view: ReviewCamera = .perspective
    @State private var cameraToken = 0
    @State private var measuring = false
    @State private var loadID = 0

    init(engine: ScanEngine) {
        _engine = ObservedObject(wrappedValue: engine)
        _link = ObservedObject(wrappedValue: engine.link)
    }

    private var measure: MeasureController {
        if case .final = source { return finalMeasure }
        return engine.measure
    }

    var body: some View {
        ZStack {
            Color(white: 0.06).ignoresSafeArea()

            if case .photo(let m) = source, let url = link.photoViewURL(m) {
                PhotoWebView(url: url)
                    .ignoresSafeArea()
            } else if let content {
                ReviewSceneView(content: content, center: center, size: size, camera: view,
                                cameraToken: cameraToken, points: pickPoints, measure: measure)
                    .ignoresSafeArea()
            } else if loading == nil && !isPhoto {
                VStack(spacing: 8) {
                    Image(systemName: "cube.transparent").font(.largeTitle)
                    Text("Todavía no hay nada que revisar.")
                    Text("Escanea con la PC conectada; al terminar, aquí aparece el modelo final.")
                        .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
                }
                .foregroundStyle(.white)
                .padding(32)
            }

            if let loading {
                VStack(spacing: 10) {
                    ProgressView().tint(.white)
                    Text(loading).font(.footnote).foregroundStyle(.white)
                }
                .padding(18)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            }

            VStack(spacing: 10) {
                HStack(spacing: 8) {
                    sourceMenu
                    Spacer()
                    Button {
                        if case .photo(let m) = source {
                            // se mide sobre el modelo exacto, no sobre la vista foto-realista
                            measuring = true
                            select(.final(m))
                            return
                        }
                        measuring.toggle()
                        measure.active = measuring
                    } label: {
                        Label("Medir", systemImage: "ruler")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal, 14).padding(.vertical, 8)
                            .background(measuring ? AnyShapeStyle(Color(red: 0.96, green: 0.66, blue: 0.13))
                                                  : AnyShapeStyle(.ultraThinMaterial), in: Capsule())
                            .foregroundStyle(measuring ? Color.black : Color.white)
                    }
                    .disabled(content == nil && !isPhoto)
                    Button("Cerrar") { dismiss() }
                        .font(.subheadline.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                }

                Spacer()

                if isPhoto {
                    Text("Vista foto-realista entrenada en la PC. Gira con un dedo, acerca con dos; doble toque para acercarte a un punto. Para medir toca Medir (cambia al modelo exacto).")
                        .font(.caption)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                        .allowsHitTesting(false)
                } else if measuring {
                    MeasurePanel(measure: measure, onDone: { measuring = false; measure.active = false })
                } else if content != nil {
                    infoPanel
                }

                if !isPhoto {
                    HStack(spacing: 10) {
                        cameraButton("Perspectiva", .perspective)
                        cameraButton("Planta", .top)
                        cameraButton("Frente", .front)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .foregroundStyle(.white)
        }
        .onAppear(perform: start)
        .onDisappear {
            engine.setReviewing(false)
            engine.measure.active = false
            finalMeasure.active = false
        }
    }

    // MARK: Fuente

    private var isPhoto: Bool {
        if case .photo = source { return true }
        return false
    }

    private var sourceTitle: String {
        switch source {
        case .photo(let m): return "Foto-realista · \(m.nombre)"
        case .final(let m): return "Modelo final · \(m.nombre)"
        case .live: return "En vivo (PC)"
        case .cloud: return "Nube del iPhone"
        case nil: return "Revisar 3D"
        }
    }

    private var sourceMenu: some View {
        Menu {
            if models.contains(where: { $0.foto }) {
                Section("Vista foto-realista (PC)") {
                    ForEach(models.filter { $0.foto }) { m in
                        Button("\(m.nombre) · \(m.fecha)") { select(.photo(m)) }
                    }
                }
            }
            if !models.isEmpty {
                Section("Modelos para medir (PC)") {
                    ForEach(models) { m in
                        Button("\(m.nombre) · \(m.fecha)") { select(.final(m)) }
                    }
                }
            }
            if engine.pointCount > 0 { Button("Nube de puntos") { select(.cloud) } }
        } label: {
            HStack(spacing: 4) {
                Text(sourceTitle).lineLimit(1)
                Image(systemName: "chevron.down").font(.caption2)
            }
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
        }
    }

    private func start() {
        engine.setReviewing(true)       // la sesión de seguimiento sigue viva; solo se dejan de sumar puntos mientras miras
        engine.measure.onChange = { [weak engine] list in engine?.sendMeasures(list) }
        // Los modelos finales de la PC ya no se abren aquí: eran enormes (millones de triángulos y muchas texturas),
        // la app se cerraba por memoria al abrirlos, y la PC ya no guarda el USDZ (solo el FBX para Navisworks).
        models = []
        chooseDefault()
    }

    /// Lo más útil primero: el modelo final si es más nuevo que lo último escaneado; si no, lo en vivo.
    private func chooseDefault() {
        // solo la nube de puntos del iPhone (la misma que ves al escanear)
        if engine.pointCount > 0 {
            select(.cloud)
        } else {
            loading = nil
        }
    }

    private func select(_ s: Source) {
        measure.active = false
        source = s
        measure.active = measuring
        loadID += 1
        let id = loadID
        switch s {
        case .photo:
            measuring = false
            content = nil
            loading = nil
        case .final(let m): loadFinal(m, id: id)
        case .live: loadLive(id: id)
        case .cloud: loadCloud(id: id)
        }
    }

    private func show(_ node: SCNNode, points: [SIMD3<Float>] = [], tris: Int, id: Int) {
        guard id == loadID else { return }
        let (lo, hi) = ReviewSceneView.worldBounds(node)
        center = (lo + hi) / 2
        size = hi - lo
        triangles = tris
        pickPoints = points
        content = node
        cameraToken += 1
        let span = max(size.x, max(size.y, size.z))
        measure.markScale = max(0.3, min(4, span / 2))
        loading = nil
    }

    private func loadFinal(_ m: PCLink.PCModel, id: Int) {
        loading = "Descargando el modelo final (\(String(format: "%.0f", m.mb)) MB)…"
        finalMeasure.onChange = { [weak engine] list in engine?.sendMeasures(list, carpeta: m.carpeta) }
        finalMeasure.load([])
        link.fetchMeasures(m.carpeta) { list in
            if id == loadID { finalMeasure.load(list) }
        }
        link.downloadModel(m, progress: { p in
            if id == loadID, p < 1 { loading = "Descargando el modelo final… \(Int(p * 100)) %" }
        }, done: { url in
            guard id == loadID else { return }
            guard let url else {
                loading = nil
                if engine.lastLiveMesh != nil { select(.live) }
                return
            }
            loading = "Abriendo el modelo…"
            DispatchQueue.global(qos: .userInitiated).async {
                let scene = try? SCNScene(url: url, options: [.checkConsistency: false])
                let node = SCNNode()
                var tris = 0
                if let scene {
                    for c in scene.rootNode.childNodes { node.addChildNode(c) }
                    node.enumerateHierarchy { n, _ in
                        guard let g = n.geometry else { return }
                        tris += g.elements.reduce(0) { $0 + $1.primitiveCount }
                        for mat in g.materials {
                            mat.lightingModel = .constant      // se ve como en las fotos, sin sombras falsas
                            mat.isDoubleSided = true
                        }
                    }
                }
                DispatchQueue.main.async {
                    if scene == nil { loading = "No se pudo abrir el modelo."; return }
                    show(node, tris: tris, id: id)
                }
            }
        })
    }

    private func loadLive(id: Int) {
        guard let data = engine.lastLiveMesh else { loading = nil; return }
        loading = "Preparando…"
        DispatchQueue.global(qos: .userInitiated).async {
            let g = FrameCodec.liveMeshGeometry(data, tint: false)
            g?.materials.first?.transparency = 1
            let tris = g?.elements.first?.primitiveCount ?? 0
            DispatchQueue.main.async {
                guard let g else { loading = nil; return }
                show(SCNNode(geometry: g), tris: tris, id: id)
            }
        }
    }

    private func loadCloud(id: Int) {
        loading = "Preparando la nube…"
        engine.makeReviewData(mode: .real) { geo, _, pts in
            guard let geo else { loading = nil; return }
            show(SCNNode(geometry: geo), points: pts, tris: 0, id: id)
        }
    }

    // MARK: Paneles

    private var infoPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Ancho × Largo × Alto").font(.caption).foregroundStyle(.secondary)
            Text(String(format: "%.2f × %.2f × %.2f m", size.x, size.z, size.y))
                .font(.system(size: 20, weight: .semibold, design: .monospaced))
            Text(sourceNote).font(.caption).foregroundStyle(.secondary)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var sourceNote: String {
        switch source {
        case .photo: return ""
        case .final: return "Modelo final de la PC con fotos (\(triangles / 1000) k triángulos). Es el más exacto para medir."
        case .live: return "Vista en vivo de la PC (resolución de 1 cm). Para medir con exactitud usa el modelo final."
        case .cloud: return "Nube rápida del iPhone (sin PC). Para medir con exactitud procesa en la PC."
        case nil: return ""
        }
    }

    private func cameraButton(_ title: String, _ cam: ReviewCamera) -> some View {
        Button(title) { view = cam; cameraToken += 1 }
            .font(.footnote.weight(.semibold))
            .padding(.horizontal, 14).padding(.vertical, 9)
            .background(view == cam ? Color.white.opacity(0.28) : Color.white.opacity(0.1), in: Capsule())
    }
}

/// Página de la PC con la vista foto-realista (Gaussian Splatting), dentro de la app.
struct PhotoWebView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> WKWebView {
        let v = WKWebView(frame: .zero, configuration: WKWebViewConfiguration())
        v.isOpaque = false
        v.backgroundColor = UIColor(white: 0.06, alpha: 1)
        v.scrollView.isScrollEnabled = false
        v.scrollView.bounces = false
        v.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
        context.coordinator.url = url
        return v
    }

    func updateUIView(_ v: WKWebView, context: Context) {
        if context.coordinator.url != url {
            context.coordinator.url = url
            v.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var url: URL? }
}

/// Vista SceneKit: girar, acercar, doble toque para acercarse a un punto y toque para medir.
struct ReviewSceneView: UIViewRepresentable {
    let content: SCNNode
    let center: SIMD3<Float>
    let size: SIMD3<Float>
    let camera: ReviewView.ReviewCamera
    let cameraToken: Int
    let points: [SIMD3<Float>]
    let measure: MeasureController

    func makeUIView(context: Context) -> SCNView {
        let v = SCNView(frame: .zero)
        v.backgroundColor = UIColor(white: 0.06, alpha: 1)
        v.allowsCameraControl = true
        v.antialiasingMode = .multisampling4X
        v.scene = SCNScene()
        let holder = SCNNode()
        holder.name = "holder"
        v.scene?.rootNode.addChildNode(holder)
        let camNode = SCNNode()
        camNode.camera = SCNCamera()
        camNode.camera?.zNear = 0.005
        camNode.camera?.zFar = 500
        camNode.camera?.fieldOfView = 60
        camNode.camera?.projectionDirection = .vertical
        camNode.name = "cam"
        v.scene?.rootNode.addChildNode(camNode)
        v.pointOfView = camNode
        v.delegate = context.coordinator

        let c = context.coordinator
        c.view = v
        let double = UITapGestureRecognizer(target: c, action: #selector(Coordinator.doubleTapped(_:)))
        double.numberOfTapsRequired = 2
        let single = UITapGestureRecognizer(target: c, action: #selector(Coordinator.tapped(_:)))
        single.require(toFail: double)
        v.addGestureRecognizer(double)
        v.addGestureRecognizer(single)
        update(v, c)
        return v
    }

    func updateUIView(_ v: SCNView, context: Context) {
        update(v, context.coordinator)
    }

    private func update(_ v: SCNView, _ c: Coordinator) {
        c.points = points
        c.measure = measure
        guard let holder = v.scene?.rootNode.childNode(withName: "holder", recursively: false) else { return }
        if c.content !== content {
            holder.childNodes.forEach { $0.removeFromParentNode() }
            holder.addChildNode(content)
            c.content = content
        }
        if measure.root.parent !== v.scene?.rootNode {
            v.scene?.rootNode.childNodes.filter { $0.name == "medidas" }.forEach { $0.removeFromParentNode() }
            measure.root.removeFromParentNode()
            measure.root.name = "medidas"
            v.scene?.rootNode.addChildNode(measure.root)
        }
        if c.cameraToken != cameraToken {
            c.cameraToken = cameraToken
            placeCamera(v)
        }
    }

    private func placeCamera(_ v: SCNView) {
        guard let camNode = v.scene?.rootNode.childNode(withName: "cam", recursively: false) else { return }
        let c = SCNVector3(center.x, center.y, center.z)
        let span = max(0.5, max(size.x, max(size.y, size.z)))
        let dist = span * 1.3
        switch camera {
        case .perspective:
            camNode.position = SCNVector3(center.x + dist * 0.6, center.y + dist * 0.5, center.z + dist * 0.8)
            camNode.look(at: c)
        case .top:
            camNode.position = SCNVector3(center.x, center.y + dist * 1.2, center.z)
            camNode.look(at: c, up: SCNVector3(0, 0, -1), localFront: SCNVector3(0, 0, -1))
        case .front:
            camNode.position = SCNVector3(center.x, center.y, center.z + dist * 1.2)
            camNode.look(at: c)
        }
        v.pointOfView = camNode
        v.defaultCameraController.target = c
    }

    /// Caja envolvente en coordenadas del mundo (incluye todos los hijos).
    static func worldBounds(_ node: SCNNode) -> (SIMD3<Float>, SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        node.enumerateHierarchy { n, _ in
            guard n.geometry != nil else { return }
            let (a, b) = n.boundingBox
            for x in [a.x, b.x] { for y in [a.y, b.y] { for z in [a.z, b.z] {
                let p = n.simdConvertPosition(SIMD3<Float>(x, y, z), to: nil)
                lo = simd_min(lo, p); hi = simd_max(hi, p)
            } } }
        }
        if lo.x > hi.x { return (.zero, .zero) }
        return (lo, hi)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, SCNSceneRendererDelegate {
        weak var view: SCNView?
        var content: SCNNode?
        var cameraToken = -1
        var points: [SIMD3<Float>] = []
        weak var measure: MeasureController?
        private var lastScaleCheck: TimeInterval = 0
        private var lastScale: Float = 0

        /// Punto 3D bajo el dedo: sobre la superficie del modelo (exacto) o, en la nube, el punto más cercano.
        func point(at p: CGPoint) -> SIMD3<Float>? {
            guard let v = view else { return nil }
            if !points.isEmpty { return PointPicker.pick(at: p, in: v, points: points) }
            guard let content else { return nil }
            let hits = v.hitTest(p, options: [.searchMode: SCNHitTestSearchMode.closest.rawValue,
                                              .rootNode: content, .ignoreHiddenNodes: true])
            return hits.first?.simdWorldCoordinates
        }

        @objc func tapped(_ g: UITapGestureRecognizer) {
            guard let v = view, let m = measure, m.active else { return }
            if let p = point(at: g.location(in: v)) { m.tap(p) } else { m.missed = true }
        }

        /// Doble toque: acercarse a ese punto (cada doble toque acerca más) y girar alrededor de él.
        @objc func doubleTapped(_ g: UITapGestureRecognizer) {
            guard let v = view, let pov = v.pointOfView, let hit = point(at: g.location(in: v)) else { return }
            let from = pov.simdWorldPosition
            let dist = simd_distance(from, hit)
            let newDist = max(0.06, min(0.45, dist * 0.45))
            let dir = simd_normalize(from - hit)
            SCNTransaction.begin()
            SCNTransaction.animationDuration = 0.35
            pov.simdWorldPosition = hit + dir * newDist
            pov.simdLook(at: hit)
            SCNTransaction.commit()
            v.defaultCameraController.target = SCNVector3(hit.x, hit.y, hit.z)
        }

        /// Marcas de medida de tamaño parecido en pantalla, de lejos o de cerca.
        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard time - lastScaleCheck > 0.25, let v = view, let pov = v.pointOfView else { return }
            lastScaleCheck = time
            let t = v.defaultCameraController.target
            let d = simd_distance(pov.simdWorldPosition, SIMD3<Float>(t.x, t.y, t.z))
            let s = max(0.08, min(6, d / 0.8))
            if lastScale == 0 || abs(s - lastScale) / lastScale > 0.25 {
                lastScale = s
                DispatchQueue.main.async { [weak self] in self?.measure?.markScale = s }
            }
        }
    }
}
