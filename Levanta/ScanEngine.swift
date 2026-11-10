import ARKit
import SceneKit
import Combine

/// Motor de escaneo: lee la profundidad del LiDAR en cada cuadro, la convierte
/// en puntos 3D del mundo con color y la acumula en una rejilla de vóxeles
/// (un punto promedio por cubito), lo que reduce ruido y duplicados.
final class ScanEngine: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: Estado visible en la interfaz (solo se modifica en el hilo principal)
    @Published var isScanning = false
    @Published var pointCount = 0
    @Published var trackingText = "Preparando…"
    @Published var trackingOK = false
    @Published var statusMessage: String?
    @Published var lidarAvailable = true
    @Published var showCloud = true
    @Published var colorMode: ColorMode = .coverage {
        didSet { tintLiveMesh = colorMode == .coverage }
    }
    /// Distancia (m) a lo que apunta el centro de la pantalla; nil si no hay lectura.
    @Published var centerDistance: Float?

    enum ColorMode: String, CaseIterable, Identifiable {
        case coverage = "Cobertura"
        case real = "Color real"
        var id: String { rawValue }
    }

    /// Resumen de la nube para la revisión 3D.
    struct CloudStats {
        var points: Int
        var size: SIMD3<Float>     // ancho (x), alto (y), largo (z) en metros
        var center: SIMD3<Float>
        var weak: Int              // puntos con una sola lectura
    }

    // Ajustes (se aplican al reiniciar o al empezar un escaneo nuevo)
    @Published var voxelSizeCm: Double = 0.5
    @Published var maxDepth: Double = 4.0
    @Published var highConfidenceOnly = true
    @Published var minObservations: Int = 2
    @Published var smoothDepth = false
    /// Quitar personas en tiempo real (segmentación con IA de ARKit, en el chip del iPhone).
    @Published var removePeople = true {
        didSet {
            let on = removePeople
            queue.async { self.activeRemovePeople = on }
            startSession(reset: false)
        }
    }
    /// Hay una persona frente a la cámara y se está ignorando.
    @Published var personInView = false
    @Published var peopleRemovalAvailable = true
    /// Guardar cuadros RGB-D (foto + profundidad + posición) para la fusión de alta calidad.
    @Published var recordRGBD = true
    @Published var keyframeCount = 0
    /// Exposición corta y balance de blancos fijos al escanear (fotos nítidas, color parejo).
    @Published var sharpCapture = true
    /// Te estás moviendo o girando demasiado rápido para fotos nítidas.
    @Published var tooFast = false
    /// Enviar el escaneo a la PC por red local.
    @Published var streamToPC = false {
        didSet { link.setEnabled(streamToPC) }
    }
    @Published var liveMeshTriangles = 0

    let link = PCLink()
    let liveMeshNode = SCNNode()
    let measure = MeasureController()

    let pointsNode = SCNNode()
    private(set) var sceneView: ARSCNView?

    // MARK: Estado interno (solo se toca dentro de `queue`)
    private let queue = DispatchQueue(label: "levanta.scan", qos: .userInitiated)
    private var keyToIndex: [Int64: Int32] = [:]
    private var sums: [SIMD3<Float>] = []
    private var colorSums: [SIMD3<Float>] = []
    private var counts: [UInt16] = []
    private var scanning = false
    private var lastProcessTime: TimeInterval = 0
    private var activeVoxel: Float = 0.005
    private var activeMaxDepth: Float = 4.0
    private var activeMinConfidence: UInt8 = 2
    private var activeMinObs: UInt16 = 2
    private var activeSmooth = false
    private var activeRemovePeople = true
    private var personState = false
    private var personFrames = 0
    private let recorder = KeyframeRecorder()
    private var activeRecordRGBD = true
    private var tracking = false
    private var previewTimer: Timer?
    private var previewBusy = false
    private var lastDistanceTime: TimeInterval = 0
    // movimiento y envío a la PC (solo en `queue`)
    private var lastMotionPose: simd_float4x4?
    private var lastMotionTime: TimeInterval = 0
    private var speedEMA: Float = 0
    private var turnEMA: Float = 0
    private var tooFastState = false
    private var lastStreamTime: TimeInterval = 0
    private var streamIndex = 0
    private var sessionName = ""        // solo en el hilo principal
    private var tintLiveMesh = true

    private let maxVoxels = 6_000_000
    private let processInterval: TimeInterval = 0.09   // ~11 cuadros por segundo
    private let pixelStep = 1                           // todos los píxeles del mapa de profundidad

    // MARK: Vista AR

    func makeView() -> ARSCNView {
        if let v = sceneView { return v }
        let view = ARSCNView(frame: .zero)
        view.automaticallyUpdatesLighting = false
        view.rendersCameraGrain = false
        view.scene.rootNode.addChildNode(pointsNode)
        liveMeshNode.renderingOrder = -1
        view.scene.rootNode.addChildNode(liveMeshNode)
        link.onMesh = { [weak self] data in self?.receiveLiveMesh(data) }
        link.onConnected = { [weak self] in
            guard let self else { return }
            if self.isScanning { self.link.sendStart(name: self.currentSessionName()) }
            if !self.measure.measures.isEmpty { self.sendMeasures(self.measure.measures) }
        }
        measure.onChange = { [weak self] list in self?.sendMeasures(list) }
        view.session.delegate = self
        view.session.delegateQueue = queue
        sceneView = view
        startSession(reset: true)
        startPreviewTimer()
        return view
    }

    private func startSession(reset: Bool) {
        guard let view = sceneView else { return }
        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                config.frameSemantics.insert(.smoothedSceneDepth)
            }
        } else {
            lidarAvailable = false
            statusMessage = "Este dispositivo no tiene LiDAR."
        }
        let peopleSemantics: ARConfiguration.FrameSemantics = [.sceneDepth, .personSegmentationWithDepth]
        if removePeople {
            if ARWorldTrackingConfiguration.supportsFrameSemantics(peopleSemantics) {
                config.frameSemantics.insert(.personSegmentationWithDepth)
                peopleRemovalAvailable = true
            } else {
                peopleRemovalAvailable = false
            }
        }
        config.worldAlignment = .gravity
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        view.session.run(config, options: options)
    }

    // MARK: Controles

    func toggleScanning() {
        let start = !isScanning
        isScanning = start
        if start && sharpCapture { CameraTuning.apply(sharp: true) }
        if start && link.isConnected { link.sendStart(name: currentSessionName()) }
        let voxel = Float(voxelSizeCm / 100.0)
        let depth = Float(maxDepth)
        let conf: UInt8 = highConfidenceOnly ? 2 : 1
        let minObs = UInt16(max(1, minObservations))
        let smooth = smoothDepth
        let rgbd = recordRGBD
        queue.async {
            self.activeSmooth = smooth
            self.activeRecordRGBD = rgbd
            if start && rgbd && !self.recorder.isActive {
                do {
                    try self.recorder.begin()
                } catch {
                    DispatchQueue.main.async {
                        self.statusMessage = "No se pudo iniciar la grabación RGB-D: \(error.localizedDescription)"
                    }
                }
            }
            if start && self.sums.isEmpty {
                self.activeVoxel = voxel
            }
            self.activeMaxDepth = depth
            self.activeMinConfidence = conf
            self.activeMinObs = minObs
            self.scanning = start
        }
    }

    func reset() {
        if link.isConnected, let d = FrameCodec.pack(kind: 4, header: [:]) {
            link.send(d, important: true)            // la PC descarta lo recibido de este escaneo
        }
        isScanning = false
        measure.clear()
        pointCount = 0
        pointsNode.geometry = nil
        liveMeshNode.geometry = nil
        liveMeshTriangles = 0
        tooFast = false
        CameraTuning.apply(sharp: false)
        sessionName = ""
        pointsNode.isHidden = false
        queue.async {
            self.streamIndex = 0
            self.lastMotionPose = nil
        }
        let voxel = Float(voxelSizeCm / 100.0)
        queue.async {
            self.scanning = false
            self.recorder.discard()
            DispatchQueue.main.async { self.keyframeCount = 0 }
            self.keyToIndex.removeAll(keepingCapacity: false)
            self.sums.removeAll(keepingCapacity: false)
            self.colorSums.removeAll(keepingCapacity: false)
            self.counts.removeAll(keepingCapacity: false)
            self.activeVoxel = voxel
            DispatchQueue.main.async { self.startSession(reset: true) }
        }
    }

    func updateMinObservations(_ value: Int) {
        let v = UInt16(max(1, value))
        queue.async { self.activeMinObs = v }
    }

    // MARK: ARSessionDelegate (se llama en `queue`)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if frame.timestamp - lastDistanceTime >= 0.25 {
            lastDistanceTime = frame.timestamp
            let d = centerDepth(frame)
            DispatchQueue.main.async { self.centerDistance = d }
        }
        guard scanning, tracking else { return }
        guard frame.timestamp - lastProcessTime >= processInterval else { return }
        lastProcessTime = frame.timestamp
        let chosen = activeSmooth ? (frame.smoothedSceneDepth ?? frame.sceneDepth)
                                  : (frame.sceneDepth ?? frame.smoothedSceneDepth)
        var mask: [Bool]?
        if activeRemovePeople, let dm = chosen?.depthMap {
            mask = FrameCodec.personMask(frame, w: CVPixelBufferGetWidth(dm), h: CVPixelBufferGetHeight(dm))
        }
        updatePersonIndicator(mask != nil)
        integrate(frame, mask: mask)
        updateMotion(frame)
        var savedKeyframe = false
        if activeRecordRGBD && recorder.isActive, let depth = chosen,
           recorder.consider(frame: frame, depth: depth, mask: mask) {
            savedKeyframe = true
            let n = recorder.count
            DispatchQueue.main.async { self.keyframeCount = n }
        }
        if link.isConnected, let depth = chosen,
           savedKeyframe || frame.timestamp - lastStreamTime >= 0.25 {
            lastStreamTime = frame.timestamp
            streamFrame(frame, depth: depth, keyframe: savedKeyframe, mask: mask)
        }
    }

    /// Indicador "persona ignorada" con un poco de histéresis para que no parpadee.
    private func updatePersonIndicator(_ seen: Bool) {
        personFrames = seen ? min(personFrames + 1, 6) : max(personFrames - 1, 0)
        let state = personState ? personFrames > 0 : personFrames >= 2
        if state != personState {
            personState = state
            DispatchQueue.main.async { self.personInView = state }
        }
    }

    /// Velocidad de avance y de giro (suavizadas) para avisar cuando hay que ir más despacio.
    private func updateMotion(_ frame: ARFrame) {
        let T = frame.camera.transform
        defer { lastMotionPose = T; lastMotionTime = frame.timestamp }
        guard let L = lastMotionPose else { return }
        let dt = Float(frame.timestamp - lastMotionTime)
        guard dt > 0.01 && dt < 1 else { return }
        let dp = simd_distance(SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z),
                               SIMD3<Float>(L.columns.3.x, L.columns.3.y, L.columns.3.z))
        var ang = (simd_quatf(L).inverse * simd_quatf(T)).angle
        if ang > .pi { ang = 2 * .pi - ang }
        speedEMA = 0.7 * speedEMA + 0.3 * (dp / dt)
        turnEMA = 0.7 * turnEMA + 0.3 * (ang / dt * 180 / .pi)
        let fast = speedEMA > 0.45 || turnEMA > 32
        let calm = speedEMA < 0.30 && turnEMA < 22
        if fast != tooFastState && (fast || calm) {
            tooFastState = fast
            DispatchQueue.main.async { self.tooFast = fast }
        }
    }

    /// Envía un cuadro completo (foto a resolución completa, profundidad, confianza y posición) a la PC.
    private func streamFrame(_ frame: ARFrame, depth: ARDepthData, keyframe: Bool, mask: [Bool]?) {
        guard let jpg = FrameCodec.jpeg(frame, scale: 1.0, quality: 0.85),
              let dc = FrameCodec.depthAndConfidence(depth, mask: mask) else { return }
        var h = FrameCodec.header(frame, index: streamIndex, scale: 1.0)
        h["dw"] = dc.w
        h["dh"] = dc.h
        h["kf"] = keyframe
        h["nj"] = jpg.count
        h["nd"] = dc.depth.count
        h["nc"] = dc.conf.count
        guard let msg = FrameCodec.pack(kind: 1, header: h, payload: [jpg, dc.depth, dc.conf]) else { return }
        streamIndex += 1
        link.send(msg, important: keyframe)
    }

    private func currentSessionName() -> String {
        if sessionName.isEmpty {
            let f = DateFormatter()
            f.locale = Locale(identifier: "es_MX")
            f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
            sessionName = f.string(from: Date())
        }
        return sessionName
    }

    /// Envía las medidas hechas en el iPhone a la PC (mismas coordenadas que el modelo).
    func sendMeasures(_ list: [MeasureController.Measure]) {
        guard link.isConnected else { return }
        let items: [[String: Any]] = list.map { m in
            ["id": m.id, "nombre": m.name, "a": [m.a.x, m.a.y, m.a.z], "b": [m.b.x, m.b.y, m.b.z]]
        }
        if let d = FrameCodec.pack(kind: 5, header: ["medidas": items]) { link.send(d, important: true) }
    }

    /// Termina el escaneo y le pide a la PC que procese todo en alta calidad.
    func finishOnPC() {
        if isScanning { toggleScanning() }
        link.sendEnd()
        sessionName = ""
        queue.async { self.streamIndex = 0 }
    }

    private func receiveLiveMesh(_ data: Data) {
        let tint = tintLiveMesh
        DispatchQueue.global(qos: .userInitiated).async {
            let g = FrameCodec.liveMeshGeometry(data, tint: tint)
            let tris = g?.elements.first?.primitiveCount ?? 0
            DispatchQueue.main.async {
                self.liveMeshNode.geometry = g
                self.liveMeshTriangles = tris
                self.pointsNode.isHidden = g != nil
            }
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        let ok: Bool
        let text: String
        switch camera.trackingState {
        case .normal:
            ok = true; text = "Seguimiento estable"
        case .notAvailable:
            ok = false; text = "Seguimiento no disponible"
        case .limited(let reason):
            ok = false
            switch reason {
            case .initializing: text = "Iniciando: mueve el iPhone despacio"
            case .excessiveMotion: text = "Muy rápido: más despacio"
            case .insufficientFeatures: text = "Poca textura: apunta a otra zona"
            case .relocalizing: text = "Reubicando…"
            @unknown default: text = "Seguimiento limitado"
            }
        }
        tracking = ok
        DispatchQueue.main.async {
            self.trackingOK = ok
            self.trackingText = text
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async {
            self.statusMessage = "Error de AR: \(error.localizedDescription)"
        }
    }

    // MARK: Integración de profundidad

    private func integrate(_ frame: ARFrame, mask: [Bool]?) {
        let chosen = activeSmooth ? (frame.smoothedSceneDepth ?? frame.sceneDepth)
                                  : (frame.sceneDepth ?? frame.smoothedSceneDepth)
        guard let depthData = chosen,
              let confMap = depthData.confidenceMap else { return }
        let depthMap = depthData.depthMap
        let image = frame.capturedImage

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(confMap, .readOnly)
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(confMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(image, .readOnly)
        }

        guard let dBase = CVPixelBufferGetBaseAddress(depthMap),
              let cBase = CVPixelBufferGetBaseAddress(confMap),
              CVPixelBufferGetPlaneCount(image) >= 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(image, 1) else { return }

        let dw = CVPixelBufferGetWidth(depthMap)
        let dh = CVPixelBufferGetHeight(depthMap)
        let dRow = CVPixelBufferGetBytesPerRow(depthMap)
        let cRow = CVPixelBufferGetBytesPerRow(confMap)
        let imgW = CVPixelBufferGetWidthOfPlane(image, 0)
        let imgH = CVPixelBufferGetHeightOfPlane(image, 0)
        let yRow = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let uvRow = CVPixelBufferGetBytesPerRowOfPlane(image, 1)

        // Intrínsecos de la cámara escalados a la resolución del mapa de profundidad
        let K = frame.camera.intrinsics
        let sx = Float(dw) / Float(imgW)
        let sy = Float(dh) / Float(imgH)
        let fx = K[0][0] * sx
        let fy = K[1][1] * sy
        // Centro óptico escalado respecto al centro del píxel (convención de píxel centrado)
        let cx = (K[2][0] + 0.5) * sx - 0.5
        let cy = (K[2][1] + 0.5) * sy - 0.5
        let T = frame.camera.transform

        let inv = 1.0 / activeVoxel
        let maxD = activeMaxDepth
        let minConf = activeMinConfidence
        let half: Int64 = 1 << 20
        let limit: Float = Float(half - 2)

        if sums.count >= maxVoxels {
            scanning = false
            DispatchQueue.main.async {
                self.isScanning = false
                self.statusMessage = "Límite de memoria alcanzado. Guarda el escaneo."
            }
            return
        }

        for v in stride(from: 0, to: dh, by: pixelStep) {
            let dPtr = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
            let cPtr = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
            let iv = min(imgH - 1, Int((Float(v) + 0.5) / sy))
            for u in stride(from: 0, to: dw, by: pixelStep) {
                if cPtr[u] < minConf { continue }
                if let mask, mask[v * dw + u] { continue }       // persona: no se integra
                let d = dPtr[u]
                if !d.isFinite || d < 0.15 || d > maxD { continue }

                // Píxel -> espacio de cámara de ARKit (x derecha, y arriba, z hacia atrás)
                let xc = (Float(u) - cx) * d / fx
                let yc = (Float(v) - cy) * d / fy
                let pw = T * SIMD4<Float>(xc, -yc, -d, 1)
                let p = SIMD3<Float>(pw.x, pw.y, pw.z)

                let fxv = (p.x * inv).rounded(.down)
                let fyv = (p.y * inv).rounded(.down)
                let fzv = (p.z * inv).rounded(.down)
                if abs(fxv) > limit || abs(fyv) > limit || abs(fzv) > limit { continue }
                let key = ((Int64(fxv) + half) << 42) | ((Int64(fyv) + half) << 21) | (Int64(fzv) + half)

                // Color desde la imagen YCbCr (BT.601, rango completo)
                let iu = min(imgW - 1, Int((Float(u) + 0.5) / sx))
                let yv = Float(yBase.load(fromByteOffset: iv * yRow + iu, as: UInt8.self))
                let uvOff = (iv / 2) * uvRow + (iu / 2) * 2
                let cb = Float(uvBase.load(fromByteOffset: uvOff, as: UInt8.self)) - 128
                let cr = Float(uvBase.load(fromByteOffset: uvOff + 1, as: UInt8.self)) - 128
                let r = min(255, max(0, yv + 1.402 * cr))
                let g = min(255, max(0, yv - 0.344136 * cb - 0.714136 * cr))
                let b = min(255, max(0, yv + 1.772 * cb))
                let c = SIMD3<Float>(r, g, b)

                if let idx = keyToIndex[key] {
                    let i = Int(idx)
                    if counts[i] < 50_000 {
                        sums[i] += p
                        colorSums[i] += c
                        counts[i] += 1
                    }
                } else {
                    keyToIndex[key] = Int32(sums.count)
                    sums.append(p)
                    colorSums.append(c)
                    counts.append(1)
                }
            }
        }
    }

    // MARK: Vista previa en vivo

    private func startPreviewTimer() {
        previewTimer?.invalidate()
        previewTimer = Timer.scheduledTimer(withTimeInterval: 0.8, repeats: true) { [weak self] _ in
            self?.refreshPreview()
        }
    }

    private func refreshPreview() {
        guard !previewBusy else { return }
        previewBusy = true
        let show = showCloud
        let mode = colorMode
        queue.async {
            let minObs = self.activeMinObs
            let total = self.counts.reduce(0) { $0 + ($1 >= minObs ? 1 : 0) }
            let geometry = show ? self.buildGeometry(mode: mode, maxPoints: 400_000,
                                                     pointSize: 5, maxRadius: 4) : nil
            DispatchQueue.main.async {
                self.pointCount = total
                self.pointsNode.geometry = geometry
                self.previewBusy = false
            }
        }
    }

    /// Geometría y resumen para la revisión 3D fuera de la cámara.
    /// También devuelve los puntos confiables (para tocar y medir sobre ellos).
    func makeReviewData(mode: ColorMode,
                        completion: @escaping (SCNGeometry?, CloudStats, [SIMD3<Float>]) -> Void) {
        queue.async {
            let geometry = self.buildGeometry(mode: mode, maxPoints: 1_500_000,
                                              pointSize: 4, maxRadius: 3)
            let stats = self.computeStats()
            let pick = self.reliablePoints(maxPoints: 1_500_000)
            DispatchQueue.main.async { completion(geometry, stats, pick) }
        }
    }

    /// Puntos con suficientes lecturas (los mismos que se guardan), promediados por vóxel.
    private func reliablePoints(maxPoints: Int) -> [SIMD3<Float>] {
        let n = sums.count
        guard n > 0 else { return [] }
        let minObs = activeMinObs
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(min(n, maxPoints))
        let step = max(1, n / maxPoints)
        var i = 0
        while i < n {
            let c = counts[i]
            if c >= minObs { out.append(sums[i] / Float(c)) }
            i += step
        }
        return out
    }

    private func computeStats() -> CloudStats {
        let minObs = activeMinObs
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var kept = 0
        var weak = 0
        for i in 0..<sums.count {
            let c = counts[i]
            if c == 1 { weak += 1 }
            guard c >= minObs else { continue }
            let p = sums[i] / Float(c)
            lo = simd_min(lo, p)
            hi = simd_max(hi, p)
            kept += 1
        }
        if kept == 0 {
            return CloudStats(points: 0, size: .zero, center: .zero, weak: weak)
        }
        return CloudStats(points: kept, size: hi - lo, center: (hi + lo) / 2, weak: weak)
    }

    /// Color según cuántas veces se leyó cada punto: rojo = 1, ámbar = 2–3, verde = 4 o más.
    private func coverageColor(_ count: UInt16) -> SIMD3<Float> {
        switch count {
        case 0...1: return SIMD3<Float>(0.95, 0.24, 0.20)
        case 2...3: return SIMD3<Float>(0.98, 0.70, 0.15)
        default:    return SIMD3<Float>(0.25, 0.85, 0.40)
        }
    }

    /// Construye la geometría de puntos. En modo cobertura se incluyen también los puntos
    /// de una sola lectura (en rojo) para que se vea el ruido y lo que falta reforzar.
    private func buildGeometry(mode: ColorMode, maxPoints: Int,
                               pointSize: CGFloat, maxRadius: CGFloat) -> SCNGeometry? {
        let n = sums.count
        guard n > 0 else { return nil }
        let step = max(1, n / maxPoints)
        let minObs: UInt16 = mode == .coverage ? 1 : activeMinObs
        var verts: [SIMD3<Float>] = []
        var cols: [SIMD3<Float>] = []
        verts.reserveCapacity(n / step + 1)
        cols.reserveCapacity(n / step + 1)
        var i = 0
        while i < n {
            let cnt = counts[i]
            if cnt >= minObs {
                let k = Float(cnt)
                verts.append(sums[i] / k)
                if mode == .coverage {
                    cols.append(coverageColor(cnt))
                } else {
                    cols.append(colorSums[i] / (k * 255))
                }
            }
            i += step
        }
        guard !verts.isEmpty else { return nil }

        let vStride = MemoryLayout<SIMD3<Float>>.stride   // 16 bytes (incluye relleno)
        let vData = verts.withUnsafeBufferPointer { Data(buffer: $0) }
        let cData = cols.withUnsafeBufferPointer { Data(buffer: $0) }
        let vSource = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: verts.count,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 0, dataStride: vStride)
        let cSource = SCNGeometrySource(data: cData, semantic: .color, vectorCount: cols.count,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 0, dataStride: vStride)
        let indices = (0..<UInt32(verts.count)).map { $0 }
        let element = SCNGeometryElement(indices: indices, primitiveType: .point)
        element.pointSize = pointSize
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = maxRadius

        let geometry = SCNGeometry(sources: [vSource, cSource], elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = UIColor.white
        material.isDoubleSided = true
        geometry.materials = [material]
        return geometry
    }

    /// Profundidad en el centro de la imagen (mediana de una ventana de 5×5), en metros.
    private func centerDepth(_ frame: ARFrame) -> Float? {
        guard let depthData = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
        let map = depthData.depthMap
        CVPixelBufferLockBaseAddress(map, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }
        let w = CVPixelBufferGetWidth(map)
        let h = CVPixelBufferGetHeight(map)
        let row = CVPixelBufferGetBytesPerRow(map)
        var values: [Float] = []
        values.reserveCapacity(25)
        for v in (h / 2 - 2)...(h / 2 + 2) {
            let ptr = base.advanced(by: v * row).assumingMemoryBound(to: Float32.self)
            for u in (w / 2 - 2)...(w / 2 + 2) {
                let d = ptr[u]
                if d.isFinite && d > 0 { values.append(d) }
            }
        }
        guard !values.isEmpty else { return nil }
        values.sort()
        return values[values.count / 2]
    }

    // MARK: Exportar

    /// Cierra la grabación RGB-D actual y la empaqueta en partes .zip listas para compartir.
    func finishRGBD(completion: @escaping (Result<[URL], Error>) -> Void) {
        queue.async {
            guard let dir = self.recorder.finish() else {
                DispatchQueue.main.async { completion(.success([])) }
                return
            }
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let parts = try KeyframeRecorder.packageParts(dir)
                    DispatchQueue.main.async {
                        self.keyframeCount = 0
                        completion(.success(parts))
                    }
                } catch {
                    DispatchQueue.main.async { completion(.failure(error)) }
                }
            }
        }
    }

    /// Guarda la nube como PLY binario (x, y, z en metros + color) en Documentos/Escaneos.
    func exportPLY(completion: @escaping (Result<URL, Error>) -> Void) {
        queue.async {
            let minObs = self.activeMinObs
            var keep: [Int] = []
            keep.reserveCapacity(self.counts.count)
            for (i, c) in self.counts.enumerated() where c >= minObs { keep.append(i) }

            guard !keep.isEmpty else {
                DispatchQueue.main.async {
                    completion(.failure(NSError(domain: "Levanta", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "No hay puntos para guardar."])))
                }
                return
            }

            let header = """
            ply
            format binary_little_endian 1.0
            comment Levanta LiDAR scan, units meters, Y up
            element vertex \(keep.count)
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header

            """
            var data = Data(header.utf8)
            var body = [UInt8](repeating: 0, count: keep.count * 15)
            var o = 0
            @inline(__always) func putFloat(_ f: Float) {
                let b = f.bitPattern
                body[o] = UInt8(b & 0xff)
                body[o + 1] = UInt8((b >> 8) & 0xff)
                body[o + 2] = UInt8((b >> 16) & 0xff)
                body[o + 3] = UInt8((b >> 24) & 0xff)
                o += 4
            }
            for i in keep {
                let k = Float(self.counts[i])
                let p = self.sums[i] / k
                let c = self.colorSums[i] / k
                putFloat(p.x); putFloat(p.y); putFloat(p.z)
                body[o] = UInt8(min(255, max(0, c.x.rounded())))
                body[o + 1] = UInt8(min(255, max(0, c.y.rounded())))
                body[o + 2] = UInt8(min(255, max(0, c.z.rounded())))
                o += 3
            }
            data.append(contentsOf: body)

            do {
                let url = try ScanStore.newScanURL()
                try data.write(to: url, options: .atomic)
                DispatchQueue.main.async { completion(.success(url)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }
}

/// Archivos de escaneo en Documentos/Escaneos (visibles también en la app Archivos).
enum ScanStore {
    static func folder() throws -> URL {
        let docs = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                               appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent("Escaneos", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    static func newScanURL() throws -> URL {
        let f = DateFormatter()
        f.locale = Locale(identifier: "es_MX")
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return try folder().appendingPathComponent("Escaneo_\(f.string(from: Date())).ply")
    }

    static func list() -> [URL] {
        guard let dir = try? folder(),
              let items = try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey]) else { return [] }
        return items.filter { ["ply", "zip"].contains($0.pathExtension.lowercased()) }.sorted { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return da > db
        }
    }

    static func sizeText(_ url: URL) -> String {
        let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}
