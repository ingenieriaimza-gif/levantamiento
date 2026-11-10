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
    /// Configuración fija (la mejor para levantamientos con la PC): no hay ajustes que mover.
    /// La nube del iPhone es solo una vista previa a 1 cm; la calidad final la hace la PC.
    @Published var voxelSizeCm: Double = 0.5
    @Published var maxDepth: Double = 3.5
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
    /// Borrar ruido: si en una vista nueva el LiDAR ve a través de un punto que ya estaba,
    /// ese punto era ruido (o algo que se movió) y se elimina.
    @Published var cleanNoise = true {
        didSet { let on = cleanNoise; queue.async { self.activeCleanNoise = on } }
    }
    /// Puntos borrados por la limpieza en este escaneo.
    @Published var removedPoints = 0
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
    /// El iPhone está quieto: ya se tomaron las lecturas de esta vista y no se agregan más puntos.
    @Published var holdingStill = false
    /// Enviar el escaneo a la PC por red local.
    @Published var streamToPC = true {
        didSet { link.setEnabled(streamToPC) }
    }
    @Published var liveMeshTriangles = 0

    let link = PCLink()
    let liveMeshNode = SCNNode()
    let measure = MeasureController()
    /// Piezas enfocadas (zonas de detalle de 2 mm).
    let focus = FocusController()
    /// Indicación mientras escaneas: acércate, aléjate, lista…
    @Published var guide: ScanGuide?
    /// Avance de cada pieza enfocada (0–100; 100 = lista).
    @Published var zoneProgress: [String: Int] = [:]
    /// Decir las indicaciones en voz alta.
    @Published var voiceGuide: Bool = (UserDefaults.standard.object(forKey: "vozGuia") as? Bool) ?? true {
        didSet { UserDefaults.standard.set(voiceGuide, forKey: "vozGuia") }
    }
    private let coach = ScanCoach()
    private let voice = GuideVoice()
    private var lastCenterDepth: Float?

    // puntos finos (2 mm) dentro de las piezas enfocadas (solo en `queue`)
    private let fineVoxel: Float = 0.002
    private var fineKeyToIndex: [Int64: Int32] = [:]
    private var fineSums: [SIMD3<Float>] = []
    private var fineColorSums: [SIMD3<Float>] = []
    private var fineCounts: [UInt16] = []
    private let maxFineVoxels = 3_000_000

    let pointsNode = SCNNode()
    private(set) var sceneView: ARSCNView?

    // MARK: Estado interno (solo se toca dentro de `queue`)
    private let queue = DispatchQueue(label: "levanta.scan", qos: .userInitiated)
    private var keyToIndex: [Int64: Int32] = [:]
    private var sums: [SIMD3<Float>] = []
    private var colorSums: [SIMD3<Float>] = []
    private var counts: [UInt16] = []
    private var keys: [Int64] = []          // llave de cada vóxel (para borrarlo del índice)
    private var misses: [UInt16] = []       // veces que el LiDAR vio a través del punto
    private var nearCounts: [UInt16] = []   // lecturas tomadas de cerca (≤ 1.5 m): las de buena calidad
    private var closeCounts: [UInt16] = []  // lecturas muy de cerca (≤ 1.0 m): las de calidad excelente
    private var firstDir: [SIMD3<Int8>] = [] // dirección de la primera lectura muy de cerca (0,0,0 = ninguna)
    private var diverse: [Bool] = []        // ya se leyó muy de cerca desde ángulos distintos (≥ 30°)
    private var activeCleanNoise = true
    private var carveCursor = 0
    private var removedTotal = 0
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
    // pausa cuando no te mueves (solo en `queue`)
    private var anchorPose: simd_float4x4?
    private var stillReadings = 0
    private var stillState = false
    /// Lecturas que se toman de una misma vista antes de pausar (las suficientes para confirmar
    /// cada punto: "lecturas por punto" es 2 por omisión).
    private let readingsPerView = 4
    /// Movimiento mínimo para que cuente como vista nueva.
    private let newViewMove: Float = 0.02          // 2 cm
    private let newViewTurn: Float = 3 * .pi / 180  // 3°
    /// Cerca de una pieza enfocada: vistas nuevas con menos movimiento y más fotos.
    private var nearFocus = false
    private var lastStreamTime: TimeInterval = 0
    private var streamIndex = 0
    private var sessionName = ""        // solo en el hilo principal
    private var tintLiveMesh = true

    private let maxVoxels = 7_000_000
    /// Ya se limpió y sigue lleno: la vista previa deja de crecer (las fotos para la PC siguen).
    private var cloudFull = false
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
        view.scene.rootNode.addChildNode(focus.root)
        view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(tappedCamera(_:))))
        focus.onChange = { [weak self] zones in self?.zonesChanged(zones) }
        coach.onGuide = { [weak self] g, prog in
            guard let self, self.isScanning else { return }
            if self.guide != g { self.guide = g }
            if self.zoneProgress != prog { self.zoneProgress = prog }
            self.voice.handle(g, enabled: self.voiceGuide)
        }
        link.onMesh = { [weak self] data in self?.receiveLiveMesh(data) }
        link.onConnected = { [weak self] in
            guard let self else { return }
            if self.isScanning { self.link.sendStart(name: self.currentSessionName()) }
            if !self.measure.measures.isEmpty { self.sendMeasures(self.measure.measures) }
            if !self.focus.zones.isEmpty { self.sendZones(self.focus.zones) }
        }
        measure.onChange = { [weak self] list in self?.sendMeasures(list) }
        view.session.delegate = self
        view.session.delegateQueue = queue
        sceneView = view
        startSession(reset: true)
        startPreviewTimer()
        link.setEnabled(true)          // siempre busca la PC
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
        if !start { guide = nil }
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

    // MARK: Piezas enfocadas

    /// Toque en la cámara: identifica la pieza bajo el dedo y la enfoca.
    @objc private func tappedCamera(_ g: UITapGestureRecognizer) {
        guard let v = sceneView, let frame = v.session.currentFrame else { return }
        let o = v.window?.windowScene?.interfaceOrientation ?? .portrait
        focus.select(at: g.location(in: v), viewSize: v.bounds.size, frame: frame, orientation: o)
        let msgAt = Date()
        DispatchQueue.main.asyncAfter(deadline: .now() + 6) { [weak self] in
            // el aviso de "pieza enfocada" se quita solo
            if let self, !self.focus.busy, Date().timeIntervalSince(msgAt) >= 5.9 { self.focus.message = nil }
        }
    }

    private func zonesChanged(_ zones: [FocusZone]) {
        sendZones(zones)
        // se guarda con el escaneo del iPhone, para procesarlo después en la PC
        if let dir = recorder.directory,
           let data = try? JSONSerialization.data(withJSONObject: ["zonas": zones.map { $0.json }]) {
            try? data.write(to: dir.appendingPathComponent("zonas.json"))
        }
    }

    func sendZones(_ zones: [FocusZone]) {
        guard link.isConnected else { return }
        if let d = FrameCodec.pack(kind: 6, header: ["zonas": zones.map { $0.json }]) { link.send(d, important: true) }
    }

    func reset() {
        if link.isConnected, let d = FrameCodec.pack(kind: 4, header: [:]) {
            link.send(d, important: true)            // la PC descarta lo recibido de este escaneo
        }
        isScanning = false
        measure.clear()
        focus.clear()
        pointCount = 0
        pointsNode.geometry = nil
        liveMeshNode.geometry = nil
        lastLiveMesh = nil
        lastLiveMeshTime = nil
        liveMeshTriangles = 0
        tooFast = false
        holdingStill = false
        guide = nil
        zoneProgress = [:]
        voice.reset()
        queue.async { self.coach.reset() }
        queue.async { self.anchorPose = nil; self.stillReadings = 0; self.stillState = false }
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
            self.keys.removeAll(keepingCapacity: false)
            self.misses.removeAll(keepingCapacity: false)
            self.nearCounts.removeAll(keepingCapacity: false)
            self.fineKeyToIndex.removeAll(keepingCapacity: false)
            self.fineSums.removeAll(keepingCapacity: false)
            self.fineColorSums.removeAll(keepingCapacity: false)
            self.fineCounts.removeAll(keepingCapacity: false)
            self.closeCounts.removeAll(keepingCapacity: false)
            self.firstDir.removeAll(keepingCapacity: false)
            self.diverse.removeAll(keepingCapacity: false)
            self.carveCursor = 0
            self.cloudFull = false
            self.removedTotal = 0
            DispatchQueue.main.async { self.removedPoints = 0 }
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
            lastCenterDepth = d
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
        updateMotion(frame)
        let zones = focus.current()
        let camT = frame.camera.transform
        let camP = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        nearFocus = zones.contains { $0.distance(to: camP) < 1.0 }
        // si no te mueves, la misma vista ya está leída: no se agregan más puntos ni se manda a la PC
        guard isNewView(frame.camera.transform) else {
            coach.guide(frame: frame, zones: zones, centerDepth: lastCenterDepth)
            return
        }
        let stats = integrate(frame, mask: mask)
        coach.ingest(stats, frame: frame, zones: zones)
        coach.guide(frame: frame, zones: zones, centerDepth: lastCenterDepth)
        var savedKeyframe = false
        if activeRecordRGBD && recorder.isActive, let depth = chosen,
           recorder.consider(frame: frame, depth: depth, mask: mask, dense: nearFocus) {
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

    /// true mientras haya que tomar lecturas: al moverse 2 cm o girar 3° desde la última vista,
    /// o en las primeras lecturas de una vista nueva.
    private func isNewView(_ T: simd_float4x4) -> Bool {
        var moved = true
        if let A = anchorPose {
            let dp = simd_distance(SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z),
                                   SIMD3<Float>(A.columns.3.x, A.columns.3.y, A.columns.3.z))
            var ang = (simd_quatf(A).inverse * simd_quatf(T)).angle
            if ang > .pi { ang = 2 * .pi - ang }
            moved = nearFocus ? (dp > newViewMove * 0.5 || ang > newViewTurn * 0.6)
                              : (dp > newViewMove || ang > newViewTurn)
        }
        if moved {
            anchorPose = T
            stillReadings = 1
        } else {
            stillReadings += 1
        }
        let still = stillReadings > readingsPerView
        if still != stillState {
            stillState = still
            DispatchQueue.main.async { self.holdingStill = still }
        }
        return !still
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
    /// `carpeta`: medidas sobre un modelo terminado (se guardan con ese escaneo en la PC).
    func sendMeasures(_ list: [MeasureController.Measure], carpeta: String? = nil) {
        guard link.isConnected else { return }
        let items: [[String: Any]] = list.map { m in
            ["id": m.id, "nombre": m.name, "a": [m.a.x, m.a.y, m.a.z], "b": [m.b.x, m.b.y, m.b.z]]
        }
        var h: [String: Any] = ["medidas": items]
        if let carpeta { h["carpeta"] = carpeta }
        if let d = FrameCodec.pack(kind: 5, header: h) { link.send(d, important: true) }
    }

    /// Termina el escaneo y le pide a la PC que procese todo en alta calidad.
    func finishOnPC() {
        if isScanning { toggleScanning() }
        link.sendEnd()
        sessionName = ""
        queue.async { self.streamIndex = 0 }
    }

    /// Último modelo en vivo que mandó la PC (para verlo y medirlo en Revisar 3D).
    private(set) var lastLiveMesh: Data?
    private(set) var lastLiveMeshTime: Date?

    private func receiveLiveMesh(_ data: Data) {
        lastLiveMesh = data
        lastLiveMeshTime = Date()
        // En la cámara solo se ven los puntos del iPhone (pintados por calidad): la superficie de la PC
        // tapaba la vista. Se guarda para Revisar 3D ("En vivo (PC)").
        if let u = FrameCodec.unpack(data), let nt = (u.header["nt"] as? NSNumber)?.intValue {
            DispatchQueue.main.async { self.liveMeshTriangles = nt }
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

    @discardableResult
    private func integrate(_ frame: ARFrame, mask: [Bool]?) -> FrameStats {
        let chosen = activeSmooth ? (frame.smoothedSceneDepth ?? frame.sceneDepth)
                                  : (frame.sceneDepth ?? frame.smoothedSceneDepth)
        guard let depthData = chosen,
              let confMap = depthData.confidenceMap else { return FrameStats() }
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
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(image, 1) else { return FrameStats() }

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
        let camPos = SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z)
        let zones = focus.current()
        let hasZones = !zones.isEmpty
        let fineInv = 1.0 / fineVoxel
        let fineLimit = Float((1 << 20) - 1)

        let inv = 1.0 / activeVoxel
        let maxD = activeMaxDepth
        let minConf = activeMinConfidence
        let half: Int64 = 1 << 20
        let limit: Float = Float(half - 2)

        if sums.count >= maxVoxels && !cloudFull {
            // lleno: se quitan los puntos de una sola lectura (ruido o muy lejos) para seguir agregando
            pruneWeak()
            if sums.count > maxVoxels * 85 / 100 {
                cloudFull = true
                DispatchQueue.main.async {
                    self.statusMessage = "La vista del iPhone llegó a su límite de puntos. El escaneo sigue grabando todo para la PC; para ver más aquí, guarda y empieza otro escaneo."
                }
            }
        }
        if sums.count >= maxVoxels {
            // la nube del iPhone es solo vista previa: se deja de crecer, pero el escaneo sigue
            // (fotos y profundidad se siguen mandando a la PC y guardando para procesar)
            return FrameStats()
        }
        var st = FrameStats()
        st.zoneIDs = zones.map { $0.id }
        st.zoneHits = [Int](repeating: 0, count: zones.count)
        st.zoneConfirmed = st.zoneHits
        st.zoneDepth = [Float](repeating: 0, count: zones.count)
        st.depths.reserveCapacity(1600)
        var sampleTick = 0

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
                sampleTick &+= 1
                if sampleTick & 31 == 0 { st.depths.append(d) }
                var zi = -1
                if hasZones {
                    for k in 0..<zones.count where zones[k].contains(p) { zi = k; break }
                    if zi >= 0 { st.zoneHits[zi] += 1; st.zoneDepth[zi] += d }
                }

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

                // dentro de una pieza enfocada (y de cerca): además, un punto cada 2 mm
                if zi >= 0 && d <= 1.2 && fineSums.count < maxFineVoxels {
                    let gx = (p.x * fineInv).rounded(.down), gy = (p.y * fineInv).rounded(.down)
                    let gz = (p.z * fineInv).rounded(.down)
                    if abs(gx) < fineLimit && abs(gy) < fineLimit && abs(gz) < fineLimit {
                        let fk = ((Int64(gx) + half) << 42) | ((Int64(gy) + half) << 21) | (Int64(gz) + half)
                        if let fi = fineKeyToIndex[fk] {
                            let i = Int(fi)
                            if fineCounts[i] < 50_000 {
                                fineSums[i] += p; fineColorSums[i] += c; fineCounts[i] += 1
                            }
                            if fineCounts[i] >= 3 { st.zoneConfirmed[zi] += 1 }
                        } else {
                            fineKeyToIndex[fk] = Int32(fineSums.count)
                            fineSums.append(p); fineColorSums.append(c); fineCounts.append(1)
                        }
                    }
                }

                let near: UInt16 = d <= 1.5 ? 1 : 0
                let close = d <= 1.0
                var dirQ = SIMD3<Int8>(0, 0, 0)
                var dirF = SIMD3<Float>(0, 0, 0)
                if close {
                    dirF = simd_normalize(p - camPos)
                    let q = simd_clamp(dirF * 127, SIMD3<Float>(repeating: -127), SIMD3<Float>(repeating: 127))
                    dirQ = SIMD3<Int8>(Int8(q.x.rounded()), Int8(q.y.rounded()), Int8(q.z.rounded()))
                    if dirQ == SIMD3<Int8>(0, 0, 0) { dirQ.z = 1 }
                }
                if let idx = keyToIndex[key] {
                    let i = Int(idx)
                    if counts[i] < 50_000 {
                        sums[i] += p
                        colorSums[i] += c
                        counts[i] += 1
                        nearCounts[i] += near
                        if close {
                            closeCounts[i] &+= 1
                            let f0 = firstDir[i]
                            if f0 == SIMD3<Int8>(0, 0, 0) {
                                firstDir[i] = dirQ
                            } else if !diverse[i] {
                                let a = simd_normalize(SIMD3<Float>(Float(f0.x), Float(f0.y), Float(f0.z)))
                                if simd_dot(a, dirF) < 0.866 { diverse[i] = true }      // más de 30° de diferencia
                            }
                        }
                    }
                    st.total += 1
                    if nearCounts[i] >= 8 || (closeCounts[i] >= 8 && diverse[i]) { st.good += 1 }
                } else {
                    st.total += 1
                    keyToIndex[key] = Int32(sums.count)
                    sums.append(p)
                    colorSums.append(c)
                    counts.append(1)
                    keys.append(key)
                    misses.append(0)
                    nearCounts.append(near)
                    closeCounts.append(close ? 1 : 0)
                    firstDir.append(dirQ)
                    diverse.append(false)
                }
            }
        }
        if activeCleanNoise {
            carve(dBase: dBase, cBase: cBase, dw: dw, dh: dh, dRow: dRow, cRow: cRow,
                  fx: fx, fy: fy, cx: cx, cy: cy, T: T, mask: mask)
        }
        return st
    }

    // MARK: Limpieza cuando se llena

    /// Quita los vóxeles débiles: una sola lectura, o dos lecturas tomadas de lejos (> 1.5 m).
    /// Casi siempre son ruido u orillas lejanas que se volverán a leer mejor al acercarte.
    private func pruneWeak() {
        let n = sums.count
        var w = 0
        for i in 0..<n {
            let c = counts[i]
            if c < 2 || (c < 3 && nearCounts[i] == 0) { continue }
            if w != i {
                sums[w] = sums[i]; colorSums[w] = colorSums[i]; counts[w] = counts[i]
                keys[w] = keys[i]; misses[w] = misses[i]; nearCounts[w] = nearCounts[i]
                closeCounts[w] = closeCounts[i]; firstDir[w] = firstDir[i]; diverse[w] = diverse[i]
            }
            w += 1
        }
        let drop = n - w
        guard drop > 0 else { return }
        sums.removeLast(drop); colorSums.removeLast(drop); counts.removeLast(drop)
        keys.removeLast(drop); misses.removeLast(drop); nearCounts.removeLast(drop)
        closeCounts.removeLast(drop); firstDir.removeLast(drop); diverse.removeLast(drop)
        keyToIndex.removeAll(keepingCapacity: true)
        for i in 0..<w { keyToIndex[keys[i]] = Int32(i) }
        carveCursor = 0
    }

    // MARK: Limpieza: borrar lo que el LiDAR ve a través

    /// Revisa una parte de los puntos (unos 600 mil por cuadro, en rotación) contra la vista actual.
    /// Si un punto queda claramente DELANTE de lo que mide el LiDAR en ese píxel (y de sus vecinos),
    /// el sensor está viendo a través de él: era ruido o algo que ya no está. Con 2 vistas así,
    /// y si se vio a través al menos la mitad de las veces que se leyó, se borra.
    private func carve(dBase: UnsafeMutableRawPointer, cBase: UnsafeMutableRawPointer, dw: Int, dh: Int,
                       dRow: Int, cRow: Int, fx: Float, fy: Float, cx: Float, cy: Float,
                       T: simd_float4x4, mask: [Bool]?) {
        let n = sums.count
        guard n > 0 else { return }
        // profundidad confiable de este cuadro y su mínimo en 3×3 (protege bordes y piezas delgadas)
        var depth = [Float](repeating: 0, count: dw * dh)
        for v in 0..<dh {
            let dp = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
            let cp = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
            for u in 0..<dw {
                let d = dp[u]
                let masked = mask?[v * dw + u] ?? false
                if cp[u] >= 2 && !masked && d.isFinite && d > 0.15 && d < activeMaxDepth { depth[v * dw + u] = d }
            }
        }
        var minDepth = [Float](repeating: 0, count: dw * dh)   // 0 = no se sabe
        for v in 1..<(dh - 1) {
            for u in 1..<(dw - 1) {
                var m = Float.greatestFiniteMagnitude
                var ok = true
                for dv in -1...1 {
                    for du in -1...1 {
                        let d = depth[(v + dv) * dw + u + du]
                        if d == 0 { ok = false; break }
                        m = min(m, d)
                    }
                    if !ok { break }
                }
                if ok { minDepth[v * dw + u] = m }
            }
        }

        let inv = simd_inverse(T)
        let span = min(n, 600_000)
        if carveCursor >= n { carveCursor = 0 }
        var toRemove: [Int] = []
        var idx = carveCursor
        for _ in 0..<span {
            let c = counts[idx]
            let p = sums[idx] / Float(c)
            let q = inv * SIMD4<Float>(p.x, p.y, p.z, 1)
            let z = -q.z
            if z > 0.15 {
                let u = Int((cx + fx * q.x / z).rounded())
                let v = Int((cy - fy * q.y / z).rounded())
                if u > 0 && v > 0 && u < dw - 1 && v < dh - 1 {
                    let D = minDepth[v * dw + u]
                    if D > 0 && z < D - max(0.02, 0.02 * D) {
                        let m = misses[idx] &+ 1
                        misses[idx] = m
                        if m >= 2 && UInt32(m) * 2 >= UInt32(c) { toRemove.append(idx) }
                    }
                }
            }
            idx += 1
            if idx >= n { idx = 0 }
        }
        carveCursor = idx
        guard !toRemove.isEmpty else { return }
        // borrar intercambiando con el último (de mayor a menor índice para no mover los pendientes)
        for i in toRemove.sorted(by: >) {
            let last = sums.count - 1
            keyToIndex.removeValue(forKey: keys[i])
            if i != last {
                sums[i] = sums[last]; colorSums[i] = colorSums[last]; counts[i] = counts[last]
                keys[i] = keys[last]; misses[i] = misses[last]; nearCounts[i] = nearCounts[last]
                closeCounts[i] = closeCounts[last]; firstDir[i] = firstDir[last]; diverse[i] = diverse[last]
                keyToIndex[keys[i]] = Int32(i)
            }
            sums.removeLast(); colorSums.removeLast(); counts.removeLast()
            keys.removeLast(); misses.removeLast(); nearCounts.removeLast()
            closeCounts.removeLast(); firstDir.removeLast(); diverse.removeLast()
        }
        removedTotal += toRemove.count
        let total = removedTotal
        DispatchQueue.main.async { self.removedPoints = total }
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
            let geometry = show ? self.buildGeometry(mode: mode, maxPoints: 900_000,
                                                     pointSize: 3, maxRadius: 3) : nil
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

    /// Calidad de cada zona (no solo si ya se vio):
    /// verde oscuro = excelente: leída muy de cerca (≤ 1 m) varias veces y desde ángulos distintos
    ///   (≥ 30°) → ya no vale la pena seguir pasando por ahí;
    /// verde = leída de cerca (≤ 1.5 m) desde al menos 2 vistas → sale nítida y exacta;
    /// amarillo = vista, pero de lejos o desde una sola vista → acércate o rodéala;
    /// rojo = apenas vista.
    static let qualityExcellent = SIMD3<Float>(0.02, 0.50, 0.26)
    static let qualityGood = SIMD3<Float>(0.48, 0.90, 0.45)
    static let qualityFair = SIMD3<Float>(0.98, 0.70, 0.15)
    static let qualityPoor = SIMD3<Float>(0.95, 0.24, 0.20)

    private func qualityColor(count: UInt16, near: UInt16, close: UInt16, diverse: Bool) -> SIMD3<Float> {
        if close >= 8 && diverse { return Self.qualityExcellent }
        if near >= 8 { return Self.qualityGood }
        if count >= 4 { return Self.qualityFair }
        return Self.qualityPoor
    }

    /// Calidad en un punto del espacio (busca su vóxel y los 6 vecinos). Solo en `queue`.
    private func quality(at p: SIMD3<Float>) -> SIMD3<Float> {
        let inv = 1.0 / activeVoxel
        let half: Int64 = 1 << 20
        let bx = Int64((p.x * inv).rounded(.down)), by = Int64((p.y * inv).rounded(.down)), bz = Int64((p.z * inv).rounded(.down))
        var bestCount: UInt16 = 0, bestNear: UInt16 = 0, bestClose: UInt16 = 0, anyDiverse = false
        for (dx, dy, dz) in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] {
            let key = ((bx + Int64(dx) + half) << 42) | ((by + Int64(dy) + half) << 21) | (bz + Int64(dz) + half)
            if let i = keyToIndex[key] {
                bestCount = max(bestCount, counts[Int(i)])
                bestNear = max(bestNear, nearCounts[Int(i)])
                bestClose = max(bestClose, closeCounts[Int(i)])
                anyDiverse = anyDiverse || diverse[Int(i)]
            }
        }
        return qualityColor(count: bestCount, near: bestNear, close: bestClose, diverse: anyDiverse)
    }

    /// ¿Ya hay puntos finos alrededor de este punto? (si no, se sigue dibujando el punto normal)
    private func hasFine(near p: SIMD3<Float>) -> Bool {
        let inv = 1.0 / fineVoxel
        let half: Int64 = 1 << 20
        let bx = Int64((p.x * inv).rounded(.down)), by = Int64((p.y * inv).rounded(.down)), bz = Int64((p.z * inv).rounded(.down))
        for dx in -1...1 { for dy in -1...1 { for dz in -1...1 {
            let k = ((bx + Int64(dx) + half) << 42) | ((by + Int64(dy) + half) << 21) | (bz + Int64(dz) + half)
            if fineKeyToIndex[k] != nil { return true }
        } } }
        return false
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
        let zones = focus.current()
        var i = 0
        while i < n {
            let cnt = counts[i]
            if cnt >= minObs {
                let k = Float(cnt)
                let pt = sums[i] / k
                if !zones.isEmpty && zones.contains(where: { $0.contains(pt) }) && hasFine(near: pt) {
                    i += step
                    continue          // ahí se dibujan los puntos finos
                }
                verts.append(pt)
                if mode == .coverage {
                    cols.append(qualityColor(count: cnt, near: nearCounts[i], close: closeCounts[i], diverse: diverse[i]))
                } else {
                    cols.append(colorSums[i] / (k * 255))
                }
            }
            i += step
        }
        // puntos finos (2 mm) de las piezas enfocadas: todos, sin saltar
        let fineMin: UInt16 = mode == .coverage ? 1 : 2
        for j in 0..<fineSums.count where fineCounts[j] >= fineMin {
            let k = Float(fineCounts[j])
            verts.append(fineSums[j] / k)
            if mode == .coverage {
                let fc = fineCounts[j]
                cols.append(fc >= 6 ? Self.qualityExcellent : fc >= 3 ? Self.qualityGood : Self.qualityFair)
            } else {
                cols.append(fineColorSums[j] / (k * 255))
            }
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
