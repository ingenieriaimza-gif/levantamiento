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
    @Published var colorMode: ColorMode = .coverage
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
    private var tracking = false
    private var previewTimer: Timer?
    private var previewBusy = false
    private var lastDistanceTime: TimeInterval = 0

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
        config.worldAlignment = .gravity
        let options: ARSession.RunOptions = reset ? [.resetTracking, .removeExistingAnchors] : []
        view.session.run(config, options: options)
    }

    // MARK: Controles

    func toggleScanning() {
        let start = !isScanning
        isScanning = start
        let voxel = Float(voxelSizeCm / 100.0)
        let depth = Float(maxDepth)
        let conf: UInt8 = highConfidenceOnly ? 2 : 1
        let minObs = UInt16(max(1, minObservations))
        let smooth = smoothDepth
        queue.async {
            self.activeSmooth = smooth
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
        isScanning = false
        pointCount = 0
        pointsNode.geometry = nil
        let voxel = Float(voxelSizeCm / 100.0)
        queue.async {
            self.scanning = false
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
        integrate(frame)
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

    private func integrate(_ frame: ARFrame) {
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
        let cx = K[2][0] * sx
        let cy = K[2][1] * sy
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
    func makeReviewData(mode: ColorMode, completion: @escaping (SCNGeometry?, CloudStats) -> Void) {
        queue.async {
            let geometry = self.buildGeometry(mode: mode, maxPoints: 1_500_000,
                                              pointSize: 4, maxRadius: 3)
            let stats = self.computeStats()
            DispatchQueue.main.async { completion(geometry, stats) }
        }
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
        return items.filter { $0.pathExtension.lowercased() == "ply" }.sorted { a, b in
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
