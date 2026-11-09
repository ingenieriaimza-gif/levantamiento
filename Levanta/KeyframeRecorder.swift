import ARKit
import CoreImage
import ImageIO
import UIKit

/// Graba cuadros clave RGB-D mientras se escanea: foto (JPEG), profundidad del LiDAR (mm),
/// confianza y la posición/orientación exacta del iPhone. Con esto se puede hacer, fuera del
/// teléfono, una fusión volumétrica y texturizado de mucha mejor calidad que la nube en vivo.
///
/// Formato del paquete (una carpeta `<nombre>_rgbd`):
///   meta.json      — descripción general
///   frames.jsonl   — una línea JSON por cuadro: i, t, fx, fy, cx, cy (a la resolución de la foto),
///                    T (matriz 4×4 cámara→mundo de ARKit, por columnas; ejes x derecha, y arriba, z atrás)
///   frames/000000.jpg    foto en orientación del sensor (horizontal)
///   frames/000000.depth  UInt16 little-endian, milímetros, fila por fila, 0 = sin dato
///   frames/000000.conf   UInt8 por píxel (0 baja, 1 media, 2 alta)
final class KeyframeRecorder {
    private let writeQueue = DispatchQueue(label: "levanta.keyframes.write", qos: .utility)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var jsonl: FileHandle?
    private var lastPose: simd_float4x4?
    private var lastTime: TimeInterval = 0
    private var depthW = 0, depthH = 0, imgW = 0, imgH = 0

    private(set) var directory: URL?
    private(set) var count = 0
    var isActive: Bool { directory != nil }

    /// Escala de la foto guardada (1920×1440 × 0.75 = 1440×1080).
    let imageScale: CGFloat = 0.75
    let jpegQuality: CGFloat = 0.8
    let minTranslation: Float = 0.07      // m
    let minRotation: Float = 6 * .pi / 180
    let minInterval: TimeInterval = 0.25
    let maxFrames = 1200

    // MARK: Sesión

    func begin() throws {
        let f = DateFormatter()
        f.locale = Locale(identifier: "es_MX")
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let dir = try ScanStore.folder().appendingPathComponent("RGBD_\(f.string(from: Date()))", isDirectory: true)
        try FileManager.default.createDirectory(at: dir.appendingPathComponent("frames"),
                                                withIntermediateDirectories: true)
        let jsonURL = dir.appendingPathComponent("frames.jsonl")
        FileManager.default.createFile(atPath: jsonURL.path, contents: nil)
        jsonl = try FileHandle(forWritingTo: jsonURL)
        directory = dir
        count = 0
        lastPose = nil
        lastTime = 0
    }

    /// Termina la sesión y devuelve la carpeta (o nil si no había nada).
    func finish() -> URL? {
        guard let dir = directory else { return nil }
        writeQueue.sync {
            try? self.jsonl?.close()
            self.jsonl = nil
        }
        let meta: [String: Any] = [
            "format": "levanta-rgbd",
            "version": 1,
            "frames": count,
            "imageWidth": imgW, "imageHeight": imgH,
            "depthWidth": depthW, "depthHeight": depthH,
            "depthUnits": "mm",
            "poseConvention": "ARKit camera-to-world, column-major 4x4; camera x right, y up, z back; world y up (gravity), meters",
            "intrinsicsAt": "image resolution",
            "device": UIDevice.current.model,
            "system": UIDevice.current.systemVersion
        ]
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: dir.appendingPathComponent("meta.json"))
        }
        directory = nil
        if count == 0 {
            try? FileManager.default.removeItem(at: dir)
            return nil
        }
        return dir
    }

    /// Descarta la sesión actual sin guardar.
    func discard() {
        guard let dir = directory else { return }
        writeQueue.sync {
            try? self.jsonl?.close()
            self.jsonl = nil
        }
        directory = nil
        count = 0
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: Cuadros

    /// Decide si este cuadro es clave (se movió o giró lo suficiente) y lo guarda.
    /// Se llama en la cola de escaneo. Devuelve true si se guardó.
    @discardableResult
    func consider(frame: ARFrame, depth: ARDepthData) -> Bool {
        guard directory != nil, count < maxFrames else { return false }
        guard frame.timestamp - lastTime >= minInterval else { return false }
        let T = frame.camera.transform
        if let L = lastPose {
            let dp = simd_distance(SIMD3<Float>(T.columns.3.x, T.columns.3.y, T.columns.3.z),
                                   SIMD3<Float>(L.columns.3.x, L.columns.3.y, L.columns.3.z))
            let q = simd_quatf(L).inverse * simd_quatf(T)
            var ang = q.angle
            if ang > .pi { ang = 2 * .pi - ang }
            if dp < minTranslation && ang < minRotation { return false }
        }

        // Foto
        let image = frame.capturedImage
        let fullW = CVPixelBufferGetWidth(image)
        let fullH = CVPixelBufferGetHeight(image)
        let ci = CIImage(cvPixelBuffer: image)
            .transformed(by: CGAffineTransform(scaleX: imageScale, y: imageScale))
        let qualityKey = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let jpeg = ciContext.jpegRepresentation(of: ci, colorSpace: space,
                                                      options: [qualityKey: jpegQuality]) else { return false }

        // Profundidad (mm) y confianza
        let dMap = depth.depthMap
        guard let cMap = depth.confidenceMap else { return false }
        CVPixelBufferLockBaseAddress(dMap, .readOnly)
        CVPixelBufferLockBaseAddress(cMap, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(dMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(cMap, .readOnly)
        }
        guard let dBase = CVPixelBufferGetBaseAddress(dMap),
              let cBase = CVPixelBufferGetBaseAddress(cMap) else { return false }
        let w = CVPixelBufferGetWidth(dMap)
        let h = CVPixelBufferGetHeight(dMap)
        let dRow = CVPixelBufferGetBytesPerRow(dMap)
        let cRow = CVPixelBufferGetBytesPerRow(cMap)
        var depthBytes = [UInt8](repeating: 0, count: w * h * 2)
        var confBytes = [UInt8](repeating: 0, count: w * h)
        for v in 0..<h {
            let dp = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
            let cp = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
            for u in 0..<w {
                let d = dp[u]
                var mm: UInt16 = 0
                if d.isFinite && d > 0 && d < 65 { mm = UInt16((d * 1000).rounded()) }
                let o = (v * w + u) * 2
                depthBytes[o] = UInt8(mm & 0xff)
                depthBytes[o + 1] = UInt8(mm >> 8)
                confBytes[v * w + u] = cp[u]
            }
        }

        // Intrínsecos a la resolución de la foto guardada
        let K = frame.camera.intrinsics
        let s = Float(imageScale)
        let savedW = Int((CGFloat(fullW) * imageScale).rounded())
        let savedH = Int((CGFloat(fullH) * imageScale).rounded())
        depthW = w; depthH = h; imgW = savedW; imgH = savedH

        var cols: [Float] = []
        cols.reserveCapacity(16)
        for c in 0..<4 {
            let col = T[c]
            cols.append(contentsOf: [col.x, col.y, col.z, col.w])
        }
        let line: [String: Any] = [
            "i": count,
            "t": frame.timestamp,
            "fx": K[0][0] * s, "fy": K[1][1] * s,
            "cx": (K[2][0] + 0.5) * s - 0.5, "cy": (K[2][1] + 0.5) * s - 0.5,
            "w": savedW, "h": savedH,
            "T": cols
        ]
        guard let lineData = try? JSONSerialization.data(withJSONObject: line) else { return false }

        let index = count
        let name = String(format: "%06d", index)
        guard let framesDir = directory?.appendingPathComponent("frames") else { return false }
        let depthData = Data(depthBytes)
        let confData = Data(confBytes)
        writeQueue.async {
            try? jpeg.write(to: framesDir.appendingPathComponent(name + ".jpg"))
            try? depthData.write(to: framesDir.appendingPathComponent(name + ".depth"))
            try? confData.write(to: framesDir.appendingPathComponent(name + ".conf"))
            self.jsonl?.write(lineData)
            self.jsonl?.write(Data("\n".utf8))
        }

        count += 1
        lastPose = T
        lastTime = frame.timestamp
        return true
    }

    // MARK: Empaquetado para compartir

    /// Divide la carpeta en partes .zip de hasta `maxPartBytes` para poder enviarlas por chat.
    /// Cada parte lleva meta.json y su propio frames.jsonl. Al terminar borra la carpeta original.
    static func packageParts(_ dir: URL, maxPartBytes: Int = 24 * 1024 * 1024) throws -> [URL] {
        let fm = FileManager.default
        let jsonText = (try? String(contentsOf: dir.appendingPathComponent("frames.jsonl"), encoding: .utf8)) ?? ""
        let lines = jsonText.split(separator: "\n").map(String.init)
        let framesDir = dir.appendingPathComponent("frames")

        func size(_ url: URL) -> Int {
            (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }

        // Agrupar cuadros en partes por tamaño
        var parts: [[(line: String, name: String)]] = [[]]
        var current = 0
        for line in lines {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let i = obj["i"] as? Int else { continue }
            let name = String(format: "%06d", i)
            let bytes = size(framesDir.appendingPathComponent(name + ".jpg"))
                + size(framesDir.appendingPathComponent(name + ".depth"))
                + size(framesDir.appendingPathComponent(name + ".conf"))
            if current + bytes > maxPartBytes && !(parts.last?.isEmpty ?? true) {
                parts.append([])
                current = 0
            }
            parts[parts.count - 1].append((line: line, name: name))
            current += bytes
        }
        parts = parts.filter { !$0.isEmpty }
        guard !parts.isEmpty else { return [] }

        let base = dir.lastPathComponent
        let outDir = dir.deletingLastPathComponent()
        let tmpRoot = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: tmpRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tmpRoot) }

        var results: [URL] = []
        for (k, part) in parts.enumerated() {
            let partName = parts.count == 1 ? base : "\(base)_parte\(k + 1)de\(parts.count)"
            let partDir = tmpRoot.appendingPathComponent(partName, isDirectory: true)
            let partFrames = partDir.appendingPathComponent("frames", isDirectory: true)
            try fm.createDirectory(at: partFrames, withIntermediateDirectories: true)
            try? fm.copyItem(at: dir.appendingPathComponent("meta.json"),
                             to: partDir.appendingPathComponent("meta.json"))
            let partLines = part.map { $0.line }.joined(separator: "\n") + "\n"
            try partLines.write(to: partDir.appendingPathComponent("frames.jsonl"), atomically: true, encoding: .utf8)
            for item in part {
                for ext in ["jpg", "depth", "conf"] {
                    let src = framesDir.appendingPathComponent(item.name + "." + ext)
                    let dst = partFrames.appendingPathComponent(item.name + "." + ext)
                    if (try? fm.linkItem(at: src, to: dst)) == nil {
                        try? fm.copyItem(at: src, to: dst)
                    }
                }
            }

            // Comprimir la carpeta en .zip con el sistema
            var coordError: NSError?
            var copyError: Error?
            let dest = outDir.appendingPathComponent(partName + ".zip")
            NSFileCoordinator().coordinate(readingItemAt: partDir, options: [.forUploading], error: &coordError) { zipURL in
                do {
                    if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                    try fm.copyItem(at: zipURL, to: dest)
                } catch {
                    copyError = error
                }
            }
            if let e = coordError { throw e }
            if let e = copyError { throw e }
            results.append(dest)
        }
        try? fm.removeItem(at: dir)
        return results
    }
}
