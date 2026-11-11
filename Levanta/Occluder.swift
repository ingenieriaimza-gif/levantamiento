import ARKit
import SceneKit
import simd

/// Pared invisible para la vista en vivo: con la profundidad del LiDAR de este cuadro arma una malla que
/// no se ve (solo escribe profundidad) y esconde los puntos que quedan DETRÁS de la superficie que estás viendo.
/// Así, al pasar al frente de una pieza no se ven los puntos de la parte de atrás como si fuera transparente.
/// Solo afecta lo que se dibuja; no cambia lo que se guarda ni lo que se manda a la PC.
final class Occluder {
    let node = SCNNode()
    private let step = 2                    // un vértice cada 2 píxeles del mapa de profundidad (128×96)
    private var lastTime: TimeInterval = 0
    private let interval: TimeInterval = 0.08

    init() {
        node.renderingOrder = -10           // antes que los puntos
        node.isHidden = true
        node.castsShadow = false
    }

    private static func makeMaterial() -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.colorBufferWriteMask = []         // invisible: solo tapa a lo que queda atrás
        m.writesToDepthBuffer = true
        m.readsFromDepthBuffer = true
        m.isDoubleSided = true
        return m
    }

    func clear() {
        DispatchQueue.main.async { self.node.geometry = nil; self.node.isHidden = true }
    }

    /// Se llama con cada cuadro de AR (en la cola de escaneo). Se limita a ~12 por segundo.
    func update(frame: ARFrame) {
        guard frame.timestamp - lastTime >= interval else { return }
        lastTime = frame.timestamp
        guard let depth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        let dMap = depth.depthMap
        CVPixelBufferLockBaseAddress(dMap, .readOnly)
        let cMap = depth.confidenceMap
        if let c = cMap { CVPixelBufferLockBaseAddress(c, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(dMap, .readOnly)
            if let c = cMap { CVPixelBufferUnlockBaseAddress(c, .readOnly) }
        }
        guard let dBase = CVPixelBufferGetBaseAddress(dMap) else { return }
        let cBase = cMap.flatMap { CVPixelBufferGetBaseAddress($0) }
        let w = CVPixelBufferGetWidth(dMap), h = CVPixelBufferGetHeight(dMap)
        let dRow = CVPixelBufferGetBytesPerRow(dMap)
        let cRow = cMap.map { CVPixelBufferGetBytesPerRow($0) } ?? 0

        let K = frame.camera.intrinsics
        let img = frame.camera.imageResolution
        let sx = Float(w) / Float(img.width), sy = Float(h) / Float(img.height)
        let fx = K[0][0] * sx, fy = K[1][1] * sy
        let cx = (K[2][0] + 0.5) * sx - 0.5, cy = (K[2][1] + 0.5) * sy - 0.5
        let T = frame.camera.transform

        let gw = w / step, gh = h / step
        var verts = [SIMD3<Float>](repeating: .zero, count: gw * gh)
        var depths = [Float](repeating: 0, count: gw * gh)
        for j in 0..<gh {
            let v = j * step
            let dp = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
            let cp = cBase?.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
            for i in 0..<gw {
                let u = i * step
                let d = dp[u]
                guard d.isFinite, d > 0.2, d < 5 else { continue }
                if let cp, cp[u] < 1 { continue }
                // un poco más atrás que la superficie para no tapar los puntos que están sobre ella
                let dd = d + 0.03 + 0.02 * d
                let pw = T * SIMD4<Float>((Float(u) - cx) * dd / fx, -(Float(v) - cy) * dd / fy, -dd, 1)
                verts[j * gw + i] = SIMD3<Float>(pw.x, pw.y, pw.z)
                depths[j * gw + i] = d
            }
        }
        var idx: [UInt32] = []
        idx.reserveCapacity(gw * gh * 4)
        for j in 0..<(gh - 1) {
            for i in 0..<(gw - 1) {
                let a = j * gw + i, b = a + 1, c = a + gw, d = c + 1
                let da = depths[a], db = depths[b], dc = depths[c], dd = depths[d]
                guard da > 0, db > 0, dc > 0, dd > 0 else { continue }
                // no unir bordes con saltos grandes de profundidad (si no, se estira una pared falsa)
                let lo = min(min(da, db), min(dc, dd)), hi = max(max(da, db), max(dc, dd))
                guard hi - lo < 0.05 + 0.04 * lo else { continue }
                idx.append(contentsOf: [UInt32(a), UInt32(c), UInt32(b), UInt32(b), UInt32(c), UInt32(d)])
            }
        }
        guard !idx.isEmpty else { return }

        let stride = MemoryLayout<SIMD3<Float>>.stride
        let vData = verts.withUnsafeBufferPointer { Data(buffer: $0) }
        let src = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: verts.count,
                                    usesFloatComponents: true, componentsPerVector: 3,
                                    bytesPerComponent: 4, dataOffset: 0, dataStride: stride)
        let element = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [src], elements: [element])
        geometry.materials = [Occluder.makeMaterial()]
        DispatchQueue.main.async {
            self.node.geometry = geometry
            self.node.isHidden = false
        }
    }
}
