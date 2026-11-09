import ARKit
import AVFoundation
import CoreImage
import ImageIO
import SceneKit

/// Utilidades para codificar cuadros (foto, profundidad, confianza, posición) y medir nitidez.
enum FrameCodec {
    static let ciContext = CIContext(options: [.cacheIntermediates: false])

    /// JPEG de la imagen de la cámara, en la orientación del sensor, escalado por `scale`.
    static func jpeg(_ frame: ARFrame, scale: CGFloat, quality: CGFloat) -> Data? {
        var ci = CIImage(cvPixelBuffer: frame.capturedImage)
        if scale != 1 {
            ci = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return ciContext.jpegRepresentation(of: ci, colorSpace: space, options: [key: quality])
    }

    /// Profundidad en milímetros (UInt16 little-endian) y confianza (UInt8), fila por fila.
    static func depthAndConfidence(_ depth: ARDepthData) -> (depth: Data, conf: Data, w: Int, h: Int)? {
        let dMap = depth.depthMap
        guard let cMap = depth.confidenceMap else { return nil }
        CVPixelBufferLockBaseAddress(dMap, .readOnly)
        CVPixelBufferLockBaseAddress(cMap, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(dMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(cMap, .readOnly)
        }
        guard let dBase = CVPixelBufferGetBaseAddress(dMap),
              let cBase = CVPixelBufferGetBaseAddress(cMap) else { return nil }
        let w = CVPixelBufferGetWidth(dMap)
        let h = CVPixelBufferGetHeight(dMap)
        let dRow = CVPixelBufferGetBytesPerRow(dMap)
        let cRow = CVPixelBufferGetBytesPerRow(cMap)
        var dBytes = [UInt8](repeating: 0, count: w * h * 2)
        var cBytes = [UInt8](repeating: 0, count: w * h)
        for v in 0..<h {
            let dp = dBase.advanced(by: v * dRow).assumingMemoryBound(to: Float32.self)
            let cp = cBase.advanced(by: v * cRow).assumingMemoryBound(to: UInt8.self)
            for u in 0..<w {
                let d = dp[u]
                var mm: UInt16 = 0
                if d.isFinite && d > 0 && d < 65 { mm = UInt16((d * 1000).rounded()) }
                let o = (v * w + u) * 2
                dBytes[o] = UInt8(mm & 0xff)
                dBytes[o + 1] = UInt8(mm >> 8)
                cBytes[v * w + u] = cp[u]
            }
        }
        return (Data(dBytes), Data(cBytes), w, h)
    }

    /// Encabezado con intrínsecos (a la resolución de la foto guardada) y posición de la cámara.
    static func header(_ frame: ARFrame, index: Int, scale: Float) -> [String: Any] {
        let K = frame.camera.intrinsics
        let T = frame.camera.transform
        let fullW = CVPixelBufferGetWidth(frame.capturedImage)
        let fullH = CVPixelBufferGetHeight(frame.capturedImage)
        var cols: [Float] = []
        cols.reserveCapacity(16)
        for c in 0..<4 {
            let col = T[c]
            cols.append(contentsOf: [col.x, col.y, col.z, col.w])
        }
        return [
            "i": index,
            "t": frame.timestamp,
            "fx": K[0][0] * scale, "fy": K[1][1] * scale,
            "cx": (K[2][0] + 0.5) * scale - 0.5, "cy": (K[2][1] + 0.5) * scale - 0.5,
            "w": Int((Float(fullW) * scale).rounded()), "h": Int((Float(fullH) * scale).rounded()),
            "T": cols
        ]
    }

    /// Nitidez de la imagen (energía del laplaciano en el centro). Más alto = más nítida.
    static func sharpness(_ image: CVPixelBuffer) -> Float {
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard CVPixelBufferGetPlaneCount(image) >= 1,
              let base = CVPixelBufferGetBaseAddressOfPlane(image, 0) else { return 0 }
        let w = CVPixelBufferGetWidthOfPlane(image, 0)
        let h = CVPixelBufferGetHeightOfPlane(image, 0)
        let row = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let p = base.assumingMemoryBound(to: UInt8.self)
        var sum: Float = 0
        var n = 0
        var y = h / 6
        while y < h - h / 6 {
            var x = w / 6
            while x < w - w / 6 {
                let c = Int(p[y * row + x])
                let lap = 4 * c - Int(p[y * row + x - 1]) - Int(p[y * row + x + 1])
                    - Int(p[(y - 1) * row + x]) - Int(p[(y + 1) * row + x])
                sum += Float(lap * lap)
                n += 1
                x += 4
            }
            y += 4
        }
        return n > 0 ? sum / Float(n) : 0
    }

    /// Mensaje binario del protocolo: [tipo][long. encabezado u32 LE][JSON][datos…]
    static func pack(kind: UInt8, header: [String: Any], payload: [Data] = []) -> Data? {
        guard let h = try? JSONSerialization.data(withJSONObject: header) else { return nil }
        var out = Data([kind])
        var len = UInt32(h.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(h)
        for p in payload { out.append(p) }
        return out
    }

    /// Desarma un mensaje: (tipo, encabezado, posición donde empiezan los datos).
    static func unpack(_ data: Data) -> (kind: UInt8, header: [String: Any], offset: Int)? {
        guard data.count >= 5 else { return nil }
        let b = data.startIndex
        let kind = data[b]
        let hl = Int(data[b + 1]) | Int(data[b + 2]) << 8 | Int(data[b + 3]) << 16 | Int(data[b + 4]) << 24
        guard data.count >= 5 + hl,
              let obj = try? JSONSerialization.jsonObject(with: data.subdata(in: (b + 5)..<(b + 5 + hl))),
              let header = obj as? [String: Any] else { return nil }
        return (kind, header, 5 + hl)
    }

    /// Geometría SceneKit de la malla en vivo que manda la PC.
    static func liveMeshGeometry(_ data: Data, tint: Bool) -> SCNGeometry? {
        guard let u = unpack(data), u.kind == 10,
              let nv = (u.header["nv"] as? NSNumber)?.intValue, let nt = (u.header["nt"] as? NSNumber)?.intValue,
              nv > 0, nt > 0 else { return nil }
        let b = data.startIndex + u.offset
        let vEnd = b + nv * 12
        let cEnd = vEnd + nv * 3
        let iEnd = cEnd + nt * 12
        guard data.count >= iEnd - data.startIndex else { return nil }
        let vData = data.subdata(in: b..<vEnd)
        let idxData = data.subdata(in: cEnd..<iEnd)
        let vSource = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: nv,
                                        usesFloatComponents: true, componentsPerVector: 3,
                                        bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
        var sources = [vSource]
        if !tint {
            let raw = data.subdata(in: vEnd..<cEnd)
            var floats = [Float](repeating: 0, count: nv * 3)
            raw.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                for k in 0..<(nv * 3) { floats[k] = Float(p[k]) / 255 }
            }
            let cData = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            sources.append(SCNGeometrySource(data: cData, semantic: .color, vectorCount: nv,
                                             usesFloatComponents: true, componentsPerVector: 3,
                                             bytesPerComponent: 4, dataOffset: 0, dataStride: 12))
        }
        let element = SCNGeometryElement(data: idxData, primitiveType: .triangles,
                                         primitiveCount: nt, bytesPerIndex: 4)
        let g = SCNGeometry(sources: sources, elements: [element])
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.isDoubleSided = true
        if tint {
            m.diffuse.contents = UIColor(red: 0.25, green: 0.85, blue: 0.45, alpha: 1)
            m.transparency = 0.45
        } else {
            m.diffuse.contents = UIColor.white
            m.transparency = 0.92
        }
        m.writesToDepthBuffer = true
        g.materials = [m]
        return g
    }
}

/// Ajuste de la cámara para fotos nítidas y color constante durante el escaneo.
enum CameraTuning {
    /// sharp = true: exposición corta fija (≈1/250 s, compensando con ISO) y balance de blancos fijo.
    static func apply(sharp: Bool) {
        guard let device = ARWorldTrackingConfiguration.configurableCaptureDeviceForPrimaryCamera else { return }
        do { try device.lockForConfiguration() } catch { return }
        defer { device.unlockForConfiguration() }
        if sharp {
            let format = device.activeFormat
            let target = CMTime(value: 1, timescale: 250)
            let duration = CMTimeMaximum(format.minExposureDuration, CMTimeMinimum(target, format.maxExposureDuration))
            let curSec = CMTimeGetSeconds(device.exposureDuration)
            let newSec = CMTimeGetSeconds(duration)
            var iso = device.iso
            if curSec > 0 && newSec > 0 { iso = device.iso * Float(curSec / newSec) }
            iso = min(max(iso, format.minISO), format.maxISO)
            if device.isExposureModeSupported(.custom) {
                device.setExposureModeCustom(duration: duration, iso: iso, completionHandler: nil)
            }
            if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
            }
        } else {
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
            }
        }
    }
}
