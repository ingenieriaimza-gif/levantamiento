import ARKit
import SceneKit

/// Vista de malla en vivo: la malla del iPhone (ARKit) dibujada como alambre blanco fino sobre la cámara,
/// y en azul lo que todavía no se escanea. Es solo para ver; el modelo exacto lo hace la PC.
final class LiveMeshView: NSObject, ARSCNViewDelegate {
    /// Se puede apagar para ver la nube de puntos con colores de calidad.
    var enabled = true {
        didSet { DispatchQueue.main.async { self.applyVisibility() } }
    }
    private var meshNodes: [UUID: SCNNode] = [:]
    private let lock = NSLock()
    private let blueOverlay = SCNNode()
    private weak var view: ARSCNView?

    static var supported: Bool { ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) }

    // Materiales compartidos
    private let occluder: SCNMaterial = {
        let m = SCNMaterial()
        m.colorBufferWriteMask = []          // invisible, pero tapa el azul donde ya hay malla
        m.writesToDepthBuffer = true
        m.isDoubleSided = true
        return m
    }()
    private let wire: SCNMaterial = {
        let m = SCNMaterial()
        m.fillMode = .lines
        m.lightingModel = .constant
        m.diffuse.contents = UIColor(white: 1, alpha: 0.45)
        m.isDoubleSided = true
        m.writesToDepthBuffer = false
        return m
    }()

    func attach(to view: ARSCNView) {
        self.view = view
        view.delegate = self
        // velo azul pegado a la cámara a 4 m: se ve donde no hay malla más cerca
        let plane = SCNPlane(width: 40, height: 40)
        let m = SCNMaterial()
        m.lightingModel = .constant
        m.diffuse.contents = UIColor(red: 0.18, green: 0.30, blue: 0.95, alpha: 0.42)
        m.readsFromDepthBuffer = true
        m.writesToDepthBuffer = false
        m.isDoubleSided = true
        plane.firstMaterial = m
        blueOverlay.geometry = plane
        blueOverlay.position = SCNVector3(0, 0, -4)
        blueOverlay.renderingOrder = 100
        view.pointOfView?.addChildNode(blueOverlay)
        applyVisibility()
    }

    /// Borra lo dibujado (al empezar de nuevo).
    func clear() {
        lock.lock(); let nodes = Array(meshNodes.values); meshNodes.removeAll(); lock.unlock()
        DispatchQueue.main.async { nodes.forEach { $0.removeFromParentNode() } }
    }

    private func applyVisibility() {
        if blueOverlay.parent == nil, let pov = view?.pointOfView { pov.addChildNode(blueOverlay) }
        blueOverlay.isHidden = !enabled
        lock.lock(); let nodes = Array(meshNodes.values); lock.unlock()
        nodes.forEach { $0.isHidden = !enabled }
    }

    // MARK: ARSCNViewDelegate

    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let a = anchor as? ARMeshAnchor else { return nil }
        let node = SCNNode()
        node.isHidden = !enabled
        update(node, with: a)
        lock.lock(); meshNodes[a.identifier] = node; lock.unlock()
        return node
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let a = anchor as? ARMeshAnchor, enabled else { return }
        update(node, with: a)
    }

    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        lock.lock(); meshNodes.removeValue(forKey: anchor.identifier); lock.unlock()
    }

    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        if blueOverlay.parent == nil, let pov = view?.pointOfView {
            DispatchQueue.main.async { pov.addChildNode(self.blueOverlay) }
        }
    }

    private func update(_ node: SCNNode, with anchor: ARMeshAnchor) {
        let g = anchor.geometry
        let v = g.vertices
        let f = g.faces
        let vData = Data(bytes: v.buffer.contents().advanced(by: v.offset), count: v.count * v.stride)
        let src = SCNGeometrySource(data: vData, semantic: .vertex, vectorCount: v.count, usesFloatComponents: true,
                                    componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: v.stride)
        let fData = Data(bytes: f.buffer.contents(), count: f.count * f.indexCountPerPrimitive * f.bytesPerIndex)
        let el = SCNGeometryElement(data: fData, primitiveType: .triangles, primitiveCount: f.count,
                                    bytesPerIndex: f.bytesPerIndex)
        let solid = SCNGeometry(sources: [src], elements: [el])
        solid.materials = [occluder]
        let lines = SCNGeometry(sources: [src], elements: [el])
        lines.materials = [wire]
        DispatchQueue.main.async {
            let occ = node.childNode(withName: "o", recursively: false) ?? {
                let n = SCNNode(); n.name = "o"; n.renderingOrder = -10; node.addChildNode(n); return n }()
            let w = node.childNode(withName: "w", recursively: false) ?? {
                let n = SCNNode(); n.name = "w"; n.renderingOrder = 50; node.addChildNode(n); return n }()
            occ.geometry = solid
            w.geometry = lines
        }
    }
}
