import Foundation
import Network
import Combine

/// Enlace con el programa de la PC por red local (Wi-Fi, hotspot del iPhone o de Windows, USB).
/// Encuentra la PC sola por Bonjour (_levanta._tcp) o usa la dirección escrita a mano,
/// envía los cuadros por WebSocket y recibe la malla fusionada en vivo.
final class PCLink: NSObject, ObservableObject, URLSessionWebSocketDelegate {

    enum State: Equatable {
        case off, searching, connecting(String), connected(String), failed(String)

        var text: String {
            switch self {
            case .off: return "Apagado"
            case .searching: return "Buscando la PC en la red…"
            case .connecting(let a): return "Conectando a \(a)…"
            case .connected(let a): return "Conectado a \(a)"
            case .failed(let m): return m
            }
        }
    }

    @Published private(set) var state: State = .off
    @Published private(set) var pcFrames = 0
    @Published private(set) var pcMessage = ""
    @Published private(set) var pcProcessing = false
    @Published var manualAddress: String = UserDefaults.standard.string(forKey: "pcAddress") ?? "" {
        didSet { UserDefaults.standard.set(manualAddress, forKey: "pcAddress") }
    }

    /// Llega cada malla en vivo (mensaje completo del protocolo).
    var onMesh: ((Data) -> Void)?
    /// Se llama al quedar conectado (para reenviar el inicio si ya se está escaneando).
    var onConnected: (() -> Void)?

    private var enabled = false
    private var browser: NWBrowser?
    private var resolver: NWConnection?
    private var task: URLSessionWebSocketTask?
    private lazy var urlSession = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    private let lock = NSLock()
    private var connectedFlag = false
    private var inflight = 0
    private var retryWork: DispatchWorkItem?

    var isConnected: Bool {
        lock.lock(); defer { lock.unlock() }
        return connectedFlag
    }

    // MARK: Encender / apagar

    func setEnabled(_ on: Bool) {
        enabled = on
        if on { connect() } else { disconnect(state: .off) }
    }

    func reconnect() {
        guard enabled else { return }
        disconnect(state: .searching)
        connect()
    }

    private func connect() {
        retryWork?.cancel()
        let manual = manualAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        if !manual.isEmpty {
            open(address: manual)
        } else {
            browse()
        }
    }

    private func disconnect(state newState: State) {
        retryWork?.cancel()
        browser?.cancel(); browser = nil
        resolver?.cancel(); resolver = nil
        task?.cancel(with: .goingAway, reason: nil); task = nil
        lock.lock(); connectedFlag = false; inflight = 0; lock.unlock()
        DispatchQueue.main.async { self.state = newState }
    }

    private func scheduleRetry(_ message: String) {
        lock.lock(); connectedFlag = false; lock.unlock()
        DispatchQueue.main.async { self.state = .failed(message) }
        guard enabled else { return }
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.enabled else { return }
            self.task = nil
            self.connect()
        }
        retryWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 3, execute: work)
    }

    // MARK: Descubrimiento (Bonjour)

    private func browse() {
        DispatchQueue.main.async { self.state = .searching }
        let b = NWBrowser(for: .bonjour(type: "_levanta._tcp", domain: nil), using: .tcp)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            guard let self, self.task == nil, self.resolver == nil, let first = results.first else { return }
            self.resolve(first.endpoint)
        }
        b.stateUpdateHandler = { [weak self] st in
            if case .failed = st { self?.scheduleRetry("No se pudo buscar en la red local. Revisa el permiso de Red local.") }
        }
        browser = b
        b.start(queue: .main)
    }

    /// Conecta un momento al servicio para conocer su IP y puerto reales.
    private func resolve(_ endpoint: NWEndpoint) {
        let params = NWParameters.tcp
        if let ip = params.defaultProtocolStack.internetProtocol as? NWProtocolIP.Options {
            ip.version = .v4
        }
        let c = NWConnection(to: endpoint, using: params)
        resolver = c
        c.stateUpdateHandler = { [weak self, weak c] st in
            guard let self, let c else { return }
            switch st {
            case .ready:
                if case let .hostPort(host, port)? = c.currentPath?.remoteEndpoint {
                    var h = "\(host)"
                    if let pct = h.firstIndex(of: "%") { h = String(h[..<pct]) }
                    c.cancel()
                    self.resolver = nil
                    self.browser?.cancel(); self.browser = nil
                    self.open(address: "\(h):\(port.rawValue)")
                }
            case .failed, .cancelled:
                self.resolver = nil
            default:
                break
            }
        }
        c.start(queue: .main)
    }

    // MARK: WebSocket

    private func open(address: String) {
        var addr = address
        if !addr.contains(":") { addr += ":8765" }
        guard let url = URL(string: "ws://\(addr)/ws") else {
            DispatchQueue.main.async { self.state = .failed("Dirección no válida: \(address)") }
            return
        }
        DispatchQueue.main.async { self.state = .connecting(addr) }
        let t = urlSession.webSocketTask(with: url)
        t.maximumMessageSize = 64 * 1024 * 1024
        task = t
        t.resume()
        receive(t)
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        guard webSocketTask === task else { return }
        lock.lock(); connectedFlag = true; inflight = 0; lock.unlock()
        let addr = webSocketTask.originalRequest?.url?.host ?? "PC"
        DispatchQueue.main.async {
            self.state = .connected(addr)
            self.onConnected?()
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        guard webSocketTask === task else { return }
        scheduleRetry("Se perdió la conexión con la PC. Reintentando…")
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard task === self.task, let error else { return }
        scheduleRetry("Sin conexión con la PC (\(error.localizedDescription)). Reintentando…")
    }

    private func receive(_ t: URLSessionWebSocketTask) {
        t.receive { [weak self] result in
            guard let self, t === self.task else { return }
            switch result {
            case .success(.data(let d)):
                self.handle(d)
                self.receive(t)
            case .success:
                self.receive(t)
            case .failure:
                break      // el delegado reporta el error y reintenta
            }
        }
    }

    private func handle(_ d: Data) {
        guard let u = FrameCodec.unpack(d) else { return }
        if u.kind == 10 {
            onMesh?(d)
        } else if u.kind == 11 {
            let frames = (u.header["cuadros"] as? NSNumber)?.intValue ?? 0
            let msg = u.header["mensaje"] as? String ?? ""
            let proc = (u.header["procesando"] as? NSNumber)?.boolValue ?? false
            DispatchQueue.main.async {
                self.pcFrames = frames
                self.pcMessage = msg
                self.pcProcessing = proc
            }
        }
    }

    // MARK: Envío

    /// Envía un mensaje. Si la red va atrasada, descarta cuadros no clave para no acumular retraso.
    func send(_ data: Data, important: Bool) {
        lock.lock()
        guard connectedFlag, let t = task else { lock.unlock(); return }
        if !important && inflight > 8 { lock.unlock(); return }
        inflight += 1
        lock.unlock()
        t.send(.data(data)) { [weak self] _ in
            guard let self else { return }
            self.lock.lock(); self.inflight = max(0, self.inflight - 1); self.lock.unlock()
        }
    }

    func sendStart(name: String) {
        let info: [String: Any] = ["nombre": name, "dispositivo": "iPhone",
                                   "sistema": ProcessInfo.processInfo.operatingSystemVersionString]
        if let d = FrameCodec.pack(kind: 2, header: info) { send(d, important: true) }
    }

    func sendEnd() {
        if let d = FrameCodec.pack(kind: 3, header: [:]) { send(d, important: true) }
    }
}
