import SwiftUI
import ARKit

private let accent = Color(red: 0.96, green: 0.66, blue: 0.13)   // ámbar de topografía

struct ContentView: View {
    @StateObject private var engine = ScanEngine()
    @State private var showSettings = false
    @State private var showScans = false
    @State private var saving = false
    @State private var confirmReset = false
    @State private var showReview = false
    @State private var toast: String?

    var body: some View {
        ZStack {
            ARContainer(engine: engine)
                .ignoresSafeArea()

            reticle

            VStack(spacing: 12) {
                topBar
                if engine.streamToPC {
                    PCStatusPill(link: engine.link, triangles: engine.liveMeshTriangles)
                    AppUpdateBanner(link: engine.link)
                }
                if engine.colorMode == .coverage && (engine.isScanning || engine.pointCount > 0) {
                    QualityLegend()
                }
                Spacer()
                if engine.isScanning && engine.personInView {
                    Label("Persona detectada: se ignora", systemImage: "person.fill.xmark")
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(Color(red: 0.35, green: 0.69, blue: 1.0).opacity(0.9), in: Capsule())
                        .foregroundStyle(.black)
                        .transition(.opacity)
                }
                if engine.isScanning && engine.holdingStill && !engine.tooFast {
                    Label("Sin movimiento: en pausa hasta que te muevas", systemImage: "pause.circle.fill")
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                        .transition(.opacity)
                }
                if engine.isScanning && engine.tooFast {
                    Label("Más despacio para fotos nítidas", systemImage: "tortoise.fill")
                        .font(.subheadline.weight(.bold))
                        .padding(.horizontal, 16).padding(.vertical, 10)
                        .background(Color(red: 0.95, green: 0.35, blue: 0.25).opacity(0.92), in: Capsule())
                        .foregroundStyle(.white)
                        .transition(.opacity)
                }
                if let msg = engine.statusMessage {
                    Text(msg)
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.red.opacity(0.85), in: Capsule())
                        .onTapGesture { engine.statusMessage = nil }
                }
                if let t = toast {
                    Text(t)
                        .font(.footnote.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                        .transition(.opacity)
                }
                bottomBar
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(engine: engine)
                .presentationDetents([.medium, .large])
        }
        .sheet(isPresented: $showScans) {
            ScansView()
        }
        .fullScreenCover(isPresented: $showReview) {
            ReviewView(engine: engine)
        }
        .confirmationDialog("¿Borrar el escaneo actual?", isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Borrar y empezar de nuevo", role: .destructive) { engine.reset() }
            Button("Cancelar", role: .cancel) {}
        } message: {
            Text("Los puntos que no hayas guardado se perderán.")
        }
    }

    // MARK: Mira central con distancia

    private var reticle: some View {
        let d = engine.centerDistance
        let color: Color
        let hint: String
        if let d {
            if d <= 1.5 { color = .green; hint = "precisión alta" }
            else if d <= 3.0 { color = accent; hint = "precisión media" }
            else { color = .red; hint = "muy lejos" }
        } else {
            color = .white.opacity(0.6); hint = "sin lectura"
        }
        return VStack(spacing: 6) {
            ZStack {
                Circle().stroke(color, lineWidth: 2).frame(width: 34, height: 34)
                Circle().fill(color).frame(width: 4, height: 4)
            }
            VStack(spacing: 0) {
                Text(d.map { String(format: "%.2f m", $0) } ?? "–")
                    .font(.system(size: 15, weight: .bold, design: .monospaced))
                Text(hint).font(.caption2)
            }
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(.black.opacity(0.45), in: Capsule())
            .foregroundStyle(color)
        }
        .allowsHitTesting(false)
    }

    // MARK: Barra superior

    private var topBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 6) {
                Circle()
                    .fill(engine.trackingOK ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(engine.trackingText)
                    .font(.footnote.weight(.medium))
                    .lineLimit(1)
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())

            Spacer()

            Button {
                engine.colorMode = engine.colorMode == .coverage ? .real : .coverage
            } label: {
                Image(systemName: engine.colorMode == .coverage ? "checkmark.seal.fill" : "paintpalette.fill")
                    .frame(width: 40, height: 40)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel(engine.colorMode == .coverage ? "Ver color real" : "Ver cobertura")

            Button {
                engine.showCloud.toggle()
            } label: {
                Image(systemName: engine.showCloud ? "circle.grid.3x3.fill" : "circle.grid.3x3")
                    .frame(width: 40, height: 40)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel(engine.showCloud ? "Ocultar nube" : "Mostrar nube")

            Button {
                showSettings = true
            } label: {
                Image(systemName: "desktopcomputer")
                    .frame(width: 40, height: 40)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel("Conexión con la PC")
        }
        .foregroundStyle(.white)
    }

    // MARK: Barra inferior

    private var bottomBar: some View {
        VStack(spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(formatCount(engine.pointCount))
                    .font(.system(size: 34, weight: .bold, design: .monospaced))
                Text("puntos")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("vista previa")
                    Text("\(engine.keyframeCount) fotos")
                }
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            }

            HStack(spacing: 18) {
                circleButton("trash", label: "Borrar") { confirmReset = true }
                    .disabled(engine.isScanning)

                Button {
                    engine.toggleScanning()
                } label: {
                    ZStack {
                        Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                        if engine.isScanning {
                            RoundedRectangle(cornerRadius: 6).fill(accent).frame(width: 30, height: 30)
                        } else {
                            Circle().fill(accent).frame(width: 62, height: 62)
                        }
                    }
                }
                .disabled(!engine.lidarAvailable)
                .accessibilityLabel(engine.isScanning ? "Pausar escaneo" : "Iniciar escaneo")

                circleButton(saving ? "hourglass" : "square.and.arrow.down", label: "Guardar") {
                    save()
                }
                .disabled(engine.isScanning || saving || engine.pointCount == 0)
            }
            .frame(maxWidth: .infinity)

            if engine.streamToPC {
                PCFinishButton(engine: engine, link: engine.link)
            }

            HStack(spacing: 24) {
                Button {
                    showReview = true
                } label: {
                    Label("Revisar 3D", systemImage: "rotate.3d")
                        .font(.footnote.weight(.semibold))
                }
                .disabled(engine.isScanning)

                Button {
                    showScans = true
                } label: {
                    Label("Mis escaneos", systemImage: "folder")
                        .font(.footnote.weight(.semibold))
                }
            }
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 22))
        .foregroundStyle(.white)
    }

    private func circleButton(_ icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.title3)
                    .frame(width: 52, height: 52)
                    .background(Color.white.opacity(0.12), in: Circle())
                Text(label).font(.caption2)
            }
        }
    }

    private func save() {
        saving = true
        engine.exportPLY { result in
            var message = ""
            switch result {
            case .success(let url):
                message = "Nube guardada: \(url.lastPathComponent)"
            case .failure(let error):
                engine.statusMessage = error.localizedDescription
            }
            engine.finishRGBD { rgbd in
                saving = false
                switch rgbd {
                case .success(let parts):
                    if !parts.isEmpty {
                        message += message.isEmpty ? "" : " · "
                        message += parts.count == 1 ? "Paquete RGB-D listo" : "Paquete RGB-D en \(parts.count) partes"
                    }
                case .failure(let error):
                    engine.statusMessage = "RGB-D: \(error.localizedDescription)"
                }
                if !message.isEmpty { showToast(message) }
            }
        }
    }

    private func showToast(_ text: String) {
        withAnimation { toast = text }
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            withAnimation { toast = nil }
        }
    }

    private func formatCount(_ n: Int) -> String {
        if n >= 1_000_000 { return String(format: "%.2f M", Double(n) / 1_000_000) }
        if n >= 1_000 { return String(format: "%.0f k", Double(n) / 1_000) }
        return "\(n)"
    }
}

// MARK: - Vista AR

struct ARContainer: UIViewRepresentable {
    let engine: ScanEngine
    func makeUIView(context: Context) -> ARSCNView { engine.makeView() }
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - Ajustes

struct SettingsView: View {
    @ObservedObject var engine: ScanEngine
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                PCSettingsSection(engine: engine, link: engine.link)
                Section {
                    Text("La app ya viene con la mejor configuración: lecturas de alta confianza a menos de 3.5 m, 2 lecturas por punto, fotos nítidas con color fijo, sin personas, borra el ruido y se pausa si no te mueves. La calidad final la hace la PC.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Conexión con la PC")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Listo") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Escaneos guardados

struct ScansView: View {
    @State private var scans: [URL] = []
    @State private var toDelete: URL?
    @State private var confirmAll = false
    @State private var error: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if scans.isEmpty {
                    Text("Aún no hay escaneos guardados en el iPhone.")
                        .foregroundStyle(.secondary)
                } else {
                    Section {
                        ForEach(scans, id: \.self) { url in
                            HStack(spacing: 14) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(url.deletingPathExtension().lastPathComponent)
                                        .font(.subheadline.weight(.medium))
                                    Text(ScanStore.sizeText(url))
                                        .font(.caption.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                ShareLink(item: url) {
                                    Image(systemName: "square.and.arrow.up")
                                        .frame(width: 36, height: 36)
                                }
                                .buttonStyle(.borderless)
                                .accessibilityLabel("Compartir")
                                Button(role: .destructive) {
                                    toDelete = url
                                } label: {
                                    Image(systemName: "trash")
                                        .frame(width: 36, height: 36)
                                        .foregroundStyle(.red)
                                }
                                .buttonStyle(.borderless)
                                .accessibilityLabel("Borrar")
                            }
                        }
                        .onDelete { idx in
                            for i in idx { remove(scans[i]) }
                        }
                    } footer: {
                        Text("Ocupan \(totalText) en el iPhone. Si ya los procesaste en la PC, puedes borrarlos aquí; en la PC se conservan.")
                    }
                }
                if let error {
                    Text(error).font(.footnote).foregroundStyle(.red)
                }
            }
            .navigationTitle("Mis escaneos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    if !scans.isEmpty {
                        Button("Borrar todos", role: .destructive) { confirmAll = true }
                            .tint(.red)
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Cerrar") { dismiss() }
                }
            }
            .confirmationDialog("¿Borrar este escaneo del iPhone?",
                                isPresented: Binding(get: { toDelete != nil }, set: { if !$0 { toDelete = nil } }),
                                titleVisibility: .visible) {
                Button("Borrar", role: .destructive) {
                    if let u = toDelete { remove(u) }
                    toDelete = nil
                }
                Button("Cancelar", role: .cancel) { toDelete = nil }
            } message: {
                Text(toDelete?.lastPathComponent ?? "")
            }
            .confirmationDialog("¿Borrar todos los escaneos del iPhone?", isPresented: $confirmAll,
                                titleVisibility: .visible) {
                Button("Borrar \(scans.count) escaneos", role: .destructive) {
                    for u in scans { remove(u, reload: false) }
                    scans = ScanStore.list()
                }
                Button("Cancelar", role: .cancel) {}
            } message: {
                Text("Los que ya están en la PC se conservan allá.")
            }
            .onAppear { scans = ScanStore.list() }
        }
    }

    private var totalText: String {
        let bytes = scans.reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    private func remove(_ url: URL, reload: Bool = true) {
        do {
            try FileManager.default.removeItem(at: url)
            error = nil
        } catch {
            self.error = "No se pudo borrar \(url.lastPathComponent): \(error.localizedDescription)"
        }
        if reload { scans = ScanStore.list() }
    }
}


// MARK: - Leyenda de calidad

struct QualityLegend: View {
    var body: some View {
        HStack(spacing: 12) {
            item(ScanEngine.qualityGood, "Bien (de cerca)")
            item(ScanEngine.qualityFair, "Acércate")
            item(ScanEngine.qualityPoor, "Falta")
        }
        .font(.caption2.weight(.semibold))
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
        .foregroundStyle(.white)
        .accessibilityElement(children: .combine)
    }

    private func item(_ c: SIMD3<Float>, _ t: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))).frame(width: 9, height: 9)
            Text(t)
        }
    }
}

// MARK: - PC (red local)

struct PCStatusPill: View {
    @ObservedObject var link: PCLink
    let triangles: Int

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
            VStack(alignment: .leading, spacing: 1) {
                Text(link.state.text).font(.footnote.weight(.semibold)).lineLimit(1)
                if case .connected = link.state {
                    Text(detail).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        .foregroundStyle(.white)
    }

    private var icon: String {
        switch link.state {
        case .connected: return "desktopcomputer"
        case .failed: return "exclamationmark.triangle.fill"
        default: return "antenna.radiowaves.left.and.right"
        }
    }

    private var detail: String {
        var parts = ["PC: \(link.pcFrames) cuadros"]
        if triangles > 0 { parts.append("superficie \(triangles / 1000) k triángulos") }
        if link.pcProcessing { parts = [link.pcMessage] }
        return parts.joined(separator: " · ")
    }
}

struct PCFinishButton: View {
    @ObservedObject var engine: ScanEngine
    @ObservedObject var link: PCLink
    @State private var confirm = false

    var body: some View {
        if link.pcProcessing {
            VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Creando el modelo en la PC").font(.subheadline.weight(.semibold))
                    Spacer()
                    Text(link.pcProgress.map { "\(Int($0 * 100)) %" } ?? "…")
                        .font(.subheadline.monospacedDigit().weight(.bold))
                }
                ProgressView(value: link.pcProgress ?? 0)
                    .tint(Color(red: 0.96, green: 0.66, blue: 0.13))
                HStack {
                    Text(link.pcStage).lineLimit(1)
                    Spacer()
                    Text(remainingText)
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            }
            .padding(12)
            .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
        } else {
            finishButton
        }
    }

    private var remainingText: String {
        guard let r = link.pcRemaining else { return "calculando…" }
        if r < 60 { return "falta menos de 1 min" }
        let m = Int((Double(r) / 60).rounded())
        return m < 60 ? "faltan ~\(m) min" : "faltan ~\(m / 60) h \(m % 60) min"
    }

    private var finishButton: some View {
        Button {
            confirm = true
        } label: {
            Label("Terminar y crear el modelo en la PC", systemImage: "checkmark.circle.fill")
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
        }
        .disabled(!link.isConnected || link.pcFrames == 0)
        .confirmationDialog("¿Terminar el escaneo?", isPresented: $confirm, titleVisibility: .visible) {
            Button("Terminar y crear el modelo") { engine.finishOnPC() }
            Button("Seguir escaneando", role: .cancel) {}
        } message: {
            Text("La PC creará los archivos para Navisworks y SolidWorks y el modelo con fotos. Verás el avance aquí y en la PC.")
        }
    }
}

struct PCSettingsSection: View {
    @ObservedObject var engine: ScanEngine
    @ObservedObject var link: PCLink

    var body: some View {
        Section {
            Text(link.state.text).font(.footnote).foregroundStyle(.secondary)
            TextField("Dirección manual (ej. 192.168.137.1:8765)", text: $link.manualAddress)
                .keyboardType(.numbersAndPunctuation)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Reconectar") { link.reconnect() }
        } header: {
            Text("Procesar en la PC")
        } footer: {
            Text("Versión de la app: \(PCLink.ownVersion) (build \(PCLink.ownBuild)). ") +
            Text("Abre el programa Levantamiento 3D en tu PC. Funciona con el mismo Wi-Fi, con el hotspot del iPhone (también por cable USB) o con la zona con cobertura inalámbrica de Windows; no necesita internet. Si la app no encuentra la PC sola, escribe la dirección que muestra el programa.")
        }
    }
}


// MARK: - Aviso de versión nueva guardada en la PC

struct AppUpdateBanner: View {
    @ObservedObject var link: PCLink
    @State private var showHelp = false

    var body: some View {
        if link.hasNewerAppOnPC {
            Button {
                showHelp = true
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.down.app.fill").font(.title3)
                    VStack(alignment: .leading, spacing: 1) {
                        Text("Hay una versión nueva en tu PC").font(.footnote.weight(.bold))
                        Text("\(link.pcAppVersion) (build \(link.pcAppBuild)) · tienes la build \(PCLink.ownBuild)")
                            .font(.caption2)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right").font(.caption)
                }
                .padding(.horizontal, 12).padding(.vertical, 9)
                .background(Color(red: 0.96, green: 0.66, blue: 0.13).opacity(0.95), in: RoundedRectangle(cornerRadius: 14))
                .foregroundStyle(Color(red: 0.1, green: 0.07, blue: 0.02))
            }
            .sheet(isPresented: $showHelp) {
                NavigationStack {
                    List {
                        Section {
                            Text("Apple no permite que una app se actualice sola fuera de la App Store, así que la instalación se hace desde la PC con tu Apple ID. Tus escaneos guardados se conservan.")
                        }
                        Section("Pasos") {
                            Label("Conecta el iPhone a la PC por cable.", systemImage: "cable.connector")
                            Label("En el programa de la PC: Actualizaciones › Instalar en el iPhone. Se abre Sideloadly con la app nueva.", systemImage: "desktopcomputer")
                            Label("Pulsa Start en Sideloadly y espera a que termine.", systemImage: "play.circle")
                        }
                    }
                    .navigationTitle("Actualizar la app")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) { Button("Listo") { showHelp = false } }
                    }
                }
                .presentationDetents([.medium])
            }
        }
    }
}
