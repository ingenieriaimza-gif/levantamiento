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
                }
                Spacer()
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
                Image(systemName: "slider.horizontal.3")
                    .frame(width: 40, height: 40)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .accessibilityLabel("Ajustes")
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
                    Text("vóxel \(Int((engine.voxelSizeCm * 10).rounded())) mm")
                    if engine.recordRGBD {
                        Text("\(engine.keyframeCount) cuadros RGB-D")
                    }
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
                .disabled(engine.isScanning || engine.pointCount == 0)

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
                Section {
                    Picker("Tamaño de vóxel", selection: $engine.voxelSizeCm) {
                        Text("2 mm (piezas pequeñas, a menos de 0.5 m)").tag(0.2)
                        Text("3 mm (máximo detalle)").tag(0.3)
                        Text("5 mm (recomendado)").tag(0.5)
                        Text("1 cm (cuartos)").tag(1.0)
                        Text("2 cm (espacios grandes)").tag(2.0)
                    }
                } footer: {
                    Text("Un punto promedio por cada cubito de este tamaño. Se aplica al borrar y empezar un escaneo nuevo.")
                }

                Section {
                    VStack(alignment: .leading) {
                        Text("Distancia máxima: \(engine.maxDepth, specifier: "%.1f") m")
                        Slider(value: $engine.maxDepth, in: 1...5, step: 0.5)
                    }
                    Toggle("Solo puntos de alta confianza", isOn: $engine.highConfidenceOnly)
                    Toggle("Profundidad suavizada", isOn: $engine.smoothDepth)
                    Toggle("Guardar cuadros RGB-D (máxima calidad)", isOn: $engine.recordRGBD)
                    Toggle("Fotos nítidas (exposición y color fijos)", isOn: $engine.sharpCapture)
                    Stepper("Lecturas mínimas por punto: \(engine.minObservations)",
                            value: $engine.minObservations, in: 1...5)
                        .onChange(of: engine.minObservations) { _, v in
                            engine.updateMinObservations(v)
                        }
                } header: {
                    Text("Precisión y ruido")
                } footer: {
                    Text("El LiDAR es más preciso de cerca: a más de 3–4 m aumenta el error. Pedir 2 o más lecturas por punto elimina destellos y puntos sueltos. La profundidad suavizada da menos ruido pero redondea bordes y detalles finos. Fotos nítidas fija una exposición corta al empezar a escanear: menos barrido al moverte y el mismo color en todas las fotos.")
                }

                PCSettingsSection(engine: engine, link: engine.link)
            }
            .navigationTitle("Ajustes")
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
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if scans.isEmpty {
                    Text("Aún no hay escaneos guardados.")
                        .foregroundStyle(.secondary)
                }
                ForEach(scans, id: \.self) { url in
                    HStack {
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
                        }
                    }
                }
                .onDelete { idx in
                    for i in idx { try? FileManager.default.removeItem(at: scans[i]) }
                    scans = ScanStore.list()
                }
            }
            .navigationTitle("Mis escaneos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Cerrar") { dismiss() }
                }
            }
            .onAppear { scans = ScanStore.list() }
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
        Button {
            confirm = true
        } label: {
            Label(link.pcProcessing ? "La PC está procesando…" : "Terminar y procesar en la PC",
                  systemImage: "checkmark.circle.fill")
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
        }
        .disabled(!link.isConnected || link.pcProcessing || link.pcFrames == 0)
        .confirmationDialog("¿Terminar el escaneo?", isPresented: $confirm, titleVisibility: .visible) {
            Button("Terminar y procesar") { engine.finishOnPC() }
            Button("Seguir escaneando", role: .cancel) {}
        } message: {
            Text("La PC generará la nube E57, el STL para SolidWorks, el modelo con fotos y los datos para Gaussian Splatting.")
        }
    }
}

struct PCSettingsSection: View {
    @ObservedObject var engine: ScanEngine
    @ObservedObject var link: PCLink

    var body: some View {
        Section {
            Toggle("Enviar a la PC por red local", isOn: $engine.streamToPC)
            if engine.streamToPC {
                Text(link.state.text).font(.footnote).foregroundStyle(.secondary)
                TextField("Dirección manual (ej. 192.168.137.1:8765)", text: $link.manualAddress)
                    .keyboardType(.numbersAndPunctuation)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Reconectar") { link.reconnect() }
            }
        } header: {
            Text("Procesar en la PC")
        } footer: {
            Text("Abre el programa Levantamiento 3D en tu PC. Funciona con el mismo Wi-Fi, con el hotspot del iPhone (también por cable USB) o con la zona con cobertura inalámbrica de Windows; no necesita internet. Si la app no encuentra la PC sola, escribe la dirección que muestra el programa.")
        }
    }
}
