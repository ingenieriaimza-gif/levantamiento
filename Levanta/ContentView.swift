import SwiftUI
import ARKit

private let accent = Color(red: 0.96, green: 0.66, blue: 0.13)   // ámbar de topografía

struct ContentView: View {
    @StateObject private var engine = ScanEngine()
    @State private var showSettings = false
    @State private var showScans = false
    @State private var saving = false
    @State private var confirmReset = false
    @State private var toast: String?

    var body: some View {
        ZStack {
            ARContainer(engine: engine)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                topBar
                Spacer()
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
        .confirmationDialog("¿Borrar el escaneo actual?", isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Borrar y empezar de nuevo", role: .destructive) { engine.reset() }
            Button("Cancelar", role: .cancel) {}
        } message: {
            Text("Los puntos que no hayas guardado se perderán.")
        }
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
                Text("vóxel \(engine.voxelSizeCm, specifier: "%.1f") cm")
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

            Button {
                showScans = true
            } label: {
                Label("Mis escaneos", systemImage: "folder")
                    .font(.footnote.weight(.semibold))
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
            saving = false
            switch result {
            case .success(let url):
                showToast("Guardado: \(url.lastPathComponent)")
            case .failure(let error):
                engine.statusMessage = error.localizedDescription
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
                        Text("0.5 cm (detalle)").tag(0.5)
                        Text("1 cm (recomendado)").tag(1.0)
                        Text("2 cm (cuartos grandes)").tag(2.0)
                        Text("3 cm (exteriores)").tag(3.0)
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
                    Stepper("Lecturas mínimas por punto: \(engine.minObservations)",
                            value: $engine.minObservations, in: 1...5)
                        .onChange(of: engine.minObservations) { _, v in
                            engine.updateMinObservations(v)
                        }
                } header: {
                    Text("Precisión y ruido")
                } footer: {
                    Text("El LiDAR es más preciso de cerca: a más de 3–4 m aumenta el error. Pedir 2 o más lecturas por punto elimina destellos y puntos sueltos.")
                }
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
