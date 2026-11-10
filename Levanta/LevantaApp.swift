import SwiftUI
import UIKit

@main
struct LevantaApp: App {
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
        .onChange(of: phase, initial: true) { _, p in
            // mientras la app está abierta la pantalla no se apaga ni se bloquea
            // (escaneando, revisando o esperando a la PC no se toca la pantalla)
            UIApplication.shared.isIdleTimerDisabled = (p == .active)
        }
    }
}
