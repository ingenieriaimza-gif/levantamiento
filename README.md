# Levantamiento 3D — escáner LiDAR propio para iPhone

App propia que usa directamente el sensor LiDAR del iPhone (ARKit) para crear nubes de puntos
con color, en metros reales, y guardarlas como PLY.

## Qué hace la versión 0.3

- Más detalle: lee todos los píxeles del LiDAR, ~11 cuadros por segundo, vóxeles de 2, 3 y 5 mm, y profundidad sin suavizar por defecto.
- Icono propio y nombre "Levantamiento 3D".

- **Modo cobertura:** puntos en rojo (1 lectura), ámbar (2–3) y verde (4 o más). Lo que no tiene puntos, falta.
- **Mira con distancia en vivo:** verde hasta 1.5 m, ámbar hasta 3 m, rojo más lejos.
- **Revisar 3D:** al pausar, la nube se gira y acerca con los dedos, con vistas de planta y frente y medidas generales.

- Lee la profundidad del LiDAR ~8 veces por segundo y la convierte en puntos 3D con color.
- Acumula los puntos en una rejilla de vóxeles (0.5–3 cm): un punto promedio por cubito, sin duplicados.
- Muestra la nube creciendo en vivo sobre la cámara, para ver qué zonas faltan.
- Filtros de ruido: solo puntos de alta confianza, distancia máxima y lecturas mínimas por punto.
- Guarda en PLY binario (x, y, z en metros, Y hacia arriba, + RGB) en la carpeta *Escaneos*,
  visible en la app Archivos y compartible a Claude, AirDrop, correo, etc.

## Cómo se compila (sin Mac)

Cada vez que se sube un cambio a la rama `main` de GitHub, GitHub Actions compila la app en una
Mac en la nube y publica `Levanta.ipa` en la sección **Releases** del repositorio.

## Cómo instalarla en tu iPhone desde Windows

1. Descarga **Sideloadly** desde https://sideloadly.io e instálalo en tu PC.
   Sideloadly pide tener iTunes e iCloud instalados (las versiones descargadas de apple.com).
2. Descarga `Levanta.ipa` desde **Releases** del repositorio.
3. Conecta el iPhone por cable a la PC y acepta "Confiar en esta computadora".
4. Abre Sideloadly, arrastra `Levanta.ipa`, escribe tu Apple ID y pulsa **Start**.
5. En el iPhone:
   - Ajustes → Privacidad y seguridad → **Modo de desarrollador** → activar (el iPhone se reinicia).
   - Ajustes → General → **VPN y gestión de dispositivos** → tu Apple ID → **Confiar**.
6. Abre Levanta y permite el acceso a la cámara.

Con Apple ID gratuito la app deja de abrir a los 7 días: vuelve a hacer el paso 4 (tus escaneos
guardados se conservan). Con una cuenta de Apple Developer ($99 USD/año) dura un año.

## Consejos de escaneo

- Muévete despacio y a 0.5–3 m de las superficies; el error crece a más distancia.
- Recorre cada zona desde dos ángulos para que cada punto tenga 2+ lecturas.
- Evita vidrios, espejos y superficies negras brillantes.

## Estructura

- `Levanta/ScanEngine.swift` — captura LiDAR, vóxeles, vista previa y exportación PLY.
- `Levanta/ContentView.swift` — interfaz.
- `project.yml` — proyecto de Xcode (se genera con XcodeGen).
- `.github/workflows/build.yml` — compilación en la nube.
