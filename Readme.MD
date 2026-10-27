# USB Webcam – Face Detection (Rust)

Rust-port af det oprindelige Python-program (Tkinter + OpenCV). Programmet viser live billede fra et USB-webcam, finder ansigter med en Haar cascade og optager automatisk video, så længe der er et ansigt i billedet.

## Funktioner

- Live webcam-visning (egui/eframe)
- Ansigtsdetektion med OpenCV Haar cascade (kører hver N frame)
- Automatisk optagelse til mp4, når et ansigt ses
- Optagelsen stopper, når der ikke har været et ansigt i et indstillet antal sekunder
- Snapshot-knap (gemmer png) med hvidt flash
- Settings-vindue: detektion hver N frame og timeout for optagelse
- Overlay med FPS og CPU-forbrug, samt REC-indikator
- Logning til `logs/events.log`

## Mapper

Oprettes automatisk ved start i den mappe, programmet køres fra:

```
captures/videos/       optagelser (video_YYYYMMDD_HHMMSS.mp4)
captures/snapshots/    snapshots (snapshot_YYYYMMDD_HHMMSS.png)
logs/events.log        hændelseslog
```

## Projektstruktur

```
usbwebcam/
├── Cargo.toml
├── README.md
├── haarcascade_frontalface_default.xml   (bygges ind i programmet)
├── usbwebcam.png                         (valgfrit programikon)
└── src/
    └── main.rs
```

`haarcascade_frontalface_default.xml` skal ligge i projektets rodmappe, når der bygges. Filen indlejres i programmet med `include_bytes!`, så den ikke skal følge med den færdige binær. Filen findes i OpenCV's repository under `data/haarcascades/`, eller i en Python-installation af `opencv-python` under `cv2/data/`.

## Afhængigheder

### Alpine Linux

```
apk add rust cargo opencv-dev clang-dev llvm-dev pkgconf
```

Til kørsel skal der desuden være GUI- og grafikbiblioteker, fx `mesa-dri-gallium`, `libxkbcommon`, `wayland-libs-client` og `libx11`.

### Rust-crates

Se `Cargo.toml`:

- `eframe` (GUI)
- `opencv` (kamera, ansigtsdetektion, optagelse). Brug en version, der passer til systemets OpenCV. Med OpenCV 4.12 skal craten være ny nok til at kunne bygge bindings, og `cvt_color` kræver da et ekstra `AlgorithmHint`-argument
- `chrono` (tidsstempler)
- `sysinfo` (CPU-forbrug)

## Byg og kør

```
cargo run --release
```

Eller byg en binær:

```
cargo build --release
./target/release/usbwebcam
```

Start gerne programmet fra den mappe, hvor `captures/` og `logs/` skal ligge.

## Indstillinger

Standardværdier står øverst i `src/main.rs`:

| Konstant | Standard | Betydning |
|---|---|---|
| `FRAME_WIDTH` / `FRAME_HEIGHT` | 640 / 480 | Ønsket kameraopløsning |
| `RECORD_FPS` | 25 | Billedfrekvens i optagede videoer |
| `DEFAULT_DETECT_EVERY_N` | 6 | Detektion køres hver N frame |
| `DEFAULT_PERSON_TIMEOUT` | 3.0 | Sekunder uden ansigt, før optagelsen stopper |
| `DETECT_WIDTH` / `DETECT_HEIGHT` | 320 / 240 | Opløsning, billedet skaleres ned til under detektion |

Detektion og timeout kan også ændres live i Settings-vinduet.

## Fejlfinding

- **Kan ikke åbne webcam:** Prøv et andet kameraindex i `VideoCapture::new(0, ...)`, fx 1 eller 2. Tjek også, at brugeren har adgang til `/dev/video0` (gruppen `video`).
- **mp4 bliver ikke skrevet:** OpenCV er muligvis bygget uden ffmpeg-understøttelse. Skift til `.avi` med fourcc `MJPG`.
- **Mange fejl i `opencv`-craten ved bygning:** Craten passer ikke til systemets OpenCV eller clang. Opgrader craten, slet `Cargo.lock`, og kør `cargo clean -p opencv`.
- **`include_bytes!` kan ikke finde filen:** Tjek, at `haarcascade_frontalface_default.xml` ligger ved siden af `Cargo.toml`.

## Forskelle fra Python-versionen

- Ansigtsrammen bliver stående til næste detektion i stedet for at blinke.
- Optagelsen bruger kameraets faktiske opløsning.
- Overlay (FPS, REC, rammer) vises i live-billedet, men ligger ikke i den optagede video.