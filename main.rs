use chrono::Local;
use eframe::egui;
use opencv::{core, imgcodecs, imgproc, objdetect, prelude::*, videoio};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};
use sysinfo::System;

// ---------------------------
// Konfiguration (defaults)
// ---------------------------
const FRAME_WIDTH: f64 = 640.0;
const FRAME_HEIGHT: f64 = 480.0;
const RECORD_FPS: f64 = 25.0;

const DEFAULT_PERSON_TIMEOUT: f32 = 3.0;
const DEFAULT_DETECT_EVERY_N: u32 = 6;

const DETECT_WIDTH: i32 = 320;
const DETECT_HEIGHT: i32 = 240;

// Haar cascade bygges ind i programmet
const CASCADE_XML: &[u8] = include_bytes!("../haarcascade_frontalface_default.xml");

// ---------------------------
// Mapper
// ---------------------------
fn video_dir() -> PathBuf {
    PathBuf::from("captures/videos")
}
fn snap_dir() -> PathBuf {
    PathBuf::from("captures/snapshots")
}
fn log_dir() -> PathBuf {
    PathBuf::from("logs")
}

// ---------------------------
// Logging
// ---------------------------
fn log_event(msg: &str) {
    let line = format!("{} | {}\n", Local::now().format("%Y-%m-%d %H:%M:%S"), msg);
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir().join("events.log"))
    {
        let _ = f.write_all(line.as_bytes());
    }
}

// ---------------------------
// Cascade
// ---------------------------
fn load_cascade() -> opencv::Result<objdetect::CascadeClassifier> {
    let path = std::env::temp_dir().join("usbwebcam_haarcascade_frontalface_default.xml");
    fs::write(&path, CASCADE_XML).map_err(|e| {
        opencv::Error::new(core::StsError, format!("Kan ikke skrive cascade-fil: {e}"))
    })?;
    let cascade = objdetect::CascadeClassifier::new(path.to_str().unwrap_or_default())?;
    if cascade.empty()? {
        return Err(opencv::Error::new(
            core::StsError,
            "Kunne ikke indlæse Haar cascade",
        ));
    }
    Ok(cascade)
}

// ---------------------------
// App
// ---------------------------
struct WebcamApp {
    // Settings (kan ændres live)
    person_timeout: f32,
    detect_every_n: u32,

    // State
    video: videoio::VideoCapture,
    writer: Option<videoio::VideoWriter>,
    recording: bool,
    last_face_time: Option<Instant>,
    frame_count: u64,
    last_time: Instant,
    flash_until: Option<Instant>,
    cpu_load: f32,
    last_cpu_refresh: Instant,
    sys: System,
    faces: Vec<core::Rect>,
    last_frame: Mat,
    status: String,

    // Face detector
    face_cascade: objdetect::CascadeClassifier,

    // UI
    texture: Option<egui::TextureHandle>,
    show_settings: bool,
    tmp_detect_n: u32,
    tmp_timeout: f32,
}

impl WebcamApp {
    fn new() -> opencv::Result<Self> {
        let face_cascade = load_cascade()?;

        let mut video = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
        video.set(videoio::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)?;
        video.set(videoio::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)?;
        if !video.is_opened()? {
            return Err(opencv::Error::new(core::StsError, "Kan ikke åbne webcam"));
        }

        log_event("App startet");

        Ok(Self {
            person_timeout: DEFAULT_PERSON_TIMEOUT,
            detect_every_n: DEFAULT_DETECT_EVERY_N,
            video,
            writer: None,
            recording: false,
            last_face_time: None,
            frame_count: 0,
            last_time: Instant::now(),
            flash_until: None,
            cpu_load: 0.0,
            last_cpu_refresh: Instant::now(),
            sys: System::new(),
            faces: Vec::new(),
            last_frame: Mat::default(),
            status: "Klar".into(),
            face_cascade,
            texture: None,
            show_settings: false,
            tmp_detect_n: DEFAULT_DETECT_EVERY_N,
            tmp_timeout: DEFAULT_PERSON_TIMEOUT,
        })
    }

    // ---------------------------
    // Snapshot
    // ---------------------------
    fn snapshot(&mut self) {
        if self.last_frame.empty() {
            return;
        }
        let name = snap_dir().join(format!(
            "snapshot_{}.png",
            Local::now().format("%Y%m%d_%H%M%S")
        ));
        let ok = imgcodecs::imwrite(
            &name,
            &self.last_frame,
            &core::Vector::new(),
        )
        .unwrap_or(false);
        if ok {
            self.flash_until = Some(Instant::now() + Duration::from_millis(150));
            self.status = "📸 Snapshot gemt".into();
            log_event(&format!("Snapshot gemt: {}", name.display()));
        } else {
            self.status = "Kunne ikke gemme snapshot".into();
        }
    }

    // ---------------------------
    // Recording
    // ---------------------------
    fn start_recording(&mut self, size: core::Size) {
        let name = video_dir().join(format!(
            "video_{}.mp4",
            Local::now().format("%Y%m%d_%H%M%S")
        ));
        let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v').unwrap_or(0);
        match videoio::VideoWriter::new(&name, fourcc, RECORD_FPS, size, true) {
            Ok(w) => {
                self.writer = Some(w);
                self.recording = true;
                log_event(&format!("Optagelse startet: {}", name.display()));
            }
            Err(e) => log_event(&format!("Kunne ikke starte optagelse: {e}")),
        }
    }

    fn stop_recording(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.release();
        }
        self.writer = None;
        self.recording = false;
        log_event("Optagelse stoppet");
    }

    // ---------------------------
    // Ansigtsdetektion
    // ---------------------------
    fn detect_faces(&mut self, frame: &Mat) -> opencv::Result<Vec<core::Rect>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(
            frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut small = Mat::default();
        imgproc::resize(
            &gray,
            &mut small,
            core::Size::new(DETECT_WIDTH, DETECT_HEIGHT),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut found = core::Vector::<core::Rect>::new();
        self.face_cascade.detect_multi_scale(
            &small,
            &mut found,
            1.2,
            5,
            0,
            core::Size::new(40, 40),
            core::Size::new(0, 0),
        )?;

        let sx = frame.cols() as f64 / DETECT_WIDTH as f64;
        let sy = frame.rows() as f64 / DETECT_HEIGHT as f64;
        Ok(found
            .iter()
            .map(|r| {
                core::Rect::new(
                    (r.x as f64 * sx) as i32,
                    (r.y as f64 * sy) as i32,
                    (r.width as f64 * sx) as i32,
                    (r.height as f64 * sy) as i32,
                )
            })
            .collect())
    }

    // ---------------------------
    // Main loop (et frame)
    // ---------------------------
    fn process_frame(&mut self, ctx: &egui::Context) -> opencv::Result<()> {
        let mut frame = Mat::default();
        if !self.video.read(&mut frame)? || frame.empty() {
            return Ok(());
        }

        let now = Instant::now();
        self.frame_count += 1;

        if now.duration_since(self.last_cpu_refresh) > Duration::from_millis(500) {
            self.sys.refresh_cpu_usage();
            self.cpu_load = self.sys.global_cpu_usage();
            self.last_cpu_refresh = now;
        }

        self.last_frame = frame.clone();

        // Detektion hver N frame
        if self.frame_count % self.detect_every_n.max(1) as u64 == 0 {
            self.faces = self.detect_faces(&frame)?;
            if !self.faces.is_empty() {
                self.last_face_time = Some(now);
                if !self.recording {
                    self.start_recording(frame.size()?);
                }
            }
        }

        if self.recording {
            let timed_out = match self.last_face_time {
                Some(t) => now.duration_since(t).as_secs_f32() > self.person_timeout,
                None => true,
            };
            if timed_out {
                self.stop_recording();
            }
        }

        // Optag ren frame (uden overlay), som i originalen
        if self.recording {
            if let Some(w) = self.writer.as_mut() {
                w.write(&frame)?;
            }
        }

        // Overlay til visning
        let orange = core::Scalar::new(0.0, 128.0, 255.0, 0.0); // BGR
        for r in &self.faces {
            imgproc::rectangle(&mut frame, *r, orange, 2, imgproc::LINE_8, 0)?;
        }

        let dt = now.duration_since(self.last_time).as_secs_f32().max(0.001);
        let fps = (1.0 / dt) as i32;
        self.last_time = now;

        imgproc::put_text(
            &mut frame,
            &format!("{} FPS | CPU {:.0}%", fps, self.cpu_load),
            core::Point::new(10, 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
        if self.recording {
            imgproc::put_text(
                &mut frame,
                "REC",
                core::Point::new(10, 42),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        // BGR -> RGBA til egui
        let (w, h) = (frame.cols() as usize, frame.rows() as usize);
        let bytes = frame.data_bytes()?;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for px in bytes.chunks_exact(3) {
            rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
        }
        let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
        match self.texture.as_mut() {
            Some(t) => t.set(image, egui::TextureOptions::LINEAR),
            None => {
                self.texture = Some(ctx.load_texture("cam", image, egui::TextureOptions::LINEAR))
            }
        }

        Ok(())
    }
}

impl eframe::App for WebcamApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Err(e) = self.process_frame(ctx) {
            self.status = format!("Fejl: {e}");
        }

        egui::TopBottomPanel::bottom("controls").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.button("📸 Snapshot").clicked() {
                    self.snapshot();
                }
                if ui.button("⚙️ Settings").clicked() {
                    self.tmp_detect_n = self.detect_every_n;
                    self.tmp_timeout = self.person_timeout;
                    self.show_settings = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("❌ Afslut").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
            ui.label(&self.status);
            ui.add_space(2.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.texture {
                let resp = ui.image((tex.id(), tex.size_vec2()));
                if let Some(until) = self.flash_until {
                    if Instant::now() < until {
                        ui.painter().rect_filled(
                            resp.rect,
                            0.0,
                            egui::Color32::from_white_alpha(120),
                        );
                    } else {
                        self.flash_until = None;
                    }
                }
            }
        });

        // Settings-vindue
        if self.show_settings {
            let mut open = true;
            let mut apply = false;
            egui::Window::new("Settings")
                .open(&mut open)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Detection hver N frame:");
                        ui.add(egui::DragValue::new(&mut self.tmp_detect_n).range(1..=20));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Stop recording efter (sek):");
                        ui.add(
                            egui::DragValue::new(&mut self.tmp_timeout)
                                .range(1.0..=10.0)
                                .speed(0.5),
                        );
                    });
                    if ui.button("Gem").clicked() {
                        apply = true;
                    }
                });
            if apply {
                self.detect_every_n = self.tmp_detect_n;
                self.person_timeout = self.tmp_timeout;
                log_event(&format!(
                    "Settings ændret: detect_every_n={}, timeout={}",
                    self.detect_every_n, self.person_timeout
                ));
                open = false;
            }
            self.show_settings = open;
        }

        ctx.request_repaint_after(Duration::from_millis(25));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.recording {
            self.stop_recording();
        }
        let _ = self.video.release();
        log_event("App lukket");
    }
}

// ---------------------------
// Start
// ---------------------------
fn main() -> eframe::Result<()> {
    for d in [video_dir(), snap_dir(), log_dir()] {
        let _ = fs::create_dir_all(d);
    }

    let mut viewport = egui::ViewportBuilder::default()
        .with_title("USB Webcam – Face Detection")
        .with_inner_size([680.0, 600.0]);

    // Ikon (valgfrit)
    if let Ok(bytes) = fs::read("usbwebcam.png") {
        if let Ok(icon) = eframe::icon_data::from_png_bytes(&bytes) {
            viewport = viewport.with_icon(icon);
        }
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "USB Webcam – Face Detection",
        options,
        Box::new(|_cc| match WebcamApp::new() {
            Ok(app) => Ok(Box::new(app)),
            Err(e) => Err(format!("{e}").into()),
        }),
    )
}