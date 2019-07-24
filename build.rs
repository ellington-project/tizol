fn main() {
    prost_build::compile_protos(&["src/spectrogram.proto"], &["src/"]).unwrap();
}
