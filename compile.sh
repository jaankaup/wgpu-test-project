#!/bin/bash

#OUT_DIR=$PWD

#RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --target wasm32-unknown-unknown
#wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/eikonal_application.wasm
#scp target/generated/* jaankaup@130.234.208.250:/home/jaankaup/public_html/wgpu-test 

RUST_BACKTRACE=1 cargo run
#cargo run 
