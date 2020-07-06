# Eikonal equation solver

An eikonal equation solver using FIM (fast iterative marching) method. The project is implemented using [wgpu-rs](https://github.com/gfx-rs/wgpu/rs). The shaders are implemented using GLSL.

## Project features. 

This project implements the following features/algorithms. 

	The Marching Cubes algorithm (implemented but not optimized).
	A Sphere tracer (ray march) (uses now now 3d textures for and a "density" function aproach).
	Fast Iterative Marching (not finished yet).

## Building

	cargo run

## usage

	Moving: aswd + c + space
	Key1: Draw a single texture.
	Key2: Draw a textured cube.
	Key3: Marching cubes test.
	Key4: 1000 random triangles.
	Key5: A volume ray caster using some noise functions.
	Key6: A volume ray caster using 3d textures.
