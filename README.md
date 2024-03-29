# An Eikonal equation solver

An eikonal equation solver using FIM (fast iterative marching) method. The project is implemented using [wgpu-rs](https://github.com/gfx-rs/wgpu/rs).
The FIM algorithm is based on paper 'A massively parallel Eikonal solver on unstructured meshes'. 

## Project features. 

This project implements the following features/algorithms. 

	The Marching Cubes algorithm (implemented but not optimized).
	A volumetric ray caster.
	A sphere tracer (also known as ray marcher). Not implemented yet.
	Fast Iterative Marching. Not implemented yet.

## usage

	cargo run

Shortkeys

* `a` Move left.
* `d` Move right.
* `w` Move forward.
* `s` Move backward.
* `c` Move down.
* `spacebar` Move up.
* `Key1` Two triangles.
* `Key2` Textured cube.
* `Key3` Marching cubes.
* `Key4` Random triangles.
* `Key5` Volumetric ray caster (noise functions).
* `Key6` Volumetric ray caster (3d texture).
* `Key7` Test for hilberd indices.
