#!/bin/bash -xe

./spirv.sh src/vtn_render_vert.spv src/vtn_renderer.vert 
./spirv.sh src/vtn_render_frag.spv src/vtn_renderer.frag 

./spirv.sh src/two_triangles_vert.spv src/two_triangles.vert 
./spirv.sh src/two_triangles_frag.spv src/two_triangles.frag 

./spirv.sh src/mc.spv src/mc.comp 

./spirv.sh src/mc_render_vert.spv src/mc_render.vert 
./spirv.sh src/mc_render_frag.spv src/mc_render.frag 

./spirv.sh src/ray.spv src/ray.comp 

./spirv.sh src/sphere_tracer_comp.spv src/sphere_tracer.comp 

./spirv.sh src/generate_noise3d_comp.spv src/generate_noise3d.comp 

./spirv.sh src/eikonal_solver_comp.spv src/eikonal_solver.comp 

./spirv.sh src/radix_comp.spv src/radix.comp 

./spirv.sh src/local_sort_comp.spv src/local_sort.comp 

./spirv.sh src/line_vert.spv src/line.vert 
./spirv.sh src/line_frag.spv src/line.frag 

# ./spirv.sh src/noise2_comp.spv src/noise2.comp 
