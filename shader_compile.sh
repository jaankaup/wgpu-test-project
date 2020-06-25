#!/bin/bash -xe

./spirv.sh src/vtn_render_vert.spv src/vtn_renderer.vert 
./spirv.sh src/vtn_render_frag.spv src/vtn_renderer.frag 

./spirv.sh src/two_triangles_vert.spv src/two_triangles.vert 
./spirv.sh src/two_triangles_frag.spv src/two_triangles.frag 

./spirv.sh src/mc.spv src/mc.comp 

./spirv.sh src/mc_render_vert.spv src/mc_render.vert 
./spirv.sh src/mc_render_frag.spv src/mc_render.frag 

./spirv.sh src/ray.spv src/ray.comp 
