#!/bin/bash -xe

./spirv.sh src/vtn_render_vert.spv src/vtn_renderer.vert 
./spirv.sh src/vtn_render_frag.spv src/vtn_renderer.frag 
./spirv.sh src/two_triangles_frag.spv src/two_triangles.frag 
./spirv.sh src/two_triangles_frag.spv src/two_triangles.frag 
