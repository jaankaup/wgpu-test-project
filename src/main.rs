use std::collections::HashMap;
use std::io::prelude::*;
use bytemuck::{Pod, Zeroable};
use rand::prelude::*;

use cgmath::{prelude::*, Vector3, Vector4};

enum Example {
    TwoTriangles,
    Cube,
    MC,
    RANDOM,
}

use winit::{
    event::{Event, WindowEvent,KeyboardInput,ElementState,VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::Window
};

use gradu::{Camera, RayCamera, CameraController, CameraUniform, Buffer, create_cube, Mc_uniform_data};

struct ShaderModuleInfo {
    name: &'static str,
    source_file: &'static str,
    stage: &'static str, // TODO: enum 
}

enum Resource {
    TextureView(&'static str),
    TextureSampler(&'static str),
    Buffer(&'static str),
}

struct BindGroupInfo {
    binding: u32,
    visibility: wgpu::ShaderStage,
    resource: Resource, 
    binding_type: wgpu::BindingType,
}

struct TextureInfo {
    name: &'static str,
    source: &'static str,
    width: Option<u32>,
    height: Option<u32>,
    depth: Option<u32>,
}

struct RenderPipelineInfo {
    vertex_shader: ShaderModuleInfo,
    fragment_shader: Option<ShaderModuleInfo>,
    bind_groups: Vec<Vec<BindGroupInfo>>,
    input_formats: Vec<(wgpu::VertexFormat, u64)>, 
}

struct ComputePipelineInfo {
    compute_shader: ShaderModuleInfo,
    bind_groups: Vec<Vec<BindGroupInfo>>,
}

//enum PipelineInfo {
//    Render(RenderPipelineInfo),
//    Compute(ComputePipelineInfo),
//}

enum PipelineResult {
    Render(Vec<wgpu::BindGroup>, wgpu::RenderPipeline),
    Compute(Vec<wgpu::BindGroup>, wgpu::ComputePipeline),
}

static CAMERA_UNIFORM_BUFFER_NAME : &'static str = "camera_uniform_buffer";
static MC_UNIFORM_BUFFER : &'static str = "mc_uniform_buffer";
static MC_COUNTER_BUFFER : &'static str = "mc_counter_buffer";
static MC_OUTPUT_BUFFER : &'static str = "mc_output_buffer";
static MC_DRAW_BUFFER : &'static str = "mc_draw_buffer";
static MC_INDEX_BUFFER : &'static str = "mc_index_buffer";
static RAY_MARCH_OUTPUT_BUFFER : &'static str = "ray_march_output";
//static RAY_MARCH_OUTPUT_STAGING_BUFFER : &'static str = "ray_march_output_staging";

static RANDOM_TRIANGLES_BUFFER : &'static str = "random_buffer";
static RANDOM_TRIANGLE_COUNT: u32 = 1000;


static DEPTH_TEXTURE_NAME : &'static str = "depth_texture";

// Define two triangles.
  
static TWO_TRIANGLES_TEXTURE: TextureInfo = TextureInfo {
    name: "diffuse_texture",  
    source: "grass1.png", // make sure this is loaded before use. 
    width: None,
    height: None,
    depth: None,
};

static TWO_TRIANGLES_INPUT_FORMATS: [(wgpu::VertexFormat, u64); 2]  = [
    (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
    (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
];

static TWO_TRIANGLES_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "two_triangles_vert", source_file: "two_triangles_vert.spv", stage: "vertex"},
    ShaderModuleInfo {name: "two_triangles_frag", source_file: "two_triangles_frag.spv", stage: "frag"},
];

static VTN_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "vtn_render_vert", source_file: "vtn_render_vert.spv", stage: "vertex"},
    ShaderModuleInfo {name: "vtn_render_frag", source_file: "vtn_render_frag.spv", stage: "frag"},
];

static MC_RENDER_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "mc_render_vert", source_file: "mc_render_vert.spv", stage: "vertex"},
    ShaderModuleInfo {name: "mc_render_frag", source_file: "mc_render_frag.spv", stage: "frag"},
];

static MARCHING_CUBES_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "mc",
           source_file: "mc.spv",
           stage: "compute",
};

static RAY_MARCH_SHADER: ShaderModuleInfo  = ShaderModuleInfo { 
           name: "ray_march",
           source_file: "ray.spv",
           stage: "compute",
};

fn create_two_triangles_info() -> RenderPipelineInfo { 
    let two_triangles_info: RenderPipelineInfo = RenderPipelineInfo {
        vertex_shader: ShaderModuleInfo {
            name: "two_triangles_vert",
            source_file: "two_triangles_vert.spv",
            stage: "vertex"
        }, 
        fragment_shader: Some(ShaderModuleInfo {
            name: "two_triangles_frag",
            source_file: "two_triangles_frag.spv",
            stage: "frag"
        }), 
        bind_groups: vec![
                vec![ 
                    BindGroupInfo {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureView(TWO_TRIANGLES_TEXTURE.name),
                        binding_type: wgpu::BindingType::SampledTexture {
                           multisampled: false,
                           component_type: wgpu::TextureComponentType::Float,
                           dimension: wgpu::TextureViewDimension::D2,
                        },
                    }, 
                    BindGroupInfo {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        resource: Resource::TextureSampler(TWO_TRIANGLES_TEXTURE.name),
                        binding_type: wgpu::BindingType::Sampler {
                           comparison: true,
                        },
                    },
                ],
        ],
        input_formats: vec![
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
            (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
        ],
    };

    two_triangles_info
}

fn vtn_renderer_info() -> RenderPipelineInfo { 
   let vtn_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
       vertex_shader: ShaderModuleInfo {
           name: VTN_SHADERS[0].name,
           source_file: VTN_SHADERS[0].source_file,
           stage: "vertex"
       }, 
       fragment_shader: Some(ShaderModuleInfo {
           name: VTN_SHADERS[1].name,
           source_file: VTN_SHADERS[1].source_file,
           stage: "frag"
       }), 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(CAMERA_UNIFORM_BUFFER_NAME),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::TextureView(TWO_TRIANGLES_TEXTURE.name),
                            binding_type: wgpu::BindingType::SampledTexture {
                               multisampled: false,
                               component_type: wgpu::TextureComponentType::Float,
                               dimension: wgpu::TextureViewDimension::D2,
                            },
                   }, 
                   BindGroupInfo {
                       binding: 1,
                       visibility: wgpu::ShaderStage::FRAGMENT,
                       resource: Resource::TextureSampler(TWO_TRIANGLES_TEXTURE.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: true,
                       },
                   },
               ],
           ],
           input_formats: vec![
               (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float2, 2 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float3, 3 * std::mem::size_of::<f32>() as u64)
           ],
    };

    vtn_renderer_info
}

fn mc_renderer_info() -> RenderPipelineInfo { 
   let mc_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
       vertex_shader: ShaderModuleInfo {
           name: MC_RENDER_SHADERS[0].name,
           source_file: MC_RENDER_SHADERS[0].source_file,
           stage: "vertex"
       }, 
       fragment_shader: Some(ShaderModuleInfo {
           name: MC_RENDER_SHADERS[1].name,
           source_file: MC_RENDER_SHADERS[1].source_file,
           stage: "frag"
       }), 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::Buffer(CAMERA_UNIFORM_BUFFER_NAME),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::FRAGMENT,
                            resource: Resource::TextureView(TWO_TRIANGLES_TEXTURE.name),
                            binding_type: wgpu::BindingType::SampledTexture {
                               multisampled: false,
                               component_type: wgpu::TextureComponentType::Float,
                               dimension: wgpu::TextureViewDimension::D2,
                            },
                   }, 
                   BindGroupInfo {
                       binding: 1,
                       visibility: wgpu::ShaderStage::FRAGMENT,
                       resource: Resource::TextureSampler(TWO_TRIANGLES_TEXTURE.name),
                       binding_type: wgpu::BindingType::Sampler {
                          comparison: true,
                       },
                   },
               ],
           ],
           input_formats: vec![
               (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
               (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
           ],
    };

    mc_renderer_info
}

fn marching_cubes_info() -> ComputePipelineInfo {
   let marching_cubes_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: MARCHING_CUBES_SHADER.name,
           source_file: MARCHING_CUBES_SHADER.source_file,
           stage: "compute"
       }, 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(MC_UNIFORM_BUFFER),
                            binding_type: wgpu::BindingType::UniformBuffer {
                               dynamic: false,
                               min_binding_size: None,
                            },
                   }, 
                   BindGroupInfo {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(MC_COUNTER_BUFFER),
                            binding_type: wgpu::BindingType::StorageBuffer {
                               dynamic: false,
                               readonly: false,
                               min_binding_size: None,
                            },
                   }, 
                   BindGroupInfo {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(MC_OUTPUT_BUFFER),
                            binding_type: wgpu::BindingType::StorageBuffer {
                               dynamic: false,
                               readonly: false,
                               min_binding_size: None,
                            },
                   }, 
               ],
           ],
    };

    marching_cubes_info
}

fn ray_march_info() -> ComputePipelineInfo {
   let ray_march_info: ComputePipelineInfo = ComputePipelineInfo {
       compute_shader: ShaderModuleInfo {
           name: RAY_MARCH_SHADER.name,
           source_file: RAY_MARCH_SHADER.source_file,
           stage: "compute"
       }, 
       bind_groups:
           vec![ 
               vec![
                   BindGroupInfo {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            resource: Resource::Buffer(RAY_MARCH_OUTPUT_BUFFER),
                            binding_type: wgpu::BindingType::StorageBuffer {
                               dynamic: false,
                               readonly: false,
                               min_binding_size: wgpu::BufferSize::new(256*256),
                            },
                   }, 
               ],
           ],
    };

    ray_march_info
}


fn create_render_pipeline_and_bind_groups(device: &wgpu::Device,
                                   sc_desc: &wgpu::SwapChainDescriptor,
                                   shaders: &HashMap<String, wgpu::ShaderModule>,
                                   textures: &HashMap<String, gradu::Texture>,
                                   buffers: &HashMap<String, gradu::Buffer>,
                                   rpi: &RenderPipelineInfo)
    -> (Vec<wgpu::BindGroup>, wgpu::RenderPipeline) {
    
    print!("    * Creating bind groups ... ");
    
    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();
    
    // Loop over all bind_groups.
    for b_group in rpi.bind_groups.iter() {
    
        let layout_entries: Vec<wgpu::BindGroupLayoutEntry>
            = b_group.into_iter().map(|x| wgpu::BindGroupLayoutEntry::new(
                x.binding,
                x.visibility,
                x.binding_type.clone(),
              )).collect();
    
        let texture_bind_group_layout =
           device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
               bindings: &layout_entries,
               label: None,
        });
    
        let bindings: Vec<wgpu::Binding> 
            = b_group.into_iter().map(|x| wgpu::Binding {
                binding: x.binding,
                resource: match x.resource {
                        Resource::TextureView(tw) =>  
                            wgpu::BindingResource::TextureView(&textures.get(tw).unwrap().view),
                        Resource::TextureSampler(ts) => 
                            wgpu::BindingResource::Sampler(&textures.get(ts).unwrap().sampler),
                        Resource::Buffer(b) => 
                            wgpu::BindingResource::Buffer(buffers.get(b).unwrap().buffer.slice(..)),
                }
            }).collect();
    
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            bindings: &bindings,
            label: None,
        });
    
        bind_group_layouts.push(texture_bind_group_layout);
        bind_groups.push(bind_group);
    }
    
    println!(" OK'");
    
    print!("    * Creating pipeline ... ");
    
      let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
          bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(), 
      });
    
      // Crete vertex attributes.
      let mut stride: u64 = 0;
      let mut vertex_attributes: Vec<wgpu::VertexAttributeDescriptor> = Vec::new();
      for i in 0..rpi.input_formats.len() {
          vertex_attributes.push(
              wgpu::VertexAttributeDescriptor {
                  format: rpi.input_formats[0].0,
                  offset: stride,
                  shader_location: i as u32,
              }
          );
          stride = stride + rpi.input_formats[i].1;  
      }
    
      let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &render_pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &shaders.get(rpi.vertex_shader.name).unwrap(),
            entry_point: "main",
        }, // TODO: do case for fragmen_shader == None
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &shaders.get(rpi.fragment_shader.as_ref().unwrap().name).unwrap(),
            entry_point: "main",
        }),
        rasterization_state: Some(wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::Back,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[
            wgpu::ColorStateDescriptor {
                format: sc_desc.format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            },
        ],
        //depth_stencil_state: None,
        depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            //stencil_read_only: false,
        }),
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: stride,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &vertex_attributes,
            }],
        },
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false,
      });
    

    println!(" OK'");
    (bind_groups, render_pipeline)
}

fn create_compute_pipeline_and_bind_groups(device: &wgpu::Device,
                                           sc_desc: &wgpu::SwapChainDescriptor,
                                           shaders: &HashMap<String, wgpu::ShaderModule>,
                                           textures: &HashMap<String, gradu::Texture>,
                                           buffers: &HashMap<String, gradu::Buffer>,
                                           rpi: &ComputePipelineInfo)
    -> (Vec<wgpu::BindGroup>, wgpu::ComputePipeline) {

    print!("    * Creating compute bind groups ... ");

    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();

    // Loop over all bind_groups.
    for b_group in rpi.bind_groups.iter() {

        let layout_entries: Vec<wgpu::BindGroupLayoutEntry>
            = b_group.into_iter().map(|x| wgpu::BindGroupLayoutEntry::new(
                x.binding,
                x.visibility,
                x.binding_type.clone(),
              )).collect();

        let texture_bind_group_layout =
           device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
               bindings: &layout_entries,
               label: None,
        });

        let bindings: Vec<wgpu::Binding> 
            = b_group.into_iter().map(|x| wgpu::Binding {
                binding: x.binding,
                resource: match x.resource {
                        Resource::TextureView(tw) =>  
                            wgpu::BindingResource::TextureView(&textures.get(tw).unwrap().view),
                        Resource::TextureSampler(ts) => 
                            wgpu::BindingResource::Sampler(&textures.get(ts).unwrap().sampler),
                        Resource::Buffer(b) => 
                            wgpu::BindingResource::Buffer(buffers.get(b).unwrap().buffer.slice(..)),
                }
            }).collect();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            bindings: &bindings,
            label: None,
        });

        bind_group_layouts.push(texture_bind_group_layout);
        bind_groups.push(bind_group);
    }

    println!(" OK'");

    print!("    * Creating compute pipeline ... ");

      let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
          bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(), 
      });

      let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
          layout: &compute_pipeline_layout,
          compute_stage: wgpu::ProgrammableStageDescriptor {
              module: &shaders.get(rpi.compute_shader.name).unwrap(),
              entry_point: "main",
          },
      });
    

    println!(" OK'");
    (bind_groups, compute_pipeline)
}


/// The resources for graphics.
pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,
    buffers: HashMap<String,gradu::Buffer>,
    bind_groups: HashMap<String,wgpu::BindGroup>,
    render_pipelines: HashMap<String,wgpu::RenderPipeline>,
    compute_pipelines: HashMap<String,wgpu::ComputePipeline>,
    textures: HashMap<String,gradu::Texture>,
    two_triangles_bind_groups: Vec<wgpu::BindGroup>,
    two_triangles_render_pipeline: wgpu::RenderPipeline,
    vtn_bind_groups: Vec<wgpu::BindGroup>,
    vtn_render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
//    ray_camera: RayCamera,
    camera_controller: CameraController,
    camera_uniform: CameraUniform,
    example: Example,
    mc_render_bind_groups: Vec<wgpu::BindGroup>,
    mc_render_pipeline: wgpu::RenderPipeline,
    mc_compute_bind_groups: Vec<wgpu::BindGroup>,
    mc_compute_pipeline: wgpu::ComputePipeline,
    mc_vertex_count: u32,
    ray_march_bind_groups: Vec<wgpu::BindGroup>,
    ray_march_compute_pipeline: wgpu::ComputePipeline,
}


use gradu::Texture;  

impl State {

    /// Initializes the project resources and returns the intance for State object. 
    pub async fn new(window: &Window) -> Self {

        let example = Example::TwoTriangles;

        // Create the surface, adapter, device and the queue.
        let (surface, device, queue, size) = create_sdqs(window).await;

        // Create the swap_chain_descriptor and swap_chain.
        let (sc_desc, swap_chain) = create_swap_chain(size, &surface, &device);
           
        // Storage for textures. It is important to load textures before creating bind groups.
        let mut textures = HashMap::new();
        create_textures(&device, &queue, &sc_desc, &mut textures); 

        // Create shaders.
        let shaders = create_shaders(&device);

        // Storage for buffers.
        let mut buffers = HashMap::new();
        create_vertex_buffers(&device, &mut buffers);
        
        // Storage for bind groups.
        let mut bind_groups: HashMap<String,wgpu::BindGroup> = HashMap::new();

        // Storage for render pipelines.
        let mut render_pipelines = HashMap::new();

        // Storage for compute pipelines.
        let mut compute_pipelines = HashMap::new();


        // The camera.
        let mut camera = Camera {
            pos: (1.0, 1.0, 1.0).into(),
            view: Vector3::new(0.0, 0.0, -1.0).normalize(),
            up: cgmath::Vector3::unit_y(),
            aspect: sc_desc.width as f32 / sc_desc.height as f32,
            fov: (45.0,45.0).into(),
            znear: 0.1,
            zfar: 1000.0,
        };

        // The camera controller.
        let camera_controller = CameraController::new(0.2,0.5);

        camera.view = Vector3::new(
            camera_controller.pitch.to_radians().cos() * camera_controller.yaw.to_radians().cos(),
            camera_controller.pitch.to_radians().sin(),
            camera_controller.pitch.to_radians().cos() * camera_controller.yaw.to_radians().sin()
        ).normalize();

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = Buffer::create_buffer_from_data::<CameraUniform>(
            &device,
            &[camera_uniform],
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            None);

        buffers.insert(CAMERA_UNIFORM_BUFFER_NAME.to_string(), camera_buffer);

        let two_triangles_info = create_two_triangles_info(); 

        println!("Creating two_triangles pipeline and bind groups.\n");

        //let (two_triangles_bind_groups, two_triangles_render_pipeline) = create_pipeline_and_bind_groups(
        let (two_triangles_bind_groups, two_triangles_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &two_triangles_info);

        let vtn_info = vtn_renderer_info();

        println!("\nCreating vtn_render pipeline and bind groups.\n");

        let (vtn_bind_groups, vtn_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &vtn_info);

        println!("");

        let mc_renderer_info = mc_renderer_info(); 

        println!("\nCreating mc_render pipeline and bind groups.\n");

        let (mc_render_bind_groups, mc_render_pipeline) = create_render_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &mc_renderer_info);

        println!("");

        println!("\nCreating mc_render pipeline and bind groups.\n");
        let mc_compute_info = marching_cubes_info();
        let (mc_compute_bind_groups, mc_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &mc_compute_info);

        println!("");

        println!("\nCreating ray_march pipeline and bind groups.\n");
        let ray_march_info = ray_march_info();
        let (ray_march_bind_groups, ray_march_compute_pipeline) = create_compute_pipeline_and_bind_groups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &ray_march_info);

        println!("");

        //println!("\nLaunching marching cubes.\n");

        //let mut mc_encoder = 
        //    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        //{
        //    let mut mc_pass = mc_encoder.begin_compute_pass();
        //    mc_pass.set_pipeline(&mc_compute_pipeline);
        //    for (e, bgs) in mc_compute_bind_groups.iter().enumerate() {
        //        mc_pass.set_bind_group(e as u32, &bgs, &[]);
        //    }
        //    mc_pass.dispatch(8,8,8);
        //}

        //mc_encoder.copy_buffer_to_buffer(&buffers.get(MC_OUTPUT_BUFFER).unwrap().buffer, 0, &buffers.get(MC_DRAW_BUFFER).unwrap().buffer, 0, 4 * 64*64*64);

        //// Launch marching cubes.
        //queue.submit(Some(mc_encoder.finish()));
        //let k = &buffers.get(MC_COUNTER_BUFFER).unwrap().to_vec::<u32>(&device, &queue, true).await;
        //let mc_vertex_count = k[0];
        let mc_vertex_count = 678;

        //print!("    * Creating marching cubes index buffer as '{}'", MC_INDEX_BUFFER);

        let mut index_buffer: Vec<u16> = Vec::new();

        //for i in 0..mc_vertex_count {
        for i in 0..666 {
            index_buffer.push(i as u16);
        }

        let marching_cubes_index_buffer = Buffer::create_buffer_from_data::<u16>(
            &device,
            &index_buffer[..],
            wgpu::BufferUsage::INDEX,
            None
        );

        //buffers.insert(MC_INDEX_BUFFER.to_string(), marching_cubes_index_buffer);

        //println!(" ... OK'");

        println!("\nLaunching ray march.\n");

        //let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //    label: None,
        //    256*256*4,
        //    usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        //    mapped_at_creation: false,
        //});

        let mut ray_encoder = 
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut ray_pass = ray_encoder.begin_compute_pass();
            ray_pass.set_pipeline(&ray_march_compute_pipeline);
            for (e, bgs) in ray_march_bind_groups.iter().enumerate() {
                ray_pass.set_bind_group(e as u32, &bgs, &[]);
            }
            ray_pass.dispatch(1,1,1);
        }

        //ray_encoder.copy_buffer_to_buffer(
        //    &buffers.get(RAY_MARCH_OUTPUT_BUFFER).unwrap().buffer,
        //    0,
        //    &buffers.get(RAY_MARCH_OUTPUT_STAGING_BUFFER).unwrap().buffer,
        //    0,
        //    256*256*4);
        //ray_encoder.copy_buffer_to_buffer(&buffers.get(MC_OUTPUT_BUFFER).unwrap().buffer, 0, &buffers.get(MC_DRAW_BUFFER).unwrap().buffer, 0, 4 * 64*64*64);

        queue.submit(Some(ray_encoder.finish()));

        //let j = &buffers.get(RAY_MARCH_OUTPUT_BUFFER).unwrap().to_vec::<u32>(&device, &queue, true).await;
        //println!("ray march output result: ");
        //for i in 0..j.len() {
        //    if j[i] != 999999 {
        //        println!("{} :: {}", i, j[i]);
        //    }
        //}

        let k = &buffers.get(RAY_MARCH_OUTPUT_BUFFER).unwrap().to_vec::<u32>(&device, &queue, true).await;
        for i in 0..256*256 {
            if k[i] == 999999 { continue; } 
            //if i % 256 == 0 { print!("("); }
            println!("{} :: {}", i, k[i]);
            //println!("(i={}, x={},y={})", i, (k[i] & 0xffff0000) >> 16 , k[i] & 0xffff );
            //if i % 256 == 0 { println!(")"); }
        }

        println!("The end of ray march output result.");

        println!(" ... OK'");

        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            buffers,
            bind_groups,
            render_pipelines,
            compute_pipelines,
            two_triangles_bind_groups,
            two_triangles_render_pipeline,
            vtn_bind_groups,
            vtn_render_pipeline,
            camera,
//            ray_camera, 
            camera_controller,
            camera_uniform,
            textures,
            example,
            mc_render_bind_groups,
            mc_render_pipeline,
            mc_compute_bind_groups,
            mc_compute_pipeline,
            mc_vertex_count,
            ray_march_bind_groups,
            ray_march_compute_pipeline,
        }
    } // new(...

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
//
//        let depth_texture = Texture::create_depth_texture(&self.device, &self.sc_desc, Some("depth-texture"));
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        let result = self.camera_controller.process_events(event);
        if result == true {
            self.camera_controller.update_camera(&mut self.camera);
            println!("({}, {}, {})",self.camera.pos.x, self.camera.pos.y, self.camera.pos.z);
        }
        result
    }

    pub fn update(&mut self) {

        //let mut camera_uniform = CameraUniform::new();
        self.camera_uniform.update_view_proj(&self.camera);

        // TODO: Create a method for this in Buffer.
        self.queue.write_buffer(
            &self.buffers.get(CAMERA_UNIFORM_BUFFER_NAME).unwrap().buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform])
        );

    }

    pub fn render(&mut self) {
        let frame = self.swap_chain.get_next_frame().expect("Failed to acquire next swap chain texture").output;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { 
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }
                ],
                //depth_stencil_attachment: None,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.textures.get("DEPTH_TEXTURE_NAME").unwrap().view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), 
                        store: true,
                        }),
                    stencil_ops: None,
                    }),
            });

            match self.example {
                Example::TwoTriangles => {
                    render_pass.set_pipeline(&self.two_triangles_render_pipeline);
                    for (e, bgs) in self.two_triangles_bind_groups.iter().enumerate() {
                        render_pass.set_bind_group(e as u32, &bgs, &[]);
                    }
                    render_pass.set_vertex_buffer(0, self.buffers.get("two_triangles_buffer").unwrap().buffer.slice(..));
                    render_pass.draw(0..6, 0..2);
                }

                Example::Cube => {
                    render_pass.set_pipeline(&self.vtn_render_pipeline);
                    for (e, bgs) in self.vtn_bind_groups.iter().enumerate() {
                        render_pass.set_bind_group(e as u32, &bgs, &[]);
                    }
                    render_pass.set_vertex_buffer(0, self.buffers.get("cube_buffer").unwrap().buffer.slice(..));
                    render_pass.draw(0..36, 0..12);
                }
                Example::MC => {
                    render_pass.set_pipeline(&self.mc_render_pipeline);
                    for (e, bgs) in self.mc_render_bind_groups.iter().enumerate() {
                        render_pass.set_bind_group(e as u32, &bgs, &[]);
                    }
                    render_pass.set_vertex_buffer(0, self.buffers.get(MC_DRAW_BUFFER).unwrap().buffer.slice(..));
                    render_pass.set_index_buffer(self.buffers.get(MC_INDEX_BUFFER).unwrap().buffer.slice(..));
                    //render_pass.draw(0..self.mc_vertex_count, 0..self.mc_vertex_count/3);
                    render_pass.draw_indexed(0..self.mc_vertex_count, 0, 0..self.mc_vertex_count/3);
                }
                Example::RANDOM => {
                    render_pass.set_pipeline(&self.mc_render_pipeline);
                    for (e, bgs) in self.mc_render_bind_groups.iter().enumerate() {
                        render_pass.set_bind_group(e as u32, &bgs, &[]);
                    }
                    render_pass.set_vertex_buffer(0, self.buffers.get(RANDOM_TRIANGLES_BUFFER).unwrap().buffer.slice(..));
                    render_pass.draw(0..RANDOM_TRIANGLE_COUNT*3, 0..RANDOM_TRIANGLE_COUNT);
                }
            }
        }

        //encoder.finish();
        self.queue.submit(Some(encoder.finish()));


    }
}

/// Load shaders.
fn create_shaders(device: &wgpu::Device) -> HashMap<String, wgpu::ShaderModule> {

    println!("\nCreating shaders.\n");
    let mut shaders = HashMap::new();

    print!("    * Creating 'two_triangles_vert' shader module from file 'two_triangles_vert.spv'");
    shaders.insert("two_triangles_vert".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles_vert.spv")));
    println!(" ... OK'");

    print!("    * Creating 'two_triangles_frag' shader module from file 'two_triangles_frag.spv'");
    shaders.insert("two_triangles_frag".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles_frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'", VTN_SHADERS[0].name, VTN_SHADERS[0].source_file);
    shaders.insert(VTN_SHADERS[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("vtn_render_vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",VTN_SHADERS[1].name, VTN_SHADERS[1].source_file);
    shaders.insert(VTN_SHADERS[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("vtn_render_frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'", MC_RENDER_SHADERS[0].name, MC_RENDER_SHADERS[0].source_file);
    shaders.insert(MC_RENDER_SHADERS[0].name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc_render_vert.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",MC_RENDER_SHADERS[1].name, MC_RENDER_SHADERS[1].source_file);
    shaders.insert(MC_RENDER_SHADERS[1].name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc_render_frag.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",MARCHING_CUBES_SHADER.name, MARCHING_CUBES_SHADER.source_file);
    shaders.insert(MARCHING_CUBES_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("mc.spv")));
    println!(" ... OK'");

    print!("    * Creating '{}' shader module from file '{}'",RAY_MARCH_SHADER.name, RAY_MARCH_SHADER.source_file);
    shaders.insert(RAY_MARCH_SHADER.name.to_string(), device.create_shader_module(wgpu::include_spirv!("ray.spv")));
    println!(" ... OK'");

    println!("\nShader created!\n");
    shaders
}

fn create_vertex_buffers(device: &wgpu::Device, buffers: &mut HashMap::<String, gradu::Buffer>)  {

    println!("\nCreating buffers.\n");

    print!("    * Creating cube buffer as 'cube_buffer'");

    // The Cube.
    let vertex_data = create_cube();
    let cube = Buffer::create_buffer_from_data::<f32>(
        device,
        &vertex_data,
        wgpu::BufferUsage::VERTEX,
        None);

    buffers.insert("cube_buffer".to_string(), cube);

    println!(" ... OK'");

    print!("    * Creating two_triangles buffer as 'two_triangles_buffer'");

    // 2-triangles.

    let two_triangles = 
        Buffer::create_buffer_from_data::<f32>(
        device,
        // gl_Position     |    point_pos
        &[-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
           1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
           1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
          -1.0,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
          -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ],
        wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_SRC,
        None
        );

    buffers.insert("two_triangles_buffer".to_string(), two_triangles);

    println!(" ... OK'");

    print!("    * Creating marching cubes output buffer as '{}'", MC_OUTPUT_BUFFER);

    let marching_cubes_output = Buffer::create_buffer_from_data::<f32>(
        device,
        &vec![0 as f32 ; 64*64*64],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::VERTEX,
        None
    );

    buffers.insert(MC_OUTPUT_BUFFER.to_string(), marching_cubes_output);

    println!(" ... OK'");

    print!("    * Creating marching cubes counter buffer as '{}'", MC_COUNTER_BUFFER);

    let marching_cubes_counter = Buffer::create_buffer_from_data::<u32>(
        device,
        &[0 as u32],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(MC_COUNTER_BUFFER.to_string(), marching_cubes_counter);

    println!(" ... OK'");

    print!("    * Creating marching cubes uniform buffer as '{}'", MC_UNIFORM_BUFFER);

    let mc_u_data = Mc_uniform_data {
        isovalue: 0.0,
        cube_length: 0.1,
        base_position: cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0),
    };

    let marching_cubes_uniform_buffer = Buffer::create_buffer_from_data::<Mc_uniform_data>(
        device,
        &[mc_u_data],
        wgpu::BufferUsage::COPY_DST |wgpu::BufferUsage::UNIFORM,
        None
    );

    buffers.insert(MC_UNIFORM_BUFFER.to_string(), marching_cubes_uniform_buffer);

    println!(" ... OK'");

    print!("    * Creating marching cubes draw buffer as '{}'", MC_DRAW_BUFFER);

    let marching_cubes_draw_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &vec![0 as f32 ; 64*64*64],
        wgpu::BufferUsage::COPY_SRC | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::VERTEX,
        None
    );

    buffers.insert(MC_DRAW_BUFFER.to_string(), marching_cubes_draw_buffer);

    let mut rng = thread_rng();
    let mut random_triangles = Vec::new();
    for i in 0..RANDOM_TRIANGLE_COUNT {
        let a1 = rng.gen(); 
        let a2 = rng.gen(); 
        let a3 = rng.gen(); 
        let b1 = rng.gen(); 
        let b2 = rng.gen(); 
        let b3 = rng.gen(); 
        let c1 = rng.gen(); 
        let c2 = rng.gen(); 
        let c3 = rng.gen(); 
        let vert_a = cgmath::Vector3::new(a1,a2,a3);
        let vert_b = cgmath::Vector3::new(b1,b2,b3);
        let vert_c = cgmath::Vector3::new(c1,c2,c3);
    
        let u = vert_b - vert_c;
        let v = vert_a - vert_c;

        let normal = u.cross(v).normalize();

        random_triangles.push(vert_a.x);
        random_triangles.push(vert_a.y);
        random_triangles.push(vert_a.z);
        random_triangles.push(1.0);

        random_triangles.push(normal.x);
        random_triangles.push(normal.y);
        random_triangles.push(normal.z);
        random_triangles.push(0.0);

        random_triangles.push(vert_b.x);
        random_triangles.push(vert_b.y);
        random_triangles.push(vert_b.z);
        random_triangles.push(1.0);

        random_triangles.push(normal.x);
        random_triangles.push(normal.y);
        random_triangles.push(normal.z);
        random_triangles.push(0.0);

        random_triangles.push(vert_c.x);
        random_triangles.push(vert_c.y);
        random_triangles.push(vert_c.z);
        random_triangles.push(1.0);

        random_triangles.push(normal.x);
        random_triangles.push(normal.y);
        random_triangles.push(normal.z);
        random_triangles.push(0.0);
    }

    let random_triangles_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &random_triangles,
        wgpu::BufferUsage::VERTEX,
        None
    );

    buffers.insert(RANDOM_TRIANGLES_BUFFER.to_string(), random_triangles_buffer);

    println!(" ... OK'");

    print!("    * Creating ray march output buffer as '{}'", RAY_MARCH_OUTPUT_BUFFER);

    let ray_march_output = Buffer::create_buffer_from_data::<u32>(
        device,
        &vec![999999 as u32 ; 256*256],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None
    );

    buffers.insert(RAY_MARCH_OUTPUT_BUFFER.to_string(), ray_march_output);
    println!(" ... OK'");

    println!("");
}

fn create_textures(device: &wgpu::Device, queue: &wgpu::Queue, sc_desc: &wgpu::SwapChainDescriptor, textures: &mut HashMap<String, gradu::Texture>) {

    println!("\nCreating textures.\n");
    // Two triangles texture.
    print!("    * Creating texture from file 'grass1.png'");
    let diffuse_texture = Texture::create_from_bytes(&queue, &device, &include_bytes!("grass1.png")[..], None);
    textures.insert(TWO_TRIANGLES_TEXTURE.name.to_string(), diffuse_texture);
    println!(" ... OK'");

    let depth_texture = Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture"));
    //let tritable_texture = Texture::create_tritable(&queue, &device);
    //let ray_output_texture = Texture::create_texture2D(&queue, &device, 256,256);
    textures.insert("DEPTH_TEXTURE_NAME".to_string(), depth_texture);
    //textures.insert("tritable_texture".to_string(), tritable_texture);
    //textures.insert("ray_output_texture".to_string(), ray_output_texture);
}

async fn create_sdqs(window: &winit::window::Window) -> (wgpu::Surface, wgpu::Device, wgpu::Queue, winit::dpi::PhysicalSize<u32>) {

        // Get the size of the window.
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        // Create the surface.
        let surface = unsafe { instance.create_surface(window) };

        // Create the adapter.
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::UnsafeExtensions::disallow(),
        )
        .await
        .unwrap();

        // TODO: check what this mean.
        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter.request_device(
             &wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions::empty(), 
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .unwrap();

        (surface, device,queue,size)
}

fn create_swap_chain(size: winit::dpi::PhysicalSize<u32>, surface: &wgpu::Surface, device: &wgpu::Device) -> (wgpu::SwapChainDescriptor, wgpu::SwapChain) {
                                            
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Mailbox,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);
        (sc_desc, swap_chain)
}

async fn run(window: Window, event_loop: EventLoop<()>) {

    let mut state = State::new(&window).await;

    event_loop.run(move |event, _, control_flow| {
        let _ = (&state,&window);

        #[cfg(target_arch = "wasm32")] {
            ControlFlow::Poll;
        }
        
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input,
                        ..
                    } => {
                        match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key1),
                                ..
                            } => state.example = Example::TwoTriangles,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key2),
                                ..
                            } => state.example = Example::Cube,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key3),
                                ..
                            } => state.example = Example::MC,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Key4),
                                ..
                            } => state.example = Example::RANDOM,
                            _ => {}
                        }
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &mut so w have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                //state.update();
                //state.render();
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                ////#[cfg(target_arch = "wasm32")] {
                ////    window.request_redraw();
                ////}
                ////#[cfg(not(target_arch = "wasm32"))] {
                state.update();
                state.render();
                //*control_flow = ControlFlow::Exit;
                ////}
            }
            _ => {}
        }
    });
}


fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        futures::executor::block_on(run(window, event_loop));
        //let th = std::thread::spawn(move || {
        //    let mut pool = futures::executor::LocalPool::new();
        //    let spawner = pool.spawner();
        //    spawner.spawn_local(async move {
        //        let event_loop = EventLoop::new();
        //        let window = winit::window::Window::new(&event_loop).unwrap();
        //        run(window, event_loop).await;
        //    }).unwrap();
        //    pool.run();
        //});
        //th.join();
    }
//    #[cfg(target_arch = "wasm32")]
//    {
//        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
//        console_log::init().expect("could not initialize logger");
//        use winit::platform::web::WindowExtWebSys;
//        // On wasm, append the canvas to the document body
//        web_sys::window()
//            .and_then(|win| win.document())
//            .and_then(|doc| doc.body())
//            .and_then(|body| {
//                body.append_child(&web_sys::Element::from(window.canvas()))
//                    .ok()
//            })
//            .expect("couldn't append canvas to document body");
//        wasm_bindgen_futures::spawn_local(run(window, event_loop));
//        //wasm_bindgen_futures::spawn_local(run_async::<E>(event_loop, window));
//    }
}
