use std::collections::HashMap;
use std::io::prelude::*;
use bytemuck::{Pod, Zeroable};

//use cgmath::{prelude::*, Vector3};
use cgmath::{Vector3};

use winit::{
    event::{Event, WindowEvent,KeyboardInput,ElementState,VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::Window
};

use gradu::{Camera, RayCamera, CameraController, CameraUniform, Buffer, create_cube};

// Map the name and the shader in create_shaders function.
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

// Define two triangles.
  
static TWO_TRIANGLES_TEXTURE: TextureInfo = TextureInfo {
    name: "diffuse_texture",  
    source: "grass1.png", // make sure this is loaded before use. 
    width: None,
    height: None,
    depth: None,
};

// Defines a render pipeline.
struct RenderPipelineInfo {
    vertex_shader: ShaderModuleInfo,
    fragment_shader: Option<ShaderModuleInfo>,
    bind_groups: Vec<Vec<BindGroupInfo>>,
    input_formats: Vec<(wgpu::VertexFormat, u64)>, 
}

// Defines a compute pipeline.
struct ComputePipelineInfo {
    compute_shader: ShaderModuleInfo,
    bind_groups: Vec<BindGroupInfo>,
}

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

// fn vtn_renderer_info() -> RenderPipelineInfo { 
//     let vtn_renderer_info: RenderPipelineInfo = RenderPipelineInfo {
//         vertex_shader: ShaderModuleInfo {
//             name: "vtn_renderer_vert", //TODO create
//             source_file: "vtn_renderer_vert.spv", //TODO create
//             stage: "vertex"
//         }, 
//         fragment_shader: Some(ShaderModuleInfo {
//             name: "vtn_renderer_frag", //TODO create
//             source_file: "vtn_renderer_frag.spv", //TODO create
//             stage: "frag"
//         }), 
//         bind_groups: vec![ 
//             BindGroupInfo {
//                 binding: 0,
//                 visibility: wgpu::ShaderStage::FRAGMENT,
//                 resource: Resource::TextureView(TWO_TRIANGLES_TEXTURE.name),
//                 binding_type: wgpu::BindingType::SampledTexture {
//                    multisampled: false,
//                    component_type: wgpu::TextureComponentType::Float,
//                    dimension: wgpu::TextureViewDimension::D2,
//                 },
//             }, 
//             BindGroupInfo {
//                 binding: 1,
//                 visibility: wgpu::ShaderStage::FRAGMENT,
//                 resource: Resource::TextureSampler(TWO_TRIANGLES_TEXTURE.name),
//                 binding_type: wgpu::BindingType::Sampler {
//                    comparison: true,
//                 },
//             },
//         ],
//         input_formats: vec![
//             (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
//             (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
//         ],
//     };
// 
//     two_triangles_info
// }

// (Attribute type, size).
static TWO_TRIANGLES_INPUT_FORMATS: [(wgpu::VertexFormat, u64); 2]  = [
    (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64),
    (wgpu::VertexFormat::Float4, 4 * std::mem::size_of::<f32>() as u64)
];

static TWO_TRIANGLES_SHADERS: [ShaderModuleInfo; 2]  = [
    ShaderModuleInfo {name: "two_triangles_vert", source_file: "two_triangles_vert.spv", stage: "vertex"},
    ShaderModuleInfo {name: "two_triangles_frag", source_file: "two_triangles_frag.spv", stage: "frag"},
];

static TWO_TRIANGLES_BIND_GROUPS: [BindGroupInfo; 2] = [
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
];

/// Create the two triangles pipeline and bind groups.
fn create_two_triangles_pipeline_bindgroups(device: &wgpu::Device,
                                            sc_desc: &wgpu::SwapChainDescriptor,
                                            shaders: &HashMap<String, wgpu::ShaderModule>,
                                            textures: &HashMap<String, gradu::Texture>,
                                            buffers: &HashMap<String, gradu::Buffer>,
                                            rpi: &RenderPipelineInfo)
    -> (Vec<wgpu::BindGroup>, wgpu::RenderPipeline) {

    print!("\nCreating two triangles bind groups. ");

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

        bind_group_layouts.push(texture_bind_group_layout);;
        bind_groups.push(bind_group);

        //all_bind_groups_layouts.push((texture_bind_group_layout, bind_group));
    }

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

    print!("\nCreating two triangles pipeline ");
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
        depth_stencil_state: None,
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

    println!(" ... OK'");
    (bind_groups, render_pipeline)
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
//    camera: Camera,
//    ray_camera: RayCamera,
//    camera_controller: CameraController,
//    camera_uniform: CameraUniform,
}


use gradu::Texture;  

impl State {

    /// Initializes the project resources and returns the intance for State object. 
    pub async fn new(window: &Window) -> Self {

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

        let two_triangles_info = create_two_triangles_info(); 


        let (two_triangles_bind_groups, two_triangles_render_pipeline) = create_two_triangles_pipeline_bindgroups(
                        &device,
                        &sc_desc,
                        &shaders,
                        &textures,
                        &buffers,
                        &two_triangles_info);


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
//            camera,
//            ray_camera, 
//            camera_controller,
//            camera_uniform,
            textures,
        }
    } // new(...

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
//        self.size = new_size;
//        self.sc_desc.width = new_size.width;
//        self.sc_desc.height = new_size.height;
//        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
//
//        let depth_texture = Texture::create_depth_texture(&self.device, &self.sc_desc, Some("depth-texture"));
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
//        let result = self.camera_controller.process_events(event);
//        if result == true {
//            self.camera_controller.update_camera(&mut self.camera);
//            println!("({}, {}, {})",self.camera.pos.x, self.camera.pos.y, self.camera.pos.z);
//        }
//        result
          false
    }

    pub fn update(&mut self) {
//        let ray_camera_uniform_buffer = Buffer::create_buffer_from_data::<Camera>(
//            &self.device,
//            &[self.camera],
//            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
//            None);
//
//        self.buffers.insert("ray_camera_uniform_buffer".to_string(), ray_camera_uniform_buffer);

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
                        //load_op: wgpu::LoadOp::Clear,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { 
                                r: 0.0,
                                g: 0.0,
                                b: 0.1,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    }
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.two_triangles_render_pipeline);
            for (e, bgs) in self.two_triangles_bind_groups.iter().enumerate() {
                render_pass.set_bind_group(e as u32, &bgs, &[]);
            }
            render_pass.set_vertex_buffer(0, self.buffers.get("two_triangles_buffer").unwrap().buffer.slice(..));
            render_pass.draw(0..6, 0..2);
        }

        //encoder.finish();
        self.queue.submit(Some(encoder.finish()));


    }
}

/// Load shaders.
fn create_shaders(device: &wgpu::Device) -> HashMap<String, wgpu::ShaderModule> {

    println!("\nCreating shaders.\n");
    let mut shaders = HashMap::new();

    print!("*   Creating 'two_triangles_vert' shader module from file 'two_triangles_vert.spv'");
    shaders.insert("two_triangles_vert".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles_vert.spv")));
    println!(" ... OK'");

    print!("*   Creating 'two_triangles_frag' shader module from file 'two_triangles_frag.spv'");
    shaders.insert("two_triangles_frag".to_string(), device.create_shader_module(wgpu::include_spirv!("two_triangles_frag.spv")));
    println!(" ... OK'");
//    shaders.insert("two_triangles_frag".to_string(), fs_module);

//    let mc_module = device.create_shader_module(wgpu::include_spirv!("mc.spv"));
//    let ray_march_module = device.create_shader_module(wgpu::include_spirv!("ray_march.spv"));
//        
//    shaders.insert("mc_module".to_string(), mc_module);
//    shaders.insert("ray_march_module".to_string(), ray_march_module);

    shaders
}

fn create_vertex_buffers(device: &wgpu::Device, buffers: &mut HashMap::<String, gradu::Buffer>)  {

    // The Cube.
    let vertex_data = create_cube();
    let cube = Buffer::create_buffer_from_data::<f32>(
        device,
        &vertex_data,
        wgpu::BufferUsage::VERTEX,
        None);

    buffers.insert("cube_buffer".to_string(), cube);

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

    //let cube = Buffer::create_buffer_from_data::<f32>(
    //    device,
    //    &vertex_data,
    //    wgpu::BufferUsage::VERTEX,
    //    None);

    //buffers.insert("cube_buffer".to_string(), cube);
}

fn create_textures(device: &wgpu::Device, queue: &wgpu::Queue, sc_desc: &wgpu::SwapChainDescriptor, textures: &mut HashMap<String, gradu::Texture>) {

    println!("\nCreating textures.\n");
    // Two triangles texture.
    print!("*   Creating texture from file 'grass1.png'");
    let diffuse_texture = Texture::create_from_bytes(&queue, &device, &include_bytes!("grass1.png")[..], None);
    textures.insert(TWO_TRIANGLES_TEXTURE.name.to_string(), diffuse_texture);
    println!(" ... OK'");

//    let depth_texture = Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture"));
//    let tritable_texture = Texture::create_tritable(&queue, &device);
//    let ray_output_texture = Texture::create_texture2D(&queue, &device, 256,256);
//    textures.insert("depth_texture".to_string(), depth_texture);
//    textures.insert("tritable_texture".to_string(), tritable_texture);
//    textures.insert("ray_output_texture".to_string(), ray_output_texture);
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
