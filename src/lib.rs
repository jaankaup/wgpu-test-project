//use std::collections::HashMap;

use std::mem;
use bytemuck::{Pod, Zeroable};
use utils::math::{clamp};
//use num_traits::{Num};
use std::convert::TryInto;
//use fixed::traits::*;

use cgmath::{prelude::*, Vector3, Vector4, Point3};

use winit::{
    event::{WindowEvent,KeyboardInput,ElementState,VirtualKeyCode,MouseButton},
};

/// A trait for things that can copy and convert a wgpu-rs buffer to
/// a std::Vec. 
pub trait Convert2Vec where Self: std::marker::Sized {
    fn convert(data: &[u8]) -> Vec<Self>;  
}

/// A macro for creating Convert2Vec for specific a primitive 
/// number type. Note that the type must implement from_ne_bytes.
/// This works only in async functions. This cannot be used
/// in winit event_loop! Use it before entering event_loop.
macro_rules! impl_convert {
  ($to_type:ty) => {
    impl Convert2Vec for $to_type {
      //fn convert(&self, data: &mut [u8]) -> Vec<Self> {
      fn convert(data: &[u8]) -> Vec<Self> {
            let result = data
                .chunks_exact(std::mem::size_of::<Self>())
                .map(|b| Self::from_ne_bytes(b.try_into().unwrap()))
                .collect();
            result
      }
    }
  }
}

impl_convert!{f32}
impl_convert!{u32}
impl_convert!{u8}

/// Opengl to wgpu matrix
#[cfg_attr(rustfmt, surtfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0,
        );

/// Uniform data for marching cubes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mc_uniform_data {
    pub isovalue: f32,
    pub cube_length: f32,
    pub base_position: cgmath::Vector4<f32>,
}

unsafe impl Pod for Mc_uniform_data {}
unsafe impl Zeroable for Mc_uniform_data {}

///////////////////////////////////////////////////////////////////////////////////////

/// Buffer.
pub struct Buffer {
    pub buffer: wgpu::Buffer,
    pub capacity: usize,
    pub capacity_used: Option<usize>,
    pub label: Option<String>,
}

impl Buffer {

    pub fn create_buffer_from_data<T: Pod>(
        device: &wgpu::Device,
        t: &[T],
        usage: wgpu::BufferUsage,
        label: Option<String>)
    -> Self {
         
        let buffer = device.create_buffer_with_data(
            bytemuck::cast_slice(&t),
            usage,
        );
        let capacity = mem::size_of::<T>() * t.len();
        let capacity_used = Some(capacity);
        Self {
            buffer,
            capacity, 
            capacity_used, 
            label,
        }
    }
    
    /// Method for copying the content of the buffer into a vector.
    pub async fn to_vec<T: Convert2Vec>(&self, device: &wgpu::Device, queue: &wgpu::Queue, whole_buffer: bool) -> Vec<T> {

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.capacity as u64, 
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let size = match whole_buffer {
            true => self.capacity,
            false => match self.capacity_used {
                Some(some_size) => some_size as usize,
                None => {
                    let error_msg = "Buffer.to_vec is called with argument whole_buffer == false, \
                                     but capacity_used is None. Consider to define \
                                     Buffer.capacity.used with some value before calling \
                                     this method or call this method \
                                     with @whole_buffer == true \
                                     to get the content of whole buffer.";
                    panic!("{}", error_msg);
                }
            }
        };

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.capacity as wgpu::BufferAddress);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        //let buffer_future = staging_buffer.map_async(wgpu::MapMode::Read, 0, wgt::BufferSize::WHOLE);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        let res: Vec<T>;

        buffer_future.await.expect("failed"); 
        let data = buffer_slice.get_mapped_range();
        res = Convert2Vec::convert(&data);
        res
    }
}

///////////////////////////////////////////////////////////////////////////////////////

/// All possible texture types. TODO: Are these necessery?
pub enum TextureType {
    Diffuse,
    Depth,
}

/// Texture.
pub struct Texture {
    pub texture_type: TextureType, 
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {

    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    //pub const IMAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn create_depth_texture(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, label: Option<&str>) -> Self {
        let size = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: label,
            size,
            // array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_SRC,
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_default_view();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let texture_type = TextureType::Depth;

        Self { texture_type, texture, view, sampler }
    }

    /// Creates a texture from a sequency of bytes (expects bytes to be in png format in rgb). Now
    /// its adding automaticallhy an alpha value of
    /// 255 to the image. TODO: check if aplha value already exists. TODO: allow a texture to been
    /// created from non png data.
    pub fn create_from_bytes(queue: &wgpu::Queue, device: &wgpu::Device, bytes: &[u8], label: Option<&str>) -> Self {

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: Some(wgpu::CompareFunction::Always),
            ..Default::default()
        });

        let png = std::io::Cursor::new(bytes);
        let decoder = png::Decoder::new(png);
        let (info, mut reader) = decoder.read_info().expect("Can't read info!");
        let width = info.width;
        let height = info.height;
        let bits_per_pixel = info.color_type.samples() as u32;

        if !(bits_per_pixel == 3 || bits_per_pixel == 4) {
            panic!("Bits per pixel must be 3 or 4. Bits per pixel == {}", bits_per_pixel); 
        }

        // println!("THE BITS PER PIXEL == {}", bits_per_pixel);
        // println!("THE WIDTH == {}", width);
        // println!("THE HEIGHT == {}", height);
        // println!("{} * {} * {}  == {}", width, bits_per_pixel, height, width*bits_per_pixel*height);

        let mut buffer = vec![0; (info.width * bits_per_pixel * info.height) as usize ];
        reader.next_frame(&mut buffer).unwrap(); //expect("Can't read next frame.");

        // TODO: check the size of the image.


        let mut temp: Vec<u8> = Vec::new();
        let mut counter = 0; 

        // The png has only rgb components. Add the alpha component to each texel. 
        if bits_per_pixel == 3 {
            for i in 0..buffer.len() {
                if counter == 2 { counter = 0; temp.push(buffer[i]); temp.push(255); }
                else {
                    temp.push(buffer[i]);
                    counter = counter + 1;
                }
            }
        }

        let texture_extent = wgpu::Extent3d {
            width: width,
            height: height,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: label,
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            match bits_per_pixel {
                3 => &temp,
                4 => &buffer,
                _ => panic!("Bits size of {} is not supported", bits_per_pixel),
            },
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: width * 4, // now only 4 bits per pixel is supported,
                rows_per_image: height,
            },
            texture_extent,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            dimension: wgpu::TextureViewDimension::D2,
            aspect: wgpu::TextureAspect::default(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        });

        let texture_type = TextureType::Diffuse;

        Self {

            texture_type, 
            texture,
            view,
            sampler,
        }
    }

    pub fn create_texture2D(queue: &wgpu::Queue, device: &wgpu::Device, width: u32, height: u32) -> Self {

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: Some(wgpu::CompareFunction::Always),
            ..Default::default()
        });

        let texture_extent = wgpu::Extent3d {
            width: width,
            height: height,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            //format: wgpu::TextureFormat::Rgba8UnormSrgb,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            dimension: wgpu::TextureViewDimension::D2,
            aspect: wgpu::TextureAspect::default(),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
        });

        let texture_type = TextureType::Diffuse;

        Self {

            texture_type, 
            texture,
            view,
            sampler,
        }
    }

    /// Creates the tritable texture for marching cubes.
    /// Creates data in rgba from. 
    pub fn create_tritable(queue: &wgpu::Queue, device: &wgpu::Device) -> Self {
        let data: Vec<u8> = vec![
       255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 , // Case: 0
        0,  8,  3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,    // Case: 1
        0,  1,  9, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,    // Case: 2
        1,  8,  3,  9,  8,  1, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  8,  3,  1,  2, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9,  2, 10,  0,  2,  9, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        2,  8,  3,  2, 10,  8, 10,  9,  8, 255, 255, 255, 255, 255, 255 ,
        3, 11,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0, 11,  2,  8, 11,  0, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  9,  0,  2,  3, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1, 11,  2,  1,  9, 11,  9,  8, 11, 255, 255, 255, 255, 255, 255 ,
        3, 10,  1, 11, 10,  3, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0, 10,  1,  0,  8, 10,  8, 11, 10, 255, 255, 255, 255, 255, 255 ,
        3,  9,  0,  3, 11,  9, 11, 10,  9, 255, 255, 255, 255, 255, 255 ,
        9,  8, 10, 10,  8, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  7,  8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  3,  0,  7,  3,  4, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  1,  9,  8,  4,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  1,  9,  4,  7,  1,  7,  3,  1, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10,  8,  4,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  4,  7,  3,  0,  4,  1,  2, 10, 255, 255, 255, 255, 255, 255 ,
        9,  2, 10,  9,  0,  2,  8,  4,  7, 255, 255, 255, 255, 255, 255 ,
        2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, 255, 255, 255 ,
        8,  4,  7,  3, 11,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       11,  4,  7, 11,  2,  4,  2,  0,  4, 255, 255, 255, 255, 255, 255 ,
        9,  0,  1,  8,  4,  7,  2,  3, 11, 255, 255, 255, 255, 255, 255 ,
        4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, 255, 255, 255 ,
        3, 10,  1,  3, 11, 10,  7,  8,  4, 255, 255, 255, 255, 255, 255 ,
        1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, 255, 255, 255 ,
        4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, 255, 255, 255 ,
        4,  7, 11,  4, 11,  9,  9, 11, 10, 255, 255, 255, 255, 255, 255 ,
        9,  5,  4, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9,  5,  4,  0,  8,  3, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  5,  4,  1,  5,  0, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        8,  5,  4,  8,  3,  5,  3,  1,  5, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10,  9,  5,  4, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  0,  8,  1,  2, 10,  4,  9,  5, 255, 255, 255, 255, 255, 255 ,
        5,  2, 10,  5,  4,  2,  4,  0,  2, 255, 255, 255, 255, 255, 255 ,
        2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, 255, 255, 255 ,
        9,  5,  4,  2,  3, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0, 11,  2,  0,  8, 11,  4,  9,  5, 255, 255, 255, 255, 255, 255 ,
        0,  5,  4,  0,  1,  5,  2,  3, 11, 255, 255, 255, 255, 255, 255 ,
        2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, 255, 255, 255 ,
       10,  3, 11, 10,  1,  3,  9,  5,  4, 255, 255, 255, 255, 255, 255 ,
        4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, 255, 255, 255 ,
        5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, 255, 255, 255 ,
        5,  4,  8,  5,  8, 10, 10,  8, 11, 255, 255, 255, 255, 255, 255 ,
        9,  7,  8,  5,  7,  9, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9,  3,  0,  9,  5,  3,  5,  7,  3, 255, 255, 255, 255, 255, 255 ,
        0,  7,  8,  0,  1,  7,  1,  5,  7, 255, 255, 255, 255, 255, 255 ,
        1,  5,  3,  3,  5,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9,  7,  8,  9,  5,  7, 10,  1,  2, 255, 255, 255, 255, 255, 255 ,
       10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, 255, 255, 255 ,
        8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, 255, 255, 255 ,
        2, 10,  5,  2,  5,  3,  3,  5,  7, 255, 255, 255, 255, 255, 255 ,
        7,  9,  5,  7,  8,  9,  3, 11,  2, 255, 255, 255, 255, 255, 255 ,
        9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, 255, 255, 255 ,
        2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, 255, 255, 255 ,
       11,  2,  1, 11,  1,  7,  7,  1,  5, 255, 255, 255, 255, 255, 255 ,
        9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, 255, 255, 255 ,
        5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0 ,
       11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0 ,
       11, 10,  5,  7, 11,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       10,  6,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  8,  3,  5, 10,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9,  0,  1,  5, 10,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  8,  3,  1,  9,  8,  5, 10,  6, 255, 255, 255, 255, 255, 255 ,
        1,  6,  5,  2,  6,  1, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  6,  5,  1,  2,  6,  3,  0,  8, 255, 255, 255, 255, 255, 255 ,
        9,  6,  5,  9,  0,  6,  0,  2,  6, 255, 255, 255, 255, 255, 255 ,
        5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, 255, 255, 255 ,
        2,  3, 11, 10,  6,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       11,  0,  8, 11,  2,  0, 10,  6,  5, 255, 255, 255, 255, 255, 255 ,
        0,  1,  9,  2,  3, 11,  5, 10,  6, 255, 255, 255, 255, 255, 255 ,
        5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, 255, 255, 255 ,
        6,  3, 11,  6,  5,  3,  5,  1,  3, 255, 255, 255, 255, 255, 255 ,
        0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, 255, 255, 255 ,
        3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, 255, 255, 255 ,
        6,  5,  9,  6,  9, 11, 11,  9,  8, 255, 255, 255, 255, 255, 255 ,
        5, 10,  6,  4,  7,  8, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  3,  0,  4,  7,  3,  6,  5, 10, 255, 255, 255, 255, 255, 255 ,
        1,  9,  0,  5, 10,  6,  8,  4,  7, 255, 255, 255, 255, 255, 255 ,
       10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, 255, 255, 255 ,
        6,  1,  2,  6,  5,  1,  4,  7,  8, 255, 255, 255, 255, 255, 255 ,
        1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, 255, 255, 255 ,
        8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, 255, 255, 255 ,
        7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9 ,
        3, 11,  2,  7,  8,  4, 10,  6,  5, 255, 255, 255, 255, 255, 255 ,
        5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, 255, 255, 255 ,
        0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, 255, 255, 255 ,
        9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6 ,
        8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, 255, 255, 255 ,
        5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11 ,
        0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7 ,
        6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, 255, 255, 255 ,
       10,  4,  9,  6,  4, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4, 10,  6,  4,  9, 10,  0,  8,  3, 255, 255, 255, 255, 255, 255 ,
       10,  0,  1, 10,  6,  0,  6,  4,  0, 255, 255, 255, 255, 255, 255 ,
        8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, 255, 255, 255 ,
        1,  4,  9,  1,  2,  4,  2,  6,  4, 255, 255, 255, 255, 255, 255 ,
        3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, 255, 255, 255 ,
        0,  2,  4,  4,  2,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        8,  3,  2,  8,  2,  4,  4,  2,  6, 255, 255, 255, 255, 255, 255 ,
       10,  4,  9, 10,  6,  4, 11,  2,  3, 255, 255, 255, 255, 255, 255 ,
        0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, 255, 255, 255 ,
        3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, 255, 255, 255 ,
        6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1 ,
        9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, 255, 255, 255 ,
        8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1 ,
        3, 11,  6,  3,  6,  0,  0,  6,  4, 255, 255, 255, 255, 255, 255 ,
        6,  4,  8, 11,  6,  8, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        7, 10,  6,  7,  8, 10,  8,  9, 10, 255, 255, 255, 255, 255, 255 ,
        0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, 255, 255, 255 ,
       10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, 255, 255, 255 ,
       10,  6,  7, 10,  7,  1,  1,  7,  3, 255, 255, 255, 255, 255, 255 ,
        1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, 255, 255, 255 ,
        2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9 ,
        7,  8,  0,  7,  0,  6,  6,  0,  2, 255, 255, 255, 255, 255, 255 ,
        7,  3,  2,  6,  7,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, 255, 255, 255 ,
        2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7 ,
        1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11 ,
       11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, 255, 255, 255 ,
        8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6 ,
        0,  9,  1, 11,  6,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, 255, 255, 255 ,
        7, 11,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        7,  6, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  0,  8, 11,  7,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  1,  9, 11,  7,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        8,  1,  9,  8,  3,  1, 11,  7,  6, 255, 255, 255, 255, 255, 255 ,
       10,  1,  2,  6, 11,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10,  3,  0,  8,  6, 11,  7, 255, 255, 255, 255, 255, 255 ,
        2,  9,  0,  2, 10,  9,  6, 11,  7, 255, 255, 255, 255, 255, 255 ,
        6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, 255, 255, 255 ,
        7,  2,  3,  6,  2,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        7,  0,  8,  7,  6,  0,  6,  2,  0, 255, 255, 255, 255, 255, 255 ,
        2,  7,  6,  2,  3,  7,  0,  1,  9, 255, 255, 255, 255, 255, 255 ,
        1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, 255, 255, 255 ,
       10,  7,  6, 10,  1,  7,  1,  3,  7, 255, 255, 255, 255, 255, 255 ,
       10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, 255, 255, 255 ,
        0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, 255, 255, 255 ,
        7,  6, 10,  7, 10,  8,  8, 10,  9, 255, 255, 255, 255, 255, 255 ,
        6,  8,  4, 11,  8,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  6, 11,  3,  0,  6,  0,  4,  6, 255, 255, 255, 255, 255, 255 ,
        8,  6, 11,  8,  4,  6,  9,  0,  1, 255, 255, 255, 255, 255, 255 ,
        9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, 255, 255, 255 ,
        6,  8,  4,  6, 11,  8,  2, 10,  1, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, 255, 255, 255 ,
        4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, 255, 255, 255 ,
       10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3 ,
        8,  2,  3,  8,  4,  2,  4,  6,  2, 255, 255, 255, 255, 255, 255 ,
        0,  4,  2,  4,  6,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, 255, 255, 255 ,
        1,  9,  4,  1,  4,  2,  2,  4,  6, 255, 255, 255, 255, 255, 255 ,
        8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, 255, 255, 255 ,
       10,  1,  0, 10,  0,  6,  6,  0,  4, 255, 255, 255, 255, 255, 255 ,
        4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3 ,
       10,  9,  4,  6, 10,  4, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  9,  5,  7,  6, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  8,  3,  4,  9,  5, 11,  7,  6, 255, 255, 255, 255, 255, 255 ,
        5,  0,  1,  5,  4,  0,  7,  6, 11, 255, 255, 255, 255, 255, 255 ,
       11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, 255, 255, 255 ,
        9,  5,  4, 10,  1,  2,  7,  6, 11, 255, 255, 255, 255, 255, 255 ,
        6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, 255, 255, 255 ,
        7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, 255, 255, 255 ,
        3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6 ,
        7,  2,  3,  7,  6,  2,  5,  4,  9, 255, 255, 255, 255, 255, 255 ,
        9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, 255, 255, 255 ,
        3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, 255, 255, 255 ,
        6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8 ,
        9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, 255, 255, 255 ,
        1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4 ,
        4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10 ,
        7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, 255, 255, 255 ,
        6,  9,  5,  6, 11,  9, 11,  8,  9, 255, 255, 255, 255, 255, 255 ,
        3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, 255, 255, 255 ,
        0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, 255, 255, 255 ,
        6, 11,  3,  6,  3,  5,  5,  3,  1, 255, 255, 255, 255, 255, 255 ,
        1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, 255, 255, 255 ,
        0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10 ,
       11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5 ,
        6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, 255, 255, 255 ,
        5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, 255, 255, 255 ,
        9,  5,  6,  9,  6,  0,  0,  6,  2, 255, 255, 255, 255, 255, 255 ,
        1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8 ,
        1,  5,  6,  2,  1,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6 ,
       10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, 255, 255, 255 ,
        0,  3,  8,  5,  6, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       10,  5,  6, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       11,  5, 10,  7,  5, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
       11,  5, 10, 11,  7,  5,  8,  3,  0, 255, 255, 255, 255, 255, 255 ,
        5, 11,  7,  5, 10, 11,  1,  9,  0, 255, 255, 255, 255, 255, 255 ,
       10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, 255, 255, 255 ,
       11,  1,  2, 11,  7,  1,  7,  5,  1, 255, 255, 255, 255, 255, 255 ,
        0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, 255, 255, 255 ,
        9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, 255, 255, 255 ,
        7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2 ,
        2,  5, 10,  2,  3,  5,  3,  7,  5, 255, 255, 255, 255, 255, 255 ,
        8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, 255, 255, 255 ,
        9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, 255, 255, 255 ,
        9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2 ,
        1,  3,  5,  3,  7,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  8,  7,  0,  7,  1,  1,  7,  5, 255, 255, 255, 255, 255, 255 ,
        9,  0,  3,  9,  3,  5,  5,  3,  7, 255, 255, 255, 255, 255, 255 ,
        9,  8,  7,  5,  9,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        5,  8,  4,  5, 10,  8, 10, 11,  8, 255, 255, 255, 255, 255, 255 ,
        5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, 255, 255, 255 ,
        0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, 255, 255, 255 ,
       10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4 ,
        2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, 255, 255, 255 ,
        0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11 ,
        0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5 ,
        9,  4,  5,  2, 11,  3, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, 255, 255, 255 ,
        5, 10,  2,  5,  2,  4,  4,  2,  0, 255, 255, 255, 255, 255, 255 ,
        3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9 ,
        5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, 255, 255, 255 ,
        8,  4,  5,  8,  5,  3,  3,  5,  1, 255, 255, 255, 255, 255, 255 ,
        0,  4,  5,  1,  0,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, 255, 255, 255 ,
        9,  4,  5, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4, 11,  7,  4,  9, 11,  9, 10, 11, 255, 255, 255, 255, 255, 255 ,
        0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, 255, 255, 255 ,
        1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, 255, 255, 255 ,
        3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4 ,
        4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, 255, 255, 255 ,
        9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3 ,
       11,  7,  4, 11,  4,  2,  2,  4,  0, 255, 255, 255, 255, 255, 255 ,
       11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, 255, 255, 255 ,
        2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, 255, 255, 255 ,
        9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7 ,
        3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10 ,
        1, 10,  2,  8,  7,  4, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  9,  1,  4,  1,  7,  7,  1,  3, 255, 255, 255, 255, 255, 255 ,
        4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, 255, 255, 255 ,
        4,  0,  3,  7,  4,  3, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        4,  8,  7, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        9, 10,  8, 10, 11,  8, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  0,  9,  3,  9, 11, 11,  9, 10, 255, 255, 255, 255, 255, 255 ,
        0,  1, 10,  0, 10,  8,  8, 10, 11, 255, 255, 255, 255, 255, 255 ,
        3,  1, 10, 11,  3, 10, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  2, 11,  1, 11,  9,  9, 11,  8, 255, 255, 255, 255, 255, 255 ,
        3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, 255, 255, 255 ,
        0,  2, 11,  8,  0, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        3,  2, 11, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        2,  3,  8,  2,  8, 10, 10,  8,  9, 255, 255, 255, 255, 255, 255 ,
        9, 10,  2,  0,  9,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, 255, 255, 255 ,
        1, 10,  2, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        1,  3,  8,  9,  1,  8, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  9,  1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 ,
        0,  3,  8, 255, 255, 255, 255 ,255 ,255, 255 ,255 ,255, 255 ,255 ,255 ,
       255, 255, 255, 255, 255, 255, 255 ,255 ,255, 255 ,255 ,255, 255 ,255 ,255];

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            //compare: Some(wgpu::CompareFunction::Never),
            compare: Some(wgpu::CompareFunction::Equal),
            ..Default::default()
        });

        // Add alpha component to the data. 
        
        //let mut buffer = vec![0; 5120];
        let mut buffer = Vec::new(); //vec![0; 5120];

        // Add the aplha value (== 255) for each pixel.
        let mut counter = 0; 
        for i in 0..data.len() {
            if counter == 2 { counter = 0; buffer.push(data[i]); buffer.push(255); }
            else {
                buffer.push(data[i]);
                counter = counter + 1;
            }
        }

        let texture_extent = wgpu::Extent3d {
            width: 1280,
            height: 1,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            // array_layer_count: 1, // only one texture now
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            //format: wgpu::TextureFormat::Rgba8Uint,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
            label: None,
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &buffer,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 5120,
                rows_per_image: 1,
            },
            texture_extent,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            dimension: wgpu::TextureViewDimension::D1,
            aspect: wgpu::TextureAspect::default(),
            base_mip_level: 0,
            level_count: 0,
            base_array_layer: 0,
            array_layer_count: 0,
        });

        let texture_type = TextureType::Diffuse;

        Self {

            texture_type, 
            texture,
            view,
            sampler,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////
  
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    _pos: [f32; 4],
    _normal: [f32; 4],
}

#[allow(dead_code)]
pub fn vertex(pos: [f32; 3], nor: [f32; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1.0],
        _normal: [nor[0], nor[1], nor[2], 0.0],
    }
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

///////////////////////////////////////////////////////////////////////////////////////

/// A camera for ray tracing purposes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RayCamera {
    pub pos: cgmath::Vector3<f32>,
    pub view: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub fov: cgmath::Vector2<f32>,
    pub apertureRadius: f32,
    pub focalDistance: f32,
}

unsafe impl bytemuck::Zeroable for RayCamera {}
unsafe impl bytemuck::Pod for RayCamera {}

///////////////////////////////////////////////////////////////////////////////////////

/// A camera for basic rendering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub pos: cgmath::Vector3<f32>,
    pub view: cgmath::Vector3<f32>,
    pub up: cgmath::Vector3<f32>,
    pub aspect: f32,
    pub fov: cgmath::Vector2<f32>,
    pub znear: f32,
    pub zfar: f32,
}

unsafe impl bytemuck::Zeroable for Camera {}
unsafe impl bytemuck::Pod for Camera {}

impl Camera {

    /// Creates a pv matrix for wgpu.
    pub fn build_projection_matrix(&self) -> cgmath::Matrix4<f32> {

        let view = self.build_view_matrix();
        let proj = cgmath::perspective(cgmath::Rad(std::f32::consts::PI/2.0), self.aspect, self.znear, self.zfar);

        OPENGL_TO_WGPU_MATRIX * (proj * view)
    }

    pub fn build_view_matrix(&self) -> cgmath::Matrix4<f32> {
        let pos3 = Point3::new(self.pos.x, self.pos.y,self.pos.z);
        let view3 = Point3::new(self.view.x + pos3.x, self.view.y + pos3.y, self.view.z + pos3.z);
        let view = cgmath::Matrix4::look_at(pos3, view3, self.up);
        view
    }
}

///////////////////////////////////////////////////////////////////////////////////////

/// A controller for handling the input and state of camera related operations.
pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_left_mouse_pressed: bool,
    start_mouse_pos: Option<(f64,f64)>,
    current_mouse_pos: Option<(f64,f64)>,
    pub pitch: f32,
    pub yaw: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_left_mouse_pressed: false,
            start_mouse_pos: Some((0 as f64,0 as f64)),
            current_mouse_pos: Some((0 as f64,0 as f64)),
            pitch: -80.5,
            yaw: -50.0,
        }
    }
    
    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                let event_happened =
                match keycode {
                    VirtualKeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::C => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                };

                event_happened // TODO: remove

            }, // WindowEvent::KeyboardInput

            WindowEvent::MouseInput {
                    state,
                    button,
                    ..
            } => { 
                let is_pressed = *state == ElementState::Pressed;
                let event_happened =
                match button {
                    MouseButton::Left => {
                        self.is_left_mouse_pressed = is_pressed;
                        true
                    }
                    _ => false,
                };

                event_happened
            }, // WindowEvent::MouseEvent

            WindowEvent::CursorMoved {
                    position,
                    ..
            } => { 

                // Initial mouse positions.
                match self.start_mouse_pos {
                    Some(_) => { },
                    None      => {
                        self.start_mouse_pos = Some((position.x, position.y));
                        self.current_mouse_pos = Some((position.x, position.y));
                    },
                }

                // Update both previous and current mouse positions.
                self.start_mouse_pos = self.current_mouse_pos;
                self.current_mouse_pos = Some((position.x, position.y));

                true
            }, // WindowEvent::CursorMoved

            _ => false, // ignore other events
        } // event
    } // end func

    pub fn update_camera(&mut self, camera: &mut Camera) {
                                                               
        let forward = camera.view;

        if self.is_forward_pressed {
            camera.pos += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.pos -= forward * self.speed;
        }

        let right = forward.cross(camera.up);

        if self.is_right_pressed {
            camera.pos += right * self.speed;
        }
        if self.is_left_pressed {
            camera.pos -= right * self.speed;
        }
        if self.is_up_pressed {
            camera.pos += camera.up * self.speed;
        }
        if self.is_down_pressed {
            camera.pos -= camera.up * self.speed;
        }
        if self.is_left_mouse_pressed {
            // Update mouse delta.
            let (x0, y0) = self.start_mouse_pos.unwrap();
            let (x1, y1) = self.current_mouse_pos.unwrap();
            let (x,y) = (x1 - x0, y1 - y0); 

            self.pitch = clamp(self.pitch + (self.sensitivity as f32 * (y * (-1.0)) as f32) , -89.0,89.0);
            self.yaw = self.yaw + self.sensitivity * x as f32 ;

            println!("yaw/pitch = ({},{})", self.yaw, self.pitch);

            camera.view = Vector3::new(
                self.pitch.to_radians().cos() * self.yaw.to_radians().cos(),
                self.pitch.to_radians().sin(),
                self.pitch.to_radians().cos() * self.yaw.to_radians().sin()
            ).normalize_to(1.0);

            println!("view = ({},{},{})", camera.view.x, camera.view.y, camera.view.z);

        }
    }

    //TODO: refactor.
    pub fn update_ray_camera(&mut self, camera: &mut RayCamera) {
                                                               
        let forward = camera.view;

        if self.is_forward_pressed {
            camera.pos += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.pos -= forward * self.speed;
        }

        let right = forward.cross(camera.up);

        if self.is_right_pressed {
            camera.pos += right * self.speed;
        }
        if self.is_left_pressed {
            camera.pos -= right * self.speed;
        }
        if self.is_up_pressed {
            camera.pos += camera.up * self.speed;
        }
        if self.is_down_pressed {
            camera.pos -= camera.up * self.speed;
        }
        if self.is_left_mouse_pressed {
            // Update mouse delta.
            let (x0, y0) = self.start_mouse_pos.unwrap();
            let (x1, y1) = self.current_mouse_pos.unwrap();
            let (x,y) = (x1 - x0, y1 - y0); 

            self.pitch = clamp(self.pitch + (self.sensitivity as f32 * (y * (-1.0)) as f32) , -89.0,89.0);
            self.yaw = self.yaw + self.sensitivity * x as f32 ;

            println!("yaw/pitch = ({},{})", self.yaw, self.pitch);

            camera.view = Vector3::new(
                self.pitch.to_radians().cos() * self.yaw.to_radians().cos(),
                self.pitch.to_radians().sin(),
                self.pitch.to_radians().cos() * self.yaw.to_radians().sin()
            ).normalize_to(1.0);

            println!("ray_view = ({},{},{})", camera.view.x, camera.view.y, camera.view.z);

        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////

///// TODO: remove this. Add to the RayCamera. 
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RayCameraUniform {
    pos: cgmath::Vector4<f32>,  // eye
    view: cgmath::Vector4<f32>, // target    // original: float3
    up: cgmath::Vector4<f32>,
    fov: cgmath::Vector4<f32>, // fovy
    apertureRadius: f32, // new!
    focalDistance: f32, // new!
}

impl RayCameraUniform {
    pub fn new() -> Self {
        Self {
            pos: (1.0, 1.0, 1.0, 1.0).into(),
            view: Vector4::new(0.0, 0.0, -1.0, 0.0).normalize(),
            up: cgmath::Vector4::unit_y(),
            fov: ((45.0 as f32).to_radians(),
                 (45.0 as f32).to_radians(),
                 111.0,
                 222.0).into(),
            apertureRadius: 0.0,
            focalDistance: 1.0,
        }
    }

    pub fn update(&mut self, camera: &RayCamera) {
            self.pos  = cgmath::Vector4::new(camera.pos.x, camera.pos.y,  camera.pos.z, 1.0);  
            self.view = cgmath::Vector4::new(camera.view.x, camera.view.y, camera.view.z, 0.0);
            self.up   = cgmath::Vector4::new(camera.up.x, camera.up.y,   camera.up.z, 0.0);  
            self.fov  = cgmath::Vector4::new(camera.fov.x, camera.fov.y, 123.0, 234.0); // 2 dummy values. 
            self.apertureRadius = camera.apertureRadius;
            self.focalDistance = camera.focalDistance;
    }
}

unsafe impl bytemuck::Zeroable for RayCameraUniform {}
unsafe impl bytemuck::Pod for RayCameraUniform {}

///////////////////////////////////////////////////////////////////////////////////////

/// Camera uniform data for the shader.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CameraUniform {
    view_proj: cgmath::Matrix4<f32>,
    pos: cgmath::Vector3<f32>,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: cgmath::Matrix4::identity(),
            pos: cgmath::Vector3::new(1.0,1.0,1.0),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_projection_matrix();
        self.pos = camera.pos;
    }
}
 
unsafe impl bytemuck::Zeroable for CameraUniform {}
unsafe impl bytemuck::Pod for CameraUniform {}

///////////////////////////////////////////////////////////////////////////////////////

/// Data for textured cube. vvvttnnn vvvttnnn vvvttnnn ...
#[allow(dead_code)]
pub fn create_cube() -> Vec<f32> {

    let v_data = [[1.0 , -1.0, -1.0],
                  [1.0 , -1.0, 1.0],
                  [-1.0, -1.0, 1.0],
                  [-1.0, -1.0, -1.0],
                  [1.0 , 1.0, -1.0],
                  [1.0, 1.0, 1.0],
                  [-1.0, 1.0, 1.0],
                  [-1.0, 1.0, -1.0],
    ];

    let t_data = [[0.748573,0.750412],
                 [0.749279,0.501284],
                 [0.999110,0.501077],
                 [0.999455,0.750380],
                 [0.250471,0.500702],
                 [0.249682,0.749677],
                 [0.001085,0.750380],
                 [0.001517,0.499994],
                 [0.499422,0.500239],
                 [0.500149,0.750166],
                 [0.748355,0.998230],
                 [0.500193,0.998728],
                 [0.498993,0.250415],
                 [0.748953,0.250920],
    ];
    
    let n_data = [ 
                  [0.0 , 0.0 , -1.0],
                  [-1.0, -0.0, 0.0],
                  [0.0, -0.0, 1.0],
                  [0.0, 0.0 , 1.0],
                  [1.0 , -0.0, 0.0],
                  [1.0 , 0.0 , 0.0],
                  [0.0 , 1.0 , 0.0],
                  [0.0, -1.0, 0.0],
    ];

    let mut vs: Vec<[f32; 3]> = Vec::new();
    let mut ts: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 3]> = Vec::new();

    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[0]);
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[0]);
    vs.push(v_data[3]);
    ts.push(t_data[2]);
    vn.push(n_data[0]);

    // Face2
    //  f 5/1/1 4/3/1 8/4/1
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[0]);
    vs.push(v_data[3]);
    ts.push(t_data[2]);
    vn.push(n_data[0]);
    vs.push(v_data[7]);
    ts.push(t_data[3]);
    vn.push(n_data[0]);

    // Face3
    //  f 3/5/2 7/6/2 8/7/2
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[1]);
    vs.push(v_data[6]);
    ts.push(t_data[5]);
    vn.push(n_data[1]);
    vs.push(v_data[7]);
    ts.push(t_data[6]);
    vn.push(n_data[1]);

  // Face4
//  f 3/5/2 8/7/2 4/8/2
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[1]);
    vs.push(v_data[7]);
    ts.push(t_data[6]);
    vn.push(n_data[1]);
    vs.push(v_data[3]);
    ts.push(t_data[7]);
    vn.push(n_data[1]);

  // Face5
//  f 2/9/3 6/10/3 3/5/3
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[2]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[2]);
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[2]);

  // Face6
//  f 6/10/4 7/6/4 3/5/4
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[3]);
    vs.push(v_data[6]);
    ts.push(t_data[5]);
    vn.push(n_data[3]);
    vs.push(v_data[2]);
    ts.push(t_data[4]);
    vn.push(n_data[3]);

  // Face7
//  f 1/2/5 5/1/5 2/9/5
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[4]);
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[4]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[4]);

  // Face8
//  f 5/1/6 6/10/6 2/9/6
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[5]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[5]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[5]);

  // Face9
//  f 5/1/7 8/11/7 6/10/7
    vs.push(v_data[4]);
    ts.push(t_data[0]);
    vn.push(n_data[6]);
    vs.push(v_data[7]);
    ts.push(t_data[10]);
    vn.push(n_data[6]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[6]);

  // Face10
 // f 8/11/7 7/12/7 6/10/7
    vs.push(v_data[7]);
    ts.push(t_data[10]);
    vn.push(n_data[6]);
    vs.push(v_data[6]);
    ts.push(t_data[11]);
    vn.push(n_data[6]);
    vs.push(v_data[5]);
    ts.push(t_data[9]);
    vn.push(n_data[6]);

  // Face11
//  f 1/2/8 2/9/8 3/13/8
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[7]);
    vs.push(v_data[1]);
    ts.push(t_data[8]);
    vn.push(n_data[7]);
    vs.push(v_data[2]);
    ts.push(t_data[12]);
    vn.push(n_data[7]);

  // Face12
//  f 1/2/8 3/13/8 4/14/8
    vs.push(v_data[0]);
    ts.push(t_data[1]);
    vn.push(n_data[7]);
    vs.push(v_data[2]);
    ts.push(t_data[12]);
    vn.push(n_data[7]);
    vs.push(v_data[3]);
    ts.push(t_data[13]);
    vn.push(n_data[7]);

    let mut p_data: Vec<f32> = Vec::new();

    for i in 0..vs.len() {
        p_data.push(vs[i][0]); 
        p_data.push(vs[i][1]); 
        p_data.push(vs[i][2]); 
        p_data.push(ts[i][0]); 
        p_data.push(ts[i][1]); 
        p_data.push(vn[i][0]); 
        p_data.push(vn[i][1]); 
        p_data.push(vn[i][2]); 
    }

    p_data

}

