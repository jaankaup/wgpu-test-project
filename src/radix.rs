use gradu::{Buffer};

/// Create both main buffers for radix sort. Returns (initial_buffer, swap_buffer). TODO:
/// parametrize the type of buffer. 
pub fn create_radix_buffers(device: &wgpu::Device, data: &[u32]) -> (gradu::Buffer, gradu::Buffer) {
    let initial_buffer = Buffer::create_buffer_from_data::<f32>(
        device,
        &data,
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None);

    pub fn create_buffer(device: &wgpu::Device, capacity: u64, usage: wgpu::BufferUsage, label: Option<&str>) -> Self {
    let swap_buffer = Buffer::create_buffer(
        device,
        std::mem::size_of::<u32>() * data.len(),
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        None);

    (initial_buffer, swap_buffer)
}
