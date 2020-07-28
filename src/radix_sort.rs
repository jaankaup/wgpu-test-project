//mod radix_sort {

    // k : number of bits per key
    // d : number of bits per digit
    // KPT : nubmer of keys per thread
    // KPB : number of keys per block
    // O^ : threshold for local sorting
    // O_ : threshold for merging buckets
    //
    // M1 : Input and auxiliary memory: 2 * n * k/8
    // M2 : Bucket histogarms: 4 * r * floor(n/O^)
    // M3 : Block histogarms: 4 * r * (floor(n/KPB) + floor(n/O_))
    // M4 : Block assigments: 2 * 16 * (floor(n/KPB) + floor(n/O^))
    // M5 : Local sort sub-bucket assignments: 12 * min(floor(2 * n/O_) + floor(n/O^), r * (floor(n/O^))

    use gradu::{Buffer};

    static LOCAL_SORT_THRESHOLD: u32 = 8192; // 9216 
    static THREADS: u32 = 256;
    static KPT: u32 = 16; //18
    static KPB: u32 = 4096; // 6912;
    static MERGE_THRESHOLD: u32 = 3000; 

    pub struct KeyBlock {
        pub key_offset: u32,
        pub key_count: u32,
        pub bucket_id: u32,
        pub bucket_offset: u32,
    }
    
    pub struct LocalSortBlock {
        pub bucket_id: u32,
        pub bucket_offset: u32,
        pub is_merged: u32,
    }


    /// Create both main buffers for radix sort.
    /// parametrize the type of buffer. 
    pub fn create_radix_buffers(device: &wgpu::Device, data: &[u32]) -> (gradu::Buffer, gradu::Buffer, gradu::Buffer) {
        let initial_buffer = Buffer::create_buffer_from_data::<u32>(
            device,
            data,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
            None);
    
        let swap_buffer = Buffer::create_buffer(
            device,
            (std::mem::size_of::<u32>() * data.len()) as u64,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
            None);

        let histogram_buffer = Buffer::create_buffer(
            device,
            (std::mem::size_of::<u32>() * 256) as u64,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
            None);
        
        (initial_buffer, swap_buffer, histogram_buffer)
    }

    /// Creates key_blocks from the given key range.
    pub fn create_key_blocks(start_index: u32, number_of_keys: u32, bucket_id: u32) -> Vec<KeyBlock>  {
        assert!(!(number_of_keys < LOCAL_SORT_THRESHOLD));

        let mut key_blocks: Vec<KeyBlock> = Vec::new();
        let mut keys_available = number_of_keys;

        let mut key_offset = start_index;

        while keys_available > 0 {
            let mut key_count = 0;
            if keys_available >= KPB {
                keys_available = keys_available - KPB;
                key_count = KPB;
            }
            else {
                key_count = keys_available;
                keys_available = 0;
            }

            key_blocks.push(KeyBlock{
                key_offset: key_offset,
                key_count: key_count,
                bucket_id: bucket_id,
                bucket_offset: start_index,
            });

            key_offset = key_offset + KPB;
        }

        key_blocks
    }

    pub fn max_number_of_buckets(n: u32, radix: u8) -> u32 {
        (radix as u32) * ((n/LOCAL_SORT_THRESHOLD + 1) as u32)
    }

// R1. Every bucket with size of n < LOCAL_SORT_THRESHOLD is sorted using local sort (bitonic sort).
// R2. Every bucket wtih size of n > LOCAL_SORT_THRESHOLD is divided to r sub-buckets using counting sort.
// R3. Sequence of sub-buckets is merged until total number of keys < MERGE_THRESHOLD.
// R4. 

//}
