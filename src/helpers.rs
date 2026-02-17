// AOgmaNeo Rust port - helpers module

use std::sync::atomic::{AtomicUsize, Ordering};

// --- Constants ---

pub const LIMIT_MIN: f32 = -999999.0;
pub const LIMIT_MAX: f32 = 999999.0;
pub const LIMIT_SMALL: f32 = 0.000001;

pub const RAND_SUBSEED_OFFSET: u64 = 12345;
pub const INIT_WEIGHT_NOISEI: u32 = 8;
pub const INIT_WEIGHT_NOISEF: f32 = 0.01;

pub const SOFTPLUS_LIMIT: f32 = 4.0;

const PCG_MULTIPLIER: u64 = 6364136223846793005;
const PCG_INCREMENT: u64 = 1442695040888963407;
pub const RAND_MAX: u32 = 0x00ffffff;

// --- Type aliases ---

pub type ByteBuffer = Vec<u8>;
pub type SByteBuffer = Vec<i8>;
pub type UShortBuffer = Vec<u16>;
pub type ShortBuffer = Vec<i16>;
pub type UIntBuffer = Vec<u32>;
pub type IntBuffer = Vec<i32>;
pub type FloatBuffer = Vec<f32>;

// --- Vector types ---

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int2 {
    pub x: i32,
    pub y: i32,
}

impl Int2 {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Int3 {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Int4 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
}

impl Int4 {
    pub fn new(x: i32, y: i32, z: i32, w: i32) -> Self {
        Self { x, y, z, w }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl Float2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Float3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Float4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}

// --- Math helpers ---

pub fn ceil_divide(x: i32, y: i32) -> i32 {
    (x + y - 1) / y
}

pub fn sigmoidf(x: f32) -> f32 {
    x.tanh() * 0.5 + 0.5
}

pub fn logitf(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

pub fn symlogf(x: f32) -> f32 {
    ((x > 0.0) as i32 as f32 * 2.0 - 1.0) * (x.abs() + 1.0).ln()
}

pub fn symexpf(x: f32) -> f32 {
    ((x > 0.0) as i32 as f32 * 2.0 - 1.0) * (x.abs().exp() - 1.0)
}

pub fn softplusf(x: f32) -> f32 {
    let in_range = if x < SOFTPLUS_LIMIT { 1.0f32 } else { 0.0f32 };
    (1.0 + (x * in_range).exp()).ln() * in_range + x * (1.0 - in_range)
}

pub fn ceilf_to_i32(x: f32) -> i32 {
    x.ceil() as i32
}

pub fn roundf2i(x: f32) -> i32 {
    (x + if x > 0.0 { 1.0 } else { 0.0 } - 0.5) as i32
}

pub fn roundf2b(x: f32) -> u8 {
    (x + 0.5) as u8
}

pub fn roundf2sb(x: f32) -> i8 {
    (x + if x > 0.0 { 1.0 } else { 0.0 } - 0.5) as i8
}

// --- Bounds checking ---

pub fn in_bounds0(pos: Int2, upper_bound: Int2) -> bool {
    pos.x >= 0 && pos.x < upper_bound.x && pos.y >= 0 && pos.y < upper_bound.y
}

pub fn in_bounds(pos: Int2, lower_bound: Int2, upper_bound: Int2) -> bool {
    pos.x >= lower_bound.x
        && pos.x < upper_bound.x
        && pos.y >= lower_bound.y
        && pos.y < upper_bound.y
}

// --- Projections ---

pub fn project(pos: Int2, to_scalars: Float2) -> Int2 {
    Int2::new(
        ((pos.x as f32 + 0.5) * to_scalars.x) as i32,
        ((pos.y as f32 + 0.5) * to_scalars.y) as i32,
    )
}

pub fn projectf(pos: Float2, to_scalars: Float2) -> Int2 {
    Int2::new(
        ((pos.x + 0.5) * to_scalars.x) as i32,
        ((pos.y + 0.5) * to_scalars.y) as i32,
    )
}

pub fn min_overhang(pos: Int2, size: Int2, radii: Int2) -> Int2 {
    let mut new_pos = pos;

    let overhang_px = new_pos.x + radii.x >= size.x;
    let overhang_nx = new_pos.x - radii.x < 0;
    let overhang_py = new_pos.y + radii.y >= size.y;
    let overhang_ny = new_pos.y - radii.y < 0;

    if overhang_px && !overhang_nx {
        new_pos.x = size.x - 1 - radii.x;
    } else if overhang_nx && !overhang_px {
        new_pos.x = radii.x;
    }

    if overhang_py && !overhang_ny {
        new_pos.y = size.y - 1 - radii.y;
    } else if overhang_ny && !overhang_py {
        new_pos.y = radii.y;
    }

    new_pos
}

pub fn min_overhang_scalar(pos: Int2, size: Int2, radius: i32) -> Int2 {
    min_overhang(pos, size, Int2::new(radius, radius))
}

// --- Addressing (row-major) ---

pub fn address2(pos: Int2, dims: Int2) -> usize {
    (pos.y + pos.x * dims.y) as usize
}

pub fn address3(pos: Int3, dims: Int3) -> usize {
    (pos.z + dims.z * (pos.y + dims.y * pos.x)) as usize
}

pub fn address4(pos: Int4, dims: Int4) -> usize {
    (pos.w + dims.w * (pos.z + dims.z * (pos.y + dims.y * pos.x))) as usize
}

// --- PCG32 RNG ---

pub fn rand_get_state(seed: u64) -> u64 {
    let state = seed.wrapping_add(PCG_INCREMENT);
    state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(PCG_INCREMENT)
}

#[inline]
fn rotr32(x: u32, r: u32) -> u32 {
    x >> r | x << (r.wrapping_neg() & 31)
}

pub fn rand_step(state: &mut u64) -> u32 {
    let x = *state;
    let count = (x >> 59) as u32;
    *state = x.wrapping_mul(PCG_MULTIPLIER).wrapping_add(PCG_INCREMENT);
    let x = x ^ (x >> 18);
    rotr32((x >> 27) as u32, count)
}

pub fn randf_step(state: &mut u64) -> f32 {
    (rand_step(state) % RAND_MAX) as f32 / RAND_MAX as f32
}

pub fn randf_range_step(low: f32, high: f32, state: &mut u64) -> f32 {
    low + (high - low) * randf_step(state)
}

pub fn rand_normalf_step(state: &mut u64) -> f32 {
    let u1 = randf_step(state);
    let u2 = randf_step(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

pub fn rand_roundf_step(x: f32, state: &mut u64) -> i32 {
    let i = x as i32;
    let abs_rem = (x - i as f32).abs();
    let s = if x > 0.0 { 1i32 } else { -1i32 };
    i + if randf_step(state) < abs_rem { s } else { 0 }
}

// Global RNG state (thread-local to avoid contention)
thread_local! {
    static GLOBAL_STATE: std::cell::Cell<u64> = std::cell::Cell::new(rand_get_state(12345));
}

pub fn global_rand() -> u32 {
    GLOBAL_STATE.with(|s| {
        let mut state = s.get();
        let r = rand_step(&mut state);
        s.set(state);
        r
    })
}

pub fn global_randf() -> f32 {
    GLOBAL_STATE.with(|s| {
        let mut state = s.get();
        let r = randf_step(&mut state);
        s.set(state);
        r
    })
}

// --- CircleBuffer ---

#[derive(Clone, Debug)]
pub struct CircleBuffer<T> {
    pub data: Vec<T>,
    pub start: usize,
}

impl<T: Default + Clone> CircleBuffer<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            start: 0,
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.data.resize(size, T::default());
    }

    pub fn push_front(&mut self) {
        if self.data.is_empty() {
            return;
        }
        if self.start == 0 {
            self.start = self.data.len() - 1;
        } else {
            self.start -= 1;
        }
    }

    pub fn front(&self) -> &T {
        &self.data[self.start]
    }

    pub fn front_mut(&mut self) -> &mut T {
        &mut self.data[self.start]
    }

    pub fn back(&self) -> &T {
        let idx = (self.start + self.data.len() - 1) % self.data.len();
        &self.data[idx]
    }

    pub fn back_mut(&mut self) -> &mut T {
        let idx = (self.start + self.data.len() - 1) % self.data.len();
        &mut self.data[idx]
    }

    pub fn get(&self, index: usize) -> &T {
        &self.data[(self.start + index) % self.data.len()]
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        let len = self.data.len();
        &mut self.data[(self.start + index) % len]
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Default + Clone> Default for CircleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

// --- Serialization traits ---

pub trait StreamWriter {
    fn write_bytes(&mut self, data: &[u8]);

    fn write_i32(&mut self, v: i32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_u8(&mut self, v: u8) {
        self.write_bytes(&[v]);
    }

    fn write_i8(&mut self, v: i8) {
        self.write_bytes(&[v as u8]);
    }

    fn write_i32_slice(&mut self, slice: &[i32]) {
        for &v in slice {
            self.write_i32(v);
        }
    }

    fn write_f32_slice(&mut self, slice: &[f32]) {
        for &v in slice {
            self.write_f32(v);
        }
    }

    fn write_u8_slice(&mut self, slice: &[u8]) {
        self.write_bytes(slice);
    }

    fn write_i8_slice(&mut self, slice: &[i8]) {
        // SAFETY: i8 and u8 have same representation
        let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) };
        self.write_bytes(bytes);
    }

    fn write_int3(&mut self, v: Int3) {
        self.write_i32(v.x);
        self.write_i32(v.y);
        self.write_i32(v.z);
    }
}

pub trait StreamReader {
    fn read_bytes(&mut self, buf: &mut [u8]);

    fn read_i32(&mut self) -> i32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        i32::from_le_bytes(buf)
    }

    fn read_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn read_f32(&mut self) -> f32 {
        let mut buf = [0u8; 4];
        self.read_bytes(&mut buf);
        f32::from_le_bytes(buf)
    }

    fn read_u8(&mut self) -> u8 {
        let mut buf = [0u8; 1];
        self.read_bytes(&mut buf);
        buf[0]
    }

    fn read_i8(&mut self) -> i8 {
        self.read_u8() as i8
    }

    fn read_i32_slice(&mut self, slice: &mut [i32]) {
        for v in slice.iter_mut() {
            *v = self.read_i32();
        }
    }

    fn read_f32_slice(&mut self, slice: &mut [f32]) {
        for v in slice.iter_mut() {
            *v = self.read_f32();
        }
    }

    fn read_u8_slice(&mut self, slice: &mut [u8]) {
        self.read_bytes(slice);
    }

    fn read_i8_slice(&mut self, slice: &mut [i8]) {
        // SAFETY: i8 and u8 have same representation
        let bytes = unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len()) };
        self.read_bytes(bytes);
    }

    fn read_int3(&mut self) -> Int3 {
        let x = self.read_i32();
        let y = self.read_i32();
        let z = self.read_i32();
        Int3::new(x, y, z)
    }
}

// --- Vec-based stream implementations ---

pub struct VecWriter {
    pub data: Vec<u8>,
}

impl VecWriter {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl Default for VecWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamWriter for VecWriter {
    fn write_bytes(&mut self, data: &[u8]) {
        self.data.extend_from_slice(data);
    }
}

pub struct SliceReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> SliceReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl<'a> StreamReader for SliceReader<'a> {
    fn read_bytes(&mut self, buf: &mut [u8]) {
        let end = self.pos + buf.len();
        buf.copy_from_slice(&self.data[self.pos..end]);
        self.pos = end;
    }
}

// --- Thread pool size ---

static NUM_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn set_num_threads(n: usize) {
    NUM_THREADS.store(n, Ordering::Relaxed);
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .ok();
}

pub fn get_num_threads() -> usize {
    NUM_THREADS.load(Ordering::Relaxed)
}
