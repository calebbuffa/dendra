#[allow(dead_code)]
pub(crate) struct SizeOf {
    pub bytes: usize,
    pub kilobytes: usize,
    pub megabytes: usize,
    pub gigabytes: usize,
}

pub(crate) fn size_of_vec<T>(vector: &[T]) -> SizeOf {
    let bytes = vector.len() * std::mem::size_of::<T>();
    SizeOf {
        bytes,
        kilobytes: bytes / 1024,
        megabytes: bytes / (1024 * 1024),
        gigabytes: bytes / (1024 * 1024 * 1024),
    }
}
