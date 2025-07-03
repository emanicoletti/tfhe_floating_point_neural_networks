use tfhe::prelude::*;
use tfhe::{set_server_key, ConfigBuilder, FheUint8, FheUint16, FheUint32, FheUint64, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use rayon::prelude::*;
use rayon::{join, scope};

/* Gpu operations */

/* Gpu-oriented sign flipping for 8 bits floating points */
pub fn fhe_negate8_gpu(
    encrypted_a: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000u8
}

/* Gpu-oriented sign flipping for 16 bits floating points */
pub fn fhe_negate16_gpu(
    encrypted_a: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000u16
}

/* Gpu-oriented sign flipping for 32 bits floating points */
pub fn fhe_negate32_gpu(
    encrypted_a: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000_0000_0000_0000_0000u32
}

/* Gpu-oriented sign flipping for 64 bits floating points */
pub fn fhe_negate64_gpu(
    encrypted_a: FheUint64,
    server_keys: CudaServerKey,
) -> FheUint64{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64
}

/* Cpu-oriented sign flipping for 8 bits floating points */
pub fn fhe_negate8_cpu(
    encrypted_a: FheUint8,
    server_keys: ServerKey,
) -> FheUint8{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000u8
}

/* Cpu-oriented sign flipping for 16 bits floating points */
pub fn fhe_negate16_cpu(
    encrypted_a: FheUint16,
    server_keys: ServerKey,
) -> FheUint16{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000u16
}

/* Cpu-oriented sign flipping for 32 bits floating points */
pub fn fhe_negate32_cpu(
    encrypted_a: FheUint32,
    server_keys: ServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000_0000_0000_0000_0000u32
}

/* Cpu-oriented sign flipping for 64 bits floating points */
pub fn fhe_negate64_cpu(
    encrypted_a: FheUint64,
    server_keys: ServerKey,
) -> FheUint64{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64
}
