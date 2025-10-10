use tfhe::{set_server_key, FheUint16, FheUint32, ServerKey, CudaServerKey};
use tfhe::prelude::*;

pub fn fhe_max16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16{
    set_server_key(server_keys.clone());
    encrypted_a.max(&encrypted_b)
}

pub fn fhe_max32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    encrypted_a.max(&encrypted_b)
}

pub fn fhe_max32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    server_keys: ServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    encrypted_a.max(&encrypted_b)
}