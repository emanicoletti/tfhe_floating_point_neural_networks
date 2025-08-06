use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};
use tfhe::prelude::*;

pub fn fhe_grad_if_equal16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    encrypted_grad: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16{
    set_server_key(server_keys.clone());
    let equal = encrypted_a.eq(&encrypted_b);
    equal.select(&encrypted_grad, &encrypted_zero)
}

pub fn fhe_grad_if_equal32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    encrypted_grad: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    let equal = encrypted_a.eq(&encrypted_b);
    equal.select(&encrypted_grad, &encrypted_zero)
}

pub fn fhe_grad_if_equal32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    encrypted_grad: FheUint32,
    server_keys: ServerKey,
) -> FheUint32{
    set_server_key(server_keys.clone());
    let equal = encrypted_a.eq(&encrypted_b);
    equal.select(&encrypted_grad, &encrypted_zero)
}