use tfhe::{prelude::*, set_server_key, CudaServerKey, FheBool, FheUint16, FheUint32};

pub fn fhe_relu16_gpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    let neg = encrypted_a.ge(32768u16);
    neg.select(
        &encrypted_zero.clone(),
        &encrypted_a.clone(),
    )
}

pub fn fhe_relu32_gpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    let neg = encrypted_a.ge(2147483648u32);
    neg.select(
        &encrypted_zero,
        &encrypted_a.clone(),
    )
}

pub fn backward_relu16_gpu(
    encrypted_grad_output: FheUint16,
    encrypted_derivative: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    let pos = encrypted_derivative.eq(0u16);
    pos.select(
        &encrypted_zero,
        &encrypted_grad_output,
    )
}

pub fn backward_relu32_gpu(
    encrypted_grad_output: FheUint32,
    encrypted_derivative: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    let pos = encrypted_derivative.eq(0u32);
    pos.select(
        &encrypted_zero,
        &encrypted_grad_output,
    )
}

