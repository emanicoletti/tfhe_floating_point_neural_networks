use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

use crate::encrypted_ops::add::*;
use crate::encrypted_ops::mul::*;

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

pub fn fhe_tanh16_gpu(
    input: FheUint16,
    server_keys: CudaServerKey,
    encrypted_zero: FheUint16,
    encrypted_mask: FheUint16,
    ranges: &[(FheUint16, FheUint16, FheUint16, FheUint16, FheUint16)],
) -> (FheUint16, FheUint16) {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Collect partial outputs and derivatives separately
    let (results, derivatives): (Vec<FheUint16>, Vec<FheUint16>) = ranges
        .par_iter()
        .map(|(min, max, a, b, derivative)| {
            let gt = input.ge(min.clone());
            let lt = input.le(max.clone());
            let in_range = gt & lt;

            let mul = fhe_lmul16_gpu(input.clone(), a.clone(), encrypted_zero.clone(), server_keys.clone());
            let add = fhe_add16_gpu(mul, b.clone(), encrypted_mask.clone(), server_keys.clone());
            let out_result = in_range.select(&add, &encrypted_zero.clone());
            let out_derivative = in_range.select(&derivative.clone(), &encrypted_zero.clone());

            (out_result, out_derivative)
        })
        .unzip();

    let result = results.into_iter()
    .reduce(|acc, x| &acc | &x)
    .unwrap_or_else(|| encrypted_zero.clone());

    let derivative = derivatives.into_iter()
        .reduce(|acc, x| &acc | &x)
        .unwrap_or(encrypted_zero);

    (result, derivative)
}

pub fn fhe_tanh32_gpu(
    input: FheUint32,
    server_keys: CudaServerKey,
    encrypted_zero: FheUint32,
    encrypted_mask: FheUint32,
    ranges: &[(FheUint32, FheUint32, FheUint32, FheUint32, FheUint32)],
) -> (FheUint32, FheUint32) {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Collect partial outputs and derivatives separately
    let (results, derivatives): (Vec<FheUint32>, Vec<FheUint32>) = ranges
        .par_iter()
        .map(|(min, max, a, b, derivative)| {
            let gt = input.ge(min.clone());
            let lt = input.lt(max.clone());
            let in_range = gt & lt;

            let mul = fhe_lmul32_gpu(input.clone(), a.clone(), encrypted_zero.clone(), server_keys.clone());
            let add = fhe_add32_gpu(mul, b.clone(), encrypted_mask.clone(), server_keys.clone());
            let out_result = in_range.select(&add, &encrypted_zero.clone());
            let out_derivative = in_range.select(&derivative.clone(), &encrypted_zero.clone());

            (out_result, out_derivative)
        })
        .unzip();

    let result = results.into_par_iter()
        .reduce(|| encrypted_zero.clone(), |acc, x| &acc | &x);

    let derivative = derivatives.into_par_iter()
        .reduce(|| encrypted_zero.clone(), |acc, x| &acc | &x);

    (result, derivative)
}

pub fn fhe_tanh16_cpu(
    input: FheUint16,
    server_keys: ServerKey,
    encrypted_zero: FheUint16,
    encrypted_mask: FheUint16,
    ranges: &[(FheUint16, FheUint16, FheUint16, FheUint16, FheUint16)],
) -> (FheUint16, FheUint16) {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Collect partial outputs and derivatives separately
    let (results, derivatives): (Vec<FheUint16>, Vec<FheUint16>) = ranges
        .par_iter()
        .map(|(min, max, a, b, derivative)| {
            let gt = input.ge(min.clone());
            let lt = input.lt(max.clone());
            let in_range = gt & lt;

            let mul = fhe_lmul16_cpu(input.clone(), a.clone(), encrypted_zero.clone(), server_keys.clone());
            let add = fhe_add16_cpu(mul, b.clone(), encrypted_mask.clone(), server_keys.clone());
            let out_result = in_range.select(&add, &encrypted_zero.clone());
            let out_derivative = in_range.select(&derivative.clone(), &encrypted_zero.clone());

            (out_result, out_derivative)
        })
        .unzip();

    let result = results.into_iter()
        .reduce(|acc, x| acc | &x)
        .unwrap_or(encrypted_zero.clone());

    let derivative = derivatives.into_iter()
        .reduce(|acc, x| acc | &x)
        .unwrap_or(encrypted_zero);

    (result, derivative)
}

pub fn fhe_tanh32_cpu(
    input: FheUint32,
    server_keys: ServerKey,
    encrypted_zero: FheUint32,
    encrypted_mask: FheUint32,
    ranges: &[(FheUint32, FheUint32, FheUint32, FheUint32, FheUint32)],
) -> (FheUint32, FheUint32) {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Collect partial outputs and derivatives separately
    let (results, derivatives): (Vec<FheUint32>, Vec<FheUint32>) = ranges
        .par_iter()
        .map(|(min, max, a, b, derivative)| {
            let gt = input.ge(min.clone());
            let lt = input.lt(max.clone());
            let in_range = gt & lt;

            let mul = fhe_lmul32_cpu(input.clone(), a.clone(), encrypted_zero.clone(), server_keys.clone());
            let add = fhe_add32_cpu(mul, b.clone(), encrypted_mask.clone(), server_keys.clone());
            let out_result = in_range.select(&add, &encrypted_zero.clone());
            let out_derivative = in_range.select(&derivative.clone(), &encrypted_zero.clone());

            (out_result, out_derivative)
        })
        .unzip();

    let result = results.into_iter()
        .reduce(|acc, x| acc | &x)
        .unwrap_or(encrypted_zero.clone());

    let derivative = derivatives.into_iter()
        .reduce(|acc, x| acc | &x)
        .unwrap_or(encrypted_zero);

    (result, derivative)
}
