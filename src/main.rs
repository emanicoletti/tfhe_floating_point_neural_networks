use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::{set_server_key};
use tfhe::{ConfigBuilder, ClientKey, CompressedServerKey, CudaServerKey, FheUint16};
use std::time::Instant;
use rand::Rng;
use half::f16;
use tfhe::prelude::*;

mod encrypted_layers;
mod plain_layers;
mod encrypted_ops;
mod encrypted_utils;
use crate::encrypted_layers::EncryptedDenseLayer;
use crate::plain_layers::PlainDenseLayer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // 1. Generate keys
    let config = ConfigBuilder::with_custom_parameters(
        PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64,
    )
    .build();

    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let cuda_server_key = compressed_server_key.decompress_to_gpu();


    set_server_key(cuda_server_key.clone());

    let mut rng = rand::thread_rng();

    let mut total_duration = std::time::Duration::new(0, 0);

    set_server_key(cuda_server_key.clone());

    for _ in 0..50{

        let float_a: f32 = rng.gen_range(-5.0..5.0);
        let float_b: f32 = rng.gen_range(-5.0..5.0);
        let float_c: f32 = rng.gen_range(-5.0..5.0);

        let float_a_f16 = f16::from_f32(float_a);
        let float_b_f16 = f16::from_f32(float_b);
        let float_c_f16= f16::from_f32(float_c);

        // Convert f16 to u16 (bit pattern)
        let clear_a: u16 = float_a_f16.to_bits();
        let clear_b: u16 = float_b_f16.to_bits();
        let clear_c: u16 = float_c_f16.to_bits();

        // Encrypting the input data using the (private) client_key
        let encrypted_a = FheUint16::try_encrypt(clear_a, &client_key)?;
        let encrypted_b = FheUint16::try_encrypt(clear_b, &client_key)?; 
        let encrypted_c = FheUint16::try_encrypt(clear_c, &client_key)?;    
        let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key)?;
        let encrypted_1024 = FheUint16::try_encrypt(1023u16, &client_key);

        let start = Instant::now();

        //let encrypted_multiply = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
        let encrypted_accumulate = fhe_ldiv16_gpu(encrypted_a, encrypted_b, encrypted_zero.clone(), cuda_server_key.clone());
        
        //let encrypted_res = fhe_add(encrypted_multiply, encrypted_c, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_add_same_sign(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone(), client_key.clone());
        //let encrypted_res = fhe_pam(encrypted_a, encrypted_b, client_key.clone());

        // Add the execution time to the total
        total_duration += start.elapsed();

        let clear_res: u16 = encrypted_accumulate.decrypt(&client_key);

        let float_res = f16::from_bits(clear_res).to_f32();

        println!("({:?} / {:?}) = {:?}, Real: {:?}", float_a, float_b, float_res, float_a/float_b);

    }
    let mean_duration = total_duration / 50;
    println!("Mean execution time for 50 multiplications: {:?}", mean_duration);

    Ok(())
}

pub fn fhe_ldiv16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Prepare mutable vars for results
    let mut result_sign = None;
    let mut denorm = None;
    let mut result_digits = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 15u8;
            let y_sign = &encrypted_b >> 15u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 15u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_exp = (&encrypted_a & 31744u16) >> 10u8;
            let y_exp = (&encrypted_b & 31744u16) >> 10u8;
            let mut exp = &x_exp - &y_exp;
            exp += 15u16;
            let d = exp.gt(31u16);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 15296u16;
            digits &= 32767u16;
            result_digits = Some(digits);
        });
    });

    // Unwrap results (safe because scope waits for threads)
    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    // Final processing as before
    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}