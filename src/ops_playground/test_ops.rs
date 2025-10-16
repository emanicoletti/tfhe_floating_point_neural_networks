// Ops Playground List:
// [add]: Addition - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Medium
// [same_sign_add]: Same Sign Addition - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Fast
// [sub]: Subtraction - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Medium
// [lmul]: Lmul Multiplication - Approximate - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Fast
// [ldiv]: Lmul Division - Approximate - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Fast
// [lmul_tanh]: PLA Tanh based on Lmul - Approximate - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [pam_mul]: PAM Multiplication - Approximate - Available fp formats: (FP16, FP32) - Execution: Fast
// [pam_div]: PAM Division - Approximate - Available fp formats: (FP16, FP32) - Execution: Fast
// [pam_tanh]: PLA Tanh based on PAM - Approximate - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [relu]: ReLU - Exact - Available fp formats: (FP8, FP16, FP32, FP64) - Execution: Very Fast
// [sqrt]: Square Root - Exact - Available fp formats: (FP16, FP32) - Execution: Very Slow
// [log2]: Base-2 Logarithm - Exact - Available fp formats: (FP16, FP32) - Execution: Extremely Slow

use tfhe::prelude::*;
use tfhe::{set_server_key, generate_keys, ConfigBuilder, FheUint8, FheUint16, FheUint32, FheUint64, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use std::time::Instant;
use rand::Rng;
use half::f16;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use tfhe::shortint::parameters::v1_3::{V1_3_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};
use crate::tfhe_nn_builder::add::*;
use crate::tfhe_nn_builder::div::*;
use crate::tfhe_nn_builder::log2::*;
use crate::tfhe_nn_builder::mul::*;
use crate::tfhe_nn_builder::negate::*;
use crate::tfhe_nn_builder::relu::*;
use crate::tfhe_nn_builder::same_sign_add::*;
use crate::tfhe_nn_builder::sqrt::*;
use crate::tfhe_nn_builder::tanh::*;

#[allow(dead_code)]
/// PLA TANH Ranges for FP16
pub static TANH16_PLA_RANGES: &[(u16, u16, u16, u16, u16)] = &[
    (49152u16, 65535u16, 0u16, 48128u16, 0u16), // [-inf, -2], output ~ -1, derivative ≈ 0
    (47787u16, 49151u16, 13312u16, 47104u16, 13312u16), // [-2, -0.8333] slope 0.25. intercept -0.5
    (32768u16, 47786u16, 15053u16, 0u16, 15053u16), // [-0.833, -0] slope=0.85 intercept = 0
    (0u16, 15019u16, 15053u16, 0u16, 15053u16), //[0, 0.833] slope=0.85 intercept = 0
    (15020u16, 16384u16, 13312u16, 14336u16, 13312u16), //[0.833, 2] slope = 0.25 intercept 0.5
    (16385u16, 32767u16, 0u16, 15360u16, 0u16), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

#[allow(dead_code)]
/// PLA TANH Ranges for FP32
pub static TANH32_PLA_RANGES: &[(u32, u32, u32, u32, u32)] = &[
    (3221225472u32, 4294967295u32, 0u32, 3212836864u32, 0u32), // [-inf, -2], output ~ -1, derivative ≈ 0
    (3210040661u32, 3221225471u32, 1048576000u32, 3204448256u32, 1048576000u32), // [-2, -0.8333] slope 0.25. intercept -0.5
    (2147483648u32, 3210040660u32, 1062836634u32, 0u32, 1062836634u32), // [-0.833, -0] slope=0.85 intercept = 0
    (0u32, 1062557013u32, 1062836634u32, 0u32, 1062836634u32), //[0, 0.833] slope=0.85 intercept = 0
    (1062557014u32, 1073741824u32, 1048576000u32, 1056964608u32, 1048576000u32), //[0.833, 2] slope = 0.25 intercept 0.5
    (1073741825u32, 2147483647u32, 0u32, 1065353217u32, 0u32), // [2.0, +inf], output ~ 1, derivative ≈ 0
];

#[allow(dead_code)]
pub fn test_encrypted_ops(ops: &str, fp_size: usize, gpu: bool, num_ops: usize, min_range: f32, max_range: f32) -> Result<(), Box<dyn std::error::Error>> {
    if gpu {
        // Configure, generate and set the keys for GPU execution
        let config = ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64).build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let server_key = compressed_server_key.decompress_to_gpu();
        set_server_key(server_key.clone());
        rayon::broadcast(|_| set_server_key(server_key.clone()));
        gpu_test(ops, fp_size, num_ops, min_range, max_range, client_key, server_key)
    }
    else{
        // Configure, generate and set the keys for CPU execution
        let config = ConfigBuilder::with_custom_parameters(V1_3_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
        let (client_key, server_key) = generate_keys(config);
        rayon::broadcast(|_| set_server_key(server_key.clone()));
        cpu_test(ops, fp_size, num_ops, min_range, max_range, client_key, server_key)
    }
}

#[allow(dead_code)]
/// GPU Test Function
fn gpu_test(ops: &str, fp_size: usize, num_ops: usize, min_range: f32, max_range: f32, client_key: ClientKey, server_key: CudaServerKey) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("[{}]", ops);
    set_server_key(server_key.clone());

    let mut ops_duration = std::time::Duration::new(0, 0);
    let mut rng = rand::thread_rng();
    let mut ranges_16= vec![];
    let mut ranges_32= vec![];

    // Encrypt the PLA TANH ranges if needed
    if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") {
        if fp_size == 16 {
            ranges_16 = TANH16_PLA_RANGES
                .iter()
                .map(|&(a, b, c, d, e)| {
                    (
                        FheUint16::try_encrypt(a, &client_key).unwrap(),
                        FheUint16::try_encrypt(b, &client_key).unwrap(),
                        FheUint16::try_encrypt(c, &client_key).unwrap(),
                        FheUint16::try_encrypt(d, &client_key).unwrap(),
                        FheUint16::try_encrypt(e, &client_key).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
        }
        else if fp_size == 32 {
            ranges_32 = TANH32_PLA_RANGES
                .iter()
                .map(|&(a, b, c, d, e)| {
                    (
                        FheUint32::try_encrypt(a, &client_key).unwrap(),
                        FheUint32::try_encrypt(b, &client_key).unwrap(),
                        FheUint32::try_encrypt(c, &client_key).unwrap(),
                        FheUint32::try_encrypt(d, &client_key).unwrap(),
                        FheUint32::try_encrypt(e, &client_key).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
        }
    }

    // Execute the operations
    for _ in 0..num_ops {

        // Generate random input values
        let mut float_a: f32 = rng.gen_range(min_range..max_range);
        let mut float_b: f32 = rng.gen_range(min_range..max_range);

        // Ensure the inputs meet the operation's requirements
        if str::eq(ops, "same_sign_add"){
            while (float_a >= 0.0 && float_b < 0.0) || (float_a < 0.0 && float_b >= 0.0) {
                float_b = rng.gen_range(min_range..max_range);
            }
        }
        if str::eq(ops, "ldiv") || str::eq(ops, "pam_div") {
            while float_b == 0.0 {
                float_b = rng.gen_range(min_range..max_range);
            }
        }
        if str::eq(ops, "sqrt") || str::eq(ops, "log2") {
            while float_a <= 0.0 {
                float_a = rng.gen_range(min_range..max_range);
            }
        }

        // Match the floating point size
        match fp_size {
            8 => {
                // Limit the range for FP8 to [0, 255]
                let clear_a: u8 = rng.gen_range(0..=255);
                let clear_b: u8 = rng.gen_range(0..=255);

                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Uint A: {}", clear_a);
                }
                else {
                    println!("Uint A: {}, Uint B: {}", clear_a, clear_b);
                }

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint8::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint8::try_encrypt(clear_b, &client_key)?;   
                let encrypted_zero = FheUint8::try_encrypt(0u8, &client_key)?;
                let encrypted_mask = FheUint8::try_encrypt(255u8, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add8_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add8_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate8_gpu(encrypted_b, server_key.clone());
                        fhe_add8_gpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul8_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv8_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => panic!("Lmul Tanh not supported for FP8"),
                    "pam_mul" => fhe_pam_mul8_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div8_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" =>  panic!("Pam Tanh not supported for FP8"),
                    "relu" => panic!("ReLU not supported for FP8"),
                    "sqrt" => panic!("Square root not supported for FP8"),
                    "log2" => panic!("Log2 not supported for FP8"),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                let res: u8 = result.decrypt(&client_key);
                println!("Result: {} \n", res);
                ops_duration += duration;
            },
            16 => {
                // Convert the float inputs to f16
                let float_a_f16 = f16::from_f32(float_a);
                let float_b_f16 = f16::from_f32(float_b);

                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a_f16);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a_f16, float_b_f16);
                }

                // Convert to u16 representation
                let clear_a: u16 = float_a_f16.to_bits();
                let clear_b: u16 = float_b_f16.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint16::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint16::try_encrypt(clear_b, &client_key)?; 
                let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key)?;
                let encrypted_mask = FheUint16::try_encrypt(1023u16, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add16_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add16_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate16_gpu(encrypted_b, server_key.clone());
                        fhe_add16_gpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul16_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv16_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => fhe_lmul_tanh16_gpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_16).0.clone(),
                    "pam_mul" => fhe_pam_mul16_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div16_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => fhe_pam_tanh16_gpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_16).0.clone(),
                    "relu" => fhe_relu16_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "sqrt" => fhe_sqrt16_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "log2" => fhe_log2_16_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f16::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            32 => {
                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a, float_b);
                }

                // Convert to u32 representation
                let clear_a: u32 = float_a.to_bits();
                let clear_b: u32 = float_b.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint32::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint32::try_encrypt(clear_b, &client_key)?; 
                let encrypted_zero = FheUint32::try_encrypt(0u32, &client_key)?;
                let encrypted_mask = FheUint32::try_encrypt(8388607u32, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add32_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add32_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate32_gpu(encrypted_b, server_key.clone());
                        fhe_add32_gpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul32_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv32_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => fhe_lmul_tanh32_gpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_32).0.clone(),
                    "pam_mul" => fhe_pam_mul32_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div32_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => fhe_pam_tanh32_gpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_32).0.clone(),
                    "relu" => fhe_relu32_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "sqrt" => fhe_sqrt32_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "log2" => fhe_log2_32_gpu(encrypted_a, encrypted_zero, server_key.clone()),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f32::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            64 => {
                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a, float_b);
                }

                // Convert to u64 representation
                let float_a_f64 = float_a as f64;
                let float_b_f64 = float_b as f64;
                let clear_a: u64 = float_a_f64.to_bits();
                let clear_b: u64 = float_b_f64.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint64::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint64::try_encrypt(clear_b, &client_key)?;
                let encrypted_zero = FheUint64::try_encrypt(0u64, &client_key)?;
                let encrypted_mask = FheUint64::try_encrypt(0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add64_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add64_gpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate64_gpu(encrypted_b, server_key.clone());
                        fhe_add64_gpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul64_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv64_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => panic!("Lmul Tanh not supported for fp64"),
                    "pam_mul" => fhe_pam_mul64_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div64_gpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => panic!("PAM Tanh not supported for fp64"),
                    "relu" => panic!("ReLU not supported for fp64"),
                    "sqrt" => panic!("Sqrt not supported for fp64"),
                    "log2" => panic!("Log2 not supported for fp64"),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f64::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            _ => panic!("Unsupported floating point size: {}", fp_size),
        }
    }
    println!("Average time for {} with fp{} over {} operations: {:?}", ops, fp_size, num_ops, ops_duration / num_ops as u32);
    Ok(())
}

#[allow(dead_code)]
/// CPU Test Function
fn cpu_test(ops: &str, fp_size: usize, num_ops: usize, min_range: f32, max_range: f32, client_key: ClientKey, server_key: ServerKey) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("[{}]", ops);
    set_server_key(server_key.clone());

    let mut ops_duration = std::time::Duration::new(0, 0);
    let mut rng = rand::thread_rng();
    let mut ranges_16= vec![];
    let mut ranges_32= vec![];

    // Encrypt the PLA TANH ranges if needed
    if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") {
        if fp_size == 16 {
            ranges_16 = TANH16_PLA_RANGES
                .iter()
                .map(|&(a, b, c, d, e)| {
                    (
                        FheUint16::try_encrypt(a, &client_key).unwrap(),
                        FheUint16::try_encrypt(b, &client_key).unwrap(),
                        FheUint16::try_encrypt(c, &client_key).unwrap(),
                        FheUint16::try_encrypt(d, &client_key).unwrap(),
                        FheUint16::try_encrypt(e, &client_key).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
        }
        else if fp_size == 32 {
            ranges_32 = TANH32_PLA_RANGES
                .iter()
                .map(|&(a, b, c, d, e)| {
                    (
                        FheUint32::try_encrypt(a, &client_key).unwrap(),
                        FheUint32::try_encrypt(b, &client_key).unwrap(),
                        FheUint32::try_encrypt(c, &client_key).unwrap(),
                        FheUint32::try_encrypt(d, &client_key).unwrap(),
                        FheUint32::try_encrypt(e, &client_key).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
        }
    }
    // Execute the operations
    for _ in 0..num_ops {
        // Generate random input values
        let mut float_a: f32 = rng.gen_range(min_range..max_range);
        let mut float_b: f32 = rng.gen_range(min_range..max_range);

        // Ensure the inputs meet the operation's requirements
        if str::eq(ops, "same_sign_add"){
            while (float_a >= 0.0 && float_b < 0.0) || (float_a < 0.0 && float_b >= 0.0) {
                float_b = rng.gen_range(min_range..max_range);
            }
        }
        if str::eq(ops, "ldiv") || str::eq(ops, "pam_div") {
            while float_b == 0.0 {
                float_b = rng.gen_range(min_range..max_range);
            }
        }
        if str::eq(ops, "sqrt") || str::eq(ops, "log2") {
            while float_a <= 0.0 {
                float_a = rng.gen_range(min_range..max_range);
            }
        }
        match fp_size {
            8 => {
                // Limit the range for FP8 to [0, 255]
                let clear_a: u8 = rng.gen_range(0..=255);
                let clear_b: u8 = rng.gen_range(0..=255);

                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Uint A: {}", clear_a);
                }
                else {
                    println!("Uint A: {}, Uint B: {}", clear_a, clear_b);
                }

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint8::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint8::try_encrypt(clear_b, &client_key)?;   
                let encrypted_zero = FheUint8::try_encrypt(0u8, &client_key)?;
                let encrypted_mask = FheUint8::try_encrypt(255u8, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add8_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add8_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate8_cpu(encrypted_b, server_key.clone());
                        fhe_add8_cpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul8_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv8_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => panic!("Lmul Tanh not supported for FP8"),
                    "pam_mul" => fhe_pam_mul8_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div8_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" =>  panic!("Pam Tanh not supported for FP8"),
                    "relu" => panic!("ReLU not supported for FP8"),
                    "sqrt" => panic!("Square root not supported for FP8"),
                    "log2" => panic!("Log2 not supported for FP8"),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                let res: u8 = result.decrypt(&client_key);
                println!("Result: {} \n", res);
                ops_duration += duration;
            },
            16 => {
                // Convert the float inputs to f16
                let float_a_f16 = f16::from_f32(float_a);
                let float_b_f16 = f16::from_f32(float_b);

                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a_f16);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a_f16, float_b_f16);
                }

                // Convert to u16 representation
                let clear_a: u16 = float_a_f16.to_bits();
                let clear_b: u16 = float_b_f16.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint16::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint16::try_encrypt(clear_b, &client_key)?; 
                let encrypted_zero = FheUint16::try_encrypt(0u16, &client_key)?;
                let encrypted_mask = FheUint16::try_encrypt(1023u16, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add16_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add16_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate16_cpu(encrypted_b, server_key.clone());
                        fhe_add16_cpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul16_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv16_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => fhe_lmul_tanh16_cpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_16).0.clone(),
                    "pam_mul" => fhe_pam_mul16_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div16_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => fhe_pam_tanh16_cpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_16).0.clone(),
                    "relu" => fhe_relu16_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "sqrt" => fhe_sqrt16_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "log2" => fhe_log2_16_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f16::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            32 => {
                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a, float_b);
                }

                // Convert to u32 representation
                let clear_a: u32 = float_a.to_bits();
                let clear_b: u32 = float_b.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint32::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint32::try_encrypt(clear_b, &client_key)?; 
                let encrypted_zero = FheUint32::try_encrypt(0u32, &client_key)?;
                let encrypted_mask = FheUint32::try_encrypt(8388607u32, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add32_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add32_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate32_cpu(encrypted_b, server_key.clone());
                        fhe_add32_cpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul32_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv32_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => fhe_lmul_tanh32_cpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_32).0.clone(),
                    "pam_mul" => fhe_pam_mul32_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div32_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => fhe_pam_tanh32_cpu(encrypted_a, server_key.clone(), encrypted_zero, encrypted_mask, &ranges_32).0.clone(),
                    "relu" => fhe_relu32_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "sqrt" => fhe_sqrt32_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    "log2" => fhe_log2_32_cpu(encrypted_a, encrypted_zero, server_key.clone()),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f32::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            64 => {
                if str::eq(ops, "lmul_tanh") || str::eq(ops, "pam_tanh") || str::eq(ops, "sqrt") || str::eq(ops, "log2") || str::eq(ops, "relu") {
                    println!("Float A: {}", float_a);
                }
                else {
                    println!("Float A: {}, Float B: {}", float_a, float_b);
                }

                // Convert to u64 representation
                let float_a_f64 = float_a as f64;
                let float_b_f64 = float_b as f64;
                let clear_a: u64 = float_a_f64.to_bits();
                let clear_b: u64 = float_b_f64.to_bits();

                // Encrypt the input data using the (private) client_key
                let encrypted_a = FheUint64::try_encrypt(clear_a, &client_key)?;
                let encrypted_b = FheUint64::try_encrypt(clear_b, &client_key)?;
                let encrypted_zero = FheUint64::try_encrypt(0u64, &client_key)?;
                let encrypted_mask = FheUint64::try_encrypt(0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64, &client_key)?;

                // Perform the operation
                let start = Instant::now();
                let result = match ops {
                    "add" => fhe_add64_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "same_sign_add" => fhe_ss_add64_cpu(encrypted_a, encrypted_b, encrypted_mask, encrypted_zero, server_key.clone()),
                    "sub" => {
                        let encrypted_b_negate = fhe_negate64_cpu(encrypted_b, server_key.clone());
                        fhe_add64_cpu(encrypted_a, encrypted_b_negate, encrypted_mask, encrypted_zero, server_key.clone())
                    },
                    "lmul" => fhe_lmul64_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "ldiv" => fhe_ldiv64_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "lmul_tanh" => panic!("Lmul Tanh not supported for fp64"),
                    "pam_mul" => fhe_pam_mul64_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_div" => fhe_pam_div64_cpu(encrypted_a, encrypted_b, encrypted_zero, server_key.clone()),
                    "pam_tanh" => panic!("PAM Tanh not supported for fp64"),
                    "relu" => panic!("ReLU not supported for fp64"),
                    "sqrt" => panic!("Sqrt not supported for fp64"),
                    "log2" => panic!("Log2 not supported for fp64"),
                    _ => panic!("Unsupported operation: {}", ops),
                };
                let duration = start.elapsed();
                println!("Result: {} \n", f64::from_bits(result.decrypt(&client_key)));
                ops_duration += duration;
            },
            _ => panic!("Unsupported floating point size: {}", fp_size),
        }
    }
    println!("Average time for {} with fp{} over {} operations: {:?}", ops, fp_size, num_ops, ops_duration / num_ops as u32);
    Ok(())
}
