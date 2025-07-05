use burn::serde::Serialize;
use tfhe::core_crypto::gpu;
use tfhe::{prelude::*, HlCompressible};
use tfhe::shortint::client_key;
use tfhe::{set_server_key, generate_keys, ConfigBuilder, FheUint8, FheUint16, FheUint32, FheUint64, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use std::time::Instant;
use rand::Rng;
use half::f16;
use rayon::prelude::*;
use rayon::{join, scope};
use std::thread;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};

/* 
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CPU
    /* 
    let config =
        ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let (client_key, gpu_key) = generate_keys(config);
    */
    
    // GPU
    let config =
        ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let gpu_key = compressed_server_key.decompress_to_gpu();
    
    let mut rng = rand::thread_rng();

    let mut total_duration = std::time::Duration::new(0, 0);

    set_server_key(gpu_key.clone());

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
        let encrypted_1024 = FheUint16::encrypt(1023u16, &client_key);

        let start = Instant::now();

        //let encrypted_multiply = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
        let encrypted_accumulate = fhe_add(encrypted_a, encrypted_b, encrypted_zero.clone(), encrypted_1024.clone(), gpu_key.clone());
        
        //let encrypted_res = fhe_add(encrypted_multiply, encrypted_c, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_add_same_sign(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone(), client_key.clone());
        //let encrypted_res = fhe_pam(encrypted_a, encrypted_b, client_key.clone());

        // Add the execution time to the total
        total_duration += start.elapsed();

        let clear_res: u16 = encrypted_accumulate.decrypt(&client_key);

        let float_res = f16::from_bits(clear_res).to_f32();

        println!("({:?} x {:?}) + {:?} = {:?}", float_a, float_b, float_c, float_res);

    }
    let mean_duration = total_duration / 50;
    println!("Mean execution time for 50 multiplications: {:?}", mean_duration);

    Ok(())
}
*/
/*
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CPU
    /* 
    let config =
        ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let (client_key, gpu_key) = generate_keys(config);
    */
    
    
    // GPU
    let config =
        ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let gpu_key = compressed_server_key.decompress_to_gpu();
    
    
    let mut rng = rand::thread_rng();

    let mut total_duration = std::time::Duration::new(0, 0);

    set_server_key(gpu_key.clone());

    for _ in 0..50{

        let float_a: f32 = rng.gen_range(-5.0..5.0);
        let float_b: f32 = rng.gen_range(-5.0..5.0);
        let float_c: f32 = rng.gen_range(-5.0..5.0);

        // Convert f16 to u16 (bit pattern)
        let clear_a: u32 = float_a.to_bits();
        let clear_b: u32 = float_b.to_bits();
        let clear_c: u32 = float_c.to_bits();

        // Encrypting the input data using the (private) client_key
        let encrypted_a = FheUint32::try_encrypt(clear_a, &client_key)?;
        let encrypted_b = FheUint32::try_encrypt(clear_b, &client_key)?; 
        let encrypted_c = FheUint32::try_encrypt(clear_c, &client_key)?;    
        let encrypted_zero = FheUint32::try_encrypt(0u32, &client_key)?;
        let encrypted_1023 = FheUint32::encrypt(8388607u32, &client_key);

        let start = Instant::now();

        //let encrypted_multiply = fhe_lmul32_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
        let encrypted_accumulate = fhe_add_32(encrypted_a, encrypted_b, encrypted_zero.clone(), encrypted_1023.clone(), gpu_key.clone());
        
        //let encrypted_res = fhe_add(encrypted_multiply, encrypted_c, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_add_same_sign(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone(), client_key.clone());
        //let encrypted_res = fhe_pam(encrypted_a, encrypted_b, client_key.clone());

        // Add the execution time to the total
        total_duration += start.elapsed();

        let clear_res: u32 = encrypted_accumulate.decrypt(&client_key);

        let float_res = f32::from_bits(clear_res);

        println!("({:?} x {:?}) = {:?}", float_a, float_b, float_res);

    }
    let mean_duration = total_duration / 50;
    println!("Mean execution time for 50 multiplications: {:?}", mean_duration);

    Ok(())
}
*/

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CPU

    let config =
        ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let (client_key, gpu_key) = generate_keys(config);

    // GPU
    /* 
    let config =
        ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
            .build();
    let client_key = ClientKey::generate(config);
    let compressed_server_key = CompressedServerKey::new(&client_key);
    let gpu_key = compressed_server_key.decompress_to_gpu();
    */
    
    let mut rng = rand::thread_rng();

    let mut total_duration = std::time::Duration::new(0, 0);

    set_server_key(gpu_key.clone());

    for _ in 0..50{

        let float_a: f64 = rng.gen_range(-5.0..5.0);
        let float_b: f64 = rng.gen_range(-5.0..5.0);
        let float_c: f64 = rng.gen_range(-5.0..5.0);

        // Convert f16 to u16 (bit pattern)
        let clear_a: u64 = float_a.to_bits();
        let clear_b: u64 = float_b.to_bits();
        let clear_c: u64 = float_c.to_bits();

        // Encrypting the input data using the (private) client_key
        let encrypted_a = FheUint64::try_encrypt(clear_a, &client_key)?;
        let encrypted_b = FheUint64::try_encrypt(clear_b, &client_key)?; 
        let encrypted_c = FheUint64::try_encrypt(clear_c, &client_key)?;    
        let encrypted_zero = FheUint64::try_encrypt(0u64, &client_key)?;
        let encrypted_1024 = FheUint64::encrypt(0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64, &client_key);

        let start = Instant::now();

        //let encrypted_multiply = fhe_lmul64_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
        let encrypted_accumulate = fhe_add_64(encrypted_a, encrypted_b, encrypted_zero.clone(), encrypted_1024.clone(), gpu_key.clone());
        
        //let encrypted_res = fhe_add(encrypted_multiply, encrypted_c, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_add_same_sign(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone());
        //let encrypted_res = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone(), client_key.clone());
        //let encrypted_res = fhe_pam(encrypted_a, encrypted_b, client_key.clone());

        // Add the execution time to the total
        total_duration += start.elapsed();

        let clear_res: u64 = encrypted_accumulate.decrypt(&client_key);

        let float_res = f64::from_bits(clear_res);

        println!("({:?} x {:?}) = {:?}", float_a, float_b, float_res);

    }
    let mean_duration = total_duration / 50;
    println!("Mean execution time for 50 multiplications: {:?}", mean_duration);

    Ok(())
}
    
    /* 
    fn main() -> Result<(), Box<dyn std::error::Error>> {
        // CPU
        /* 
        let config =
            ConfigBuilder::with_custom_parameters(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
                .build();
        let (client_key, gpu_key) = generate_keys(config);
        */
        
        
        // GPU
        let config =
            ConfigBuilder::with_custom_parameters(PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64)
                .build();
        let client_key = ClientKey::generate(config);
        let compressed_server_key = CompressedServerKey::new(&client_key);
        let gpu_key = compressed_server_key.decompress_to_gpu();
        
        
        
        let mut rng = rand::thread_rng();
    
        let mut total_duration = std::time::Duration::new(0, 0);
    
        set_server_key(gpu_key.clone());
    
        for _ in 0..50{
            
            /* 
            let float_a: f16 = rng.gen_range(-5.0..5.0);
            let float_b: f64 = rng.gen_range(-5.0..5.0);
            let float_c: f64 = rng.gen_range(-5.0..5.0);
            */
    
            // Convert f16 to u16 (bit pattern)
            let clear_a: u8 = rng.gen_range(0..=255);
            let clear_b: u8 = rng.gen_range(0..=255);
            let clear_c: u8 = rng.gen_range(0..=255);
    
            // Encrypting the input data using the (private) client_key
            let encrypted_a = FheUint8::try_encrypt(clear_a, &client_key)?;
            let encrypted_b = FheUint8::try_encrypt(clear_b, &client_key)?; 
            let encrypted_c = FheUint8::try_encrypt(clear_c, &client_key)?;    
            let encrypted_zero = FheUint8::try_encrypt(0u8, &client_key)?;
            let encrypted_1024 = FheUint8::encrypt(7u8, &client_key);
    
            let start = Instant::now();
    
            //let encrypted_multiply = fhe_lmul8_parallel(encrypted_a, encrypted_b, encrypted_zero.clone(), gpu_key.clone());
            let encrypted_accumulate = fhe_add_8(encrypted_a, encrypted_b, encrypted_zero.clone(), encrypted_1024.clone(), gpu_key.clone());
            
            //let encrypted_res = fhe_add(encrypted_multiply, encrypted_c, encrypted_zero, gpu_key.clone());
            //let encrypted_res = fhe_add_same_sign(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone());
            //let encrypted_res = fhe_lmul16_parallel(encrypted_a, encrypted_b, encrypted_zero, gpu_key.clone(), client_key.clone());
            //let encrypted_res = fhe_pam(encrypted_a, encrypted_b, client_key.clone());
    
            // Add the execution time to the total
            total_duration += start.elapsed();
    
            let clear_res: u8 = encrypted_accumulate.decrypt(&client_key);
    
            //let float_res = f64::from_bits(clear_res);
    
            println!("({:?} x {:?}) = {:?}", clear_a, clear_b, clear_res);
    
        }
        let mean_duration = total_duration / 50;
        println!("Mean execution time for 50 multiplications: {:?}", mean_duration);
    
        Ok(())
    }
    */

pub fn fhe_add_16(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    encrypted_1023: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a << 1u8;
    let ns_b = &encrypted_b << 1u8;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_mant, y_mant), ((x_exp, diff_exp), (x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let x_mant = (&encrypted_x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            let y_mant = (&encrypted_y & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            (x_mant, y_mant)
        },
        || {
            // Thread 2 + 3: Nested join
            rayon::join(
                || {
                    // Thread 2: Exponents
                    let x_exp = &encrypted_x & 0b0111_1100_0000_0000u16;
                    let y_exp = &encrypted_y & 0b0111_1100_0000_0000u16;
                    let diff_exp = (&x_exp - &y_exp) >> 10u16;
                    let clipped_diff_exp = diff_exp.min(15u16);
                    (x_exp, clipped_diff_exp)
                },
                || {
                    // Thread 3: Signs
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000u16;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000u16;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_sign, same_sign)
                },
            )
        },
    );
    
    let (sum_mant, diff_mant) = rayon::join(
        ||{
            &x_mant + (&y_mant >> &diff_exp)
        },
        || {
            &x_mant - (&y_mant >> &diff_exp)
        }
    );
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint16::cast_from(op_mant.leading_zeros());

    let (ov_result, result) = rayon::join(
        ||{
            let mant = (&op_mant & 0b0000_0111_1111_1110u16) >> 1u16;
            let res_exp = &x_exp + 1024u16;
            let result = &x_sign | &res_exp | &mant;
            result
        },
        ||{
            let diff = (&leading_zeros - 5u16);
            let mask = &encrypted_1023 >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 10u16;
            let res_exp = &x_exp - &sub_exp;
            let result = &x_sign | &res_exp | &mant;
            result
        }
    );

    let overflow = leading_zeros.eq(4u16);
    overflow.select(&ov_result, &result)
}

pub fn fhe_add_32(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    encrypted_1023: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a << 1u8;
    let ns_b = &encrypted_b << 1u8;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_mant, y_mant), ((x_exp, diff_exp), (x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let x_mant = (&encrypted_x & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
            let y_mant = (&encrypted_y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
            (x_mant, y_mant)
        },
        || {
            // Thread 2 + 3: Nested join
            rayon::join(
                || {
                    // Thread 2: Exponents
                    let x_exp = &encrypted_x & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
                    let y_exp = &encrypted_y & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
                    let diff_exp = (&x_exp - &y_exp) >> 23u16;
                    let clipped_diff_exp = diff_exp.min(31u16);
                    (x_exp, clipped_diff_exp)
                },
                || {
                    // Thread 3: Signs
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_sign, same_sign)
                },
            )
        },
    );
    
    let (sum_mant, diff_mant) = rayon::join(
        ||{
            &x_mant + (&y_mant >> &diff_exp)
        },
        || {
            &x_mant - (&y_mant >> &diff_exp)
        }
    );
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = op_mant.leading_zeros();

    let (ov_result, result) = rayon::join(
        ||{
            let mant = (&op_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u16;
            let res_exp = &x_exp + 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
            let result = &x_sign | &res_exp | &mant;
            result
        },
        ||{
            let diff = (&leading_zeros - 8u32);
            let mask = &encrypted_1023 >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 23u16;
            let res_exp = &x_exp - &sub_exp;
            let result = &x_sign | &res_exp | &mant;
            result
        }
    );

    let overflow = &leading_zeros.eq(7u32);
    overflow.select(&ov_result, &result)
}

pub fn fhe_add_8(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    encrypted_1023: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a << 1u8;
    let ns_b = &encrypted_b << 1u8;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_mant, y_mant), ((x_exp, diff_exp), (x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let x_mant = (&encrypted_x & 0b0000_0111u8) | 0b0000_1000u8;
            let y_mant = (&encrypted_y & 0b0000_0111u8) | 0b0000_1000u8;
            (x_mant, y_mant)
        },
        || {
            // Thread 2 + 3: Nested join
            rayon::join(
                || {
                    // Thread 2: Exponents
                    let x_exp = &encrypted_x & 0b0111_1000u8;
                    let y_exp = &encrypted_y & 0b0111_1000u8;
                    let diff_exp = (&x_exp - &y_exp) >> 3u16;
                    let clipped_diff_exp = diff_exp.min(7u16);
                    (x_exp, clipped_diff_exp)
                },
                || {
                    // Thread 3: Signs
                    let x_sign = &encrypted_x & 0b1000_0000u8;
                    let y_sign = &encrypted_y & 0b1000_0000u8;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_sign, same_sign)
                },
            )
        },
    );
    
    let (sum_mant, diff_mant) = rayon::join(
        ||{
            &x_mant + (&y_mant >> &diff_exp)
        },
        || {
            &x_mant - (&y_mant >> &diff_exp)
        }
    );
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros_16: FheUint16 = FheUint16::cast_from(op_mant.leading_zeros());
    let leading_zeros: FheUint8 = FheUint8::cast_from(leading_zeros_16);

    let (ov_result, result) = rayon::join(
        ||{
            let mant = (&op_mant & 0b0000_1110u8) >> 1u8;
            let res_exp = &x_exp + 0b0000_1000u8;
            let result = &x_sign | &res_exp | &mant;
            result
        },
        ||{
            let diff = (&leading_zeros - 4u8);
            let mask = &encrypted_1023 >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 3u8;
            let res_exp = &x_exp - &sub_exp;
            let result = &x_sign | &res_exp | &mant;
            result
        }
    );

    let overflow = &leading_zeros.eq(3u8);
    overflow.select(&ov_result, &result)
}

pub fn fhe_add_64(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    encrypted_1023: FheUint64,
    server_keys: ServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));
    
    let ns_a = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
    let ns_b = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_mant, y_mant), ((x_exp, diff_exp), (x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let x_mant = (&encrypted_x & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let y_mant = (&encrypted_y & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            (x_mant, y_mant)
        },
        || {
            // Thread 2 + 3: Nested join
            rayon::join(
                || {
                    // Thread 2: Exponents
                    let x_exp = &encrypted_x & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let y_exp = &encrypted_y & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let diff_exp = (&x_exp - &y_exp) >> 52u16;
                    let clipped_diff_exp = diff_exp.min(1023u16);
                    (x_exp, clipped_diff_exp)
                },
                || {
                    // Thread 3: Signs
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_sign, same_sign)
                },
            )
        },
    );
    
    let (sum_mant, diff_mant) = rayon::join(
        ||{
            &x_mant + (&y_mant >> &diff_exp)
        },
        || {
            &x_mant - (&y_mant >> &diff_exp)
        }
    );

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint64::cast_from(op_mant.leading_zeros());

    let (ov_result, result) = rayon::join(
        ||{
            let mant = (&op_mant & 0b0000_0000_0001_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110u64) >> 1u8;
            let res_exp = &x_exp + 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let result = &x_sign | &res_exp | &mant;
            result
        },
        ||{
            let diff = (&leading_zeros - 11u64);
            let mask = &encrypted_1023 >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 52u8;
            let res_exp = &x_exp - &sub_exp;
            let result = &x_sign | &res_exp | &mant;
            result
        }
    );

    let overflow = &leading_zeros.eq(10u8);
    overflow.select(&ov_result, &result)
}

pub fn fhe_lmul16_parallel(
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
            let exp = &x_exp + &y_exp;
            let d = exp.lt(15u8);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 15296u16;
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

pub fn fhe_lmul32_parallel(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: ServerKey,
) -> FheUint32 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Prepare mutable vars for results
    let mut result_sign = None;
    let mut denorm = None;
    let mut result_digits = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 31u8;
            let y_sign = &encrypted_b >> 31u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 31u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_exp = (&encrypted_a & 2139095040u32) >> 23u8;
            let y_exp = (&encrypted_b & 2139095040u32) >> 23u8;
            let exp = &x_exp + &y_exp;
            let d = exp.lt(127u8);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 2147483647u32;
            let y_digits = &encrypted_b & 2147483647u32;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 1064828928u32;
            digits &= 2147483647u32;
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

pub fn fhe_lmul64_parallel(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: ServerKey,
) -> FheUint64 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Prepare mutable vars for results
    let mut result_sign = None;
    let mut denorm = None;
    let mut result_digits = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 63u8;
            let y_sign = &encrypted_b >> 63u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 63u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_exp = (&encrypted_a & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64) >> 52u8;
            let y_exp = (&encrypted_b & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64) >> 52u8;
            let exp = &x_exp + &y_exp;
            let d = exp.lt(1023u16);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let y_digits = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0b0011_1111_1110_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            digits &= 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
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

pub fn fhe_lmul8_parallel(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: ServerKey,
) -> FheUint8 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    // Prepare mutable vars for results
    let mut result_sign = None;
    let mut denorm = None;
    let mut result_digits = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 7u8;
            let y_sign = &encrypted_b >> 7u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 7u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_exp = (&encrypted_a & 0b0111_1000u8) >> 3u8;
            let y_exp = (&encrypted_b & 0b0111_1000u8) >> 3u8;
            let exp = &x_exp + &y_exp;
            let d = exp.lt(7u16);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111u8;
            let y_digits = &encrypted_b & 0b0111_1111u8;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0b0011_0111u8;
            digits &= 0b0111_1111u8;
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

/* 
fn fhe_lmul16(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    client_key: ClientKey
) -> Result<FheUint16, Box<dyn std::error::Error>> {

    let x_sign = &encrypted_a >> 15u8;
    let y_sign = &encrypted_b >> 15u8;

    let x_exp = (&encrypted_a & 31744u16) >> 10u8;
    let y_exp = (&encrypted_b & 31744u16) >> 10u8;

    let x_digits = &encrypted_a & 32767u16;
    let y_digits = &encrypted_b & 32767u16;

    let mut result_sign = &x_sign ^ &y_sign;

    result_sign <<= 15u8;

    let result_exp = &x_exp + &y_exp;

    let denorm = result_exp.lt(15u8);

    let sum_digits = &x_digits + &y_digits;

    let mut result = sum_digits - 15296u16;

    result &= 32767u16;

    let encrypted_0 = FheUint16::try_encrypt(0u16, &client_key)?;

    result = denorm.select(&encrypted_0, &result);
    
    let final_result = result | result_sign;

    Ok(final_result)

}

fn fhe_lmul16_parallel(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    // Cloning data as required by threads
    let a1 = encrypted_a.clone();
    let a2 = encrypted_a.clone();
    let a3 = encrypted_a.clone();
    let b1 = encrypted_b.clone();
    let b2 = encrypted_b.clone();
    let b3 = encrypted_b.clone();
    
    // Cloning server keys for each thread
    let sk1 = server_keys.clone();
    let sk2 = server_keys.clone();
    let sk3 = server_keys.clone();

    // Spawn threads to compute each part in parallel
    let sign_handle = thread::spawn(move || {
        set_server_key(sk1);
        let x_sign = &a1 >> 15u8;
        let y_sign = &b1 >> 15u8;
        let mut result_sign = &x_sign ^ &y_sign;
        result_sign <<= 15u8;
        result_sign
    });

    let exp_handle = thread::spawn(move || {
        set_server_key(sk2);
        let x_exp = (&a2 & 31744u16) >> 10u8;
        let y_exp = (&b2 & 31744u16) >> 10u8;
        let result_exp = &x_exp + &y_exp;
        let denorm = result_exp.lt(15u8);
        denorm
    });

    let digits_handle = thread::spawn(move || {
        set_server_key(sk3);
        let x_digits = &a3 & 32767u16;
        let y_digits = &b3 & 32767u16;
        let mut result = &x_digits + &y_digits;
        result = result - 15296u16;
        result &= 32767u16;
        result
    });

    // Wait for threads to complete and get their results
    let result_sign = sign_handle.join().unwrap();
    let denorm = exp_handle.join().unwrap();
    let mut result_digits = digits_handle.join().unwrap();

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

fn fhe_lmul32_parallel(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    server_keys: ServerKey,
    client_key: ClientKey,
) -> Result<FheUint32, Box<dyn std::error::Error>> {
    // Cloning data as required by threads
    let a1 = encrypted_a.clone();
    let a2 = encrypted_a.clone();
    let a3 = encrypted_a.clone();
    let b1 = encrypted_b.clone();
    let b2 = encrypted_b.clone();
    let b3 = encrypted_b.clone();
    
    // Cloning server keys for each thread
    let sk1 = server_keys.clone();
    let sk2 = server_keys.clone();
    let sk3 = server_keys.clone();

    // Spawn threads to compute each part in parallel
    let sign_handle = thread::spawn(move || {
        set_server_key(sk1);
        let x_sign = &a1 >> 31u8;
        let y_sign = &b1 >> 31u8;
        let mut result_sign = &x_sign ^ &y_sign;
        result_sign <<= 31u8;
        result_sign
    });

    let exp_handle = thread::spawn(move || {
        set_server_key(sk2);
        let x_exp = (&a2 & 2139095040u32) >> 23u8;
        let y_exp = (&b2 & 2139095040u32) >> 23u8;
        let result_exp = &x_exp + &y_exp;
        let denorm = result_exp.lt(127u8);
        denorm
    });

    let digits_handle = thread::spawn(move || {
        set_server_key(sk3);
        let x_digits = &a3 & 2147483647u32;
        let y_digits = &b3 & 2147483647u32;
        let mut result = &x_digits + &y_digits;
        result = result - 1064828928u32;
        result &= 2147483647u32;
        result
    });

    // Wait for threads to complete and get their results
    let result_sign = sign_handle.join().unwrap();
    let denorm = exp_handle.join().unwrap();
    let mut result_digits = digits_handle.join().unwrap();

    // Perform the remaining computation
    let encrypted_0 = FheUint32::try_encrypt(0u32, &client_key)?;
    result_digits = denorm.select(&encrypted_0, &result_digits);
    let final_result = result_digits | result_sign;

    Ok(final_result)
}

fn fhe_pam(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    client_key: ClientKey,
) -> Result<FheUint16, Box<dyn std::error::Error>> {

    let x_sign = &encrypted_a >> 15u8;
    let y_sign = &encrypted_b >> 15u8;

    let x_exp = (&encrypted_a & 31744u16) >> 10u8;
    let y_exp = (&encrypted_b & 31744u16) >> 10u8;

    let x_digits = &encrypted_a & 32767u16;
    let y_digits = &encrypted_b & 32767u16;

    let mut result_sign = &x_sign ^ &y_sign;

    result_sign <<= 15u8;

    let result_exp = &x_exp + &y_exp;

    let denorm = result_exp.lt(15u8);

    let sum_digits = &x_digits + &y_digits;

    let mut result = sum_digits - 15360u16;

    result &= 32767u16;

    let encrypted_0 = FheUint16::try_encrypt(0u16, &client_key)?;

    result = denorm.select(&encrypted_0, &result);
    
    let final_result = result | result_sign;

    Ok(final_result)

}

pub fn fhe_add_same_sign(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let ab_cmp = encrypted_a.ge(&encrypted_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_sign, x_mant, y_mant), (x_exp, diff_exp)) = rayon::join(
        || {
            let x_sign = &encrypted_x & 0b1000_0000_0000_0000u16;
            let x_mant = (&encrypted_x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            let y_mant = (&encrypted_y & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            (x_sign, x_mant, y_mant)
        },
        || {
            let x_exp = &encrypted_x & 0b0111_1100_0000_0000u16;
            let y_exp = &encrypted_y & 0b0111_1100_0000_0000u16;
            let mut diff_exp = &x_exp - &y_exp;
            diff_exp >>= 10u16;
            (x_exp, diff_exp)
        },
    );

    // Sum mantissas
    let sum_mant = x_mant + (y_mant >> diff_exp.clone());

    // Compute the two potential result values in parallel
    let (result12, result11) = rayon::join(
        || {
            let plus_one_cmp = sum_mant.ge(2048u16);
            let res_exp = &x_exp + 1024u16;
            let mant = (&sum_mant & 0b0000_0111_1111_1110u16) >> 1u16;
            let result = &x_sign | &res_exp | &mant;
            plus_one_cmp.select(&result, &encrypted_zero)
        },
        || {
            let same_cmp = sum_mant.lt(2048u16);
            let mant = &sum_mant & 0b0000_0011_1111_1111u16;
            let result = &x_sign | &x_exp | &mant;
            same_cmp.select(&result, &encrypted_zero)
        },
    );
    
    result12 | result11
}

pub fn fhe_add(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let ns_a = &encrypted_a << 1u8;
    let ns_b = &encrypted_b << 1u8;

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );
    
    // Extract mantissas, exponent difference, and sign in parallel
    let ((x_mant, y_mant), ((x_exp, diff_exp), (x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let x_mant = (&encrypted_x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            let y_mant = (&encrypted_y & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
            (x_mant, y_mant)
        },
        || {
            // Thread 2 + 3: Nested join
            rayon::join(
                || {
                    // Thread 2: Exponents
                    let x_exp = &encrypted_x & 0b0111_1100_0000_0000u16;
                    let y_exp = &encrypted_y & 0b0111_1100_0000_0000u16;
                    let diff_exp = (&x_exp - &y_exp) >> 10u16;
                    (x_exp, diff_exp)
                },
                || {
                    // Thread 3: Signs
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000u16;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000u16;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_sign, same_sign)
                },
            )
        },
    );
    
    let (sum_mant, diff_mant) = rayon::join(
        ||{
            &x_mant + (&y_mant >> &diff_exp)
        },
        || {
            &x_mant - (&y_mant >> &diff_exp)
        }
    );
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    // Compute the two potential result values in parallel
    let ((r1, r2, r3, r4, r5, r6), (r7, r8, r9, r10, r11, r12)) = join(
        // First 6 results (threads 1–6)
        || {
            let ((r1, r2), (r3, r4)) = join(
                || {
                    join(
                        || {
                            // Thread 1
                            let cmp = op_mant.ge(2048u16);
                            let res_exp = &x_exp + 1024u16;
                            let mant = (&op_mant & 0b0000_0111_1111_1110u16) >> 1u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                        || {
                            // Thread 2
                            let cmp = op_mant.lt(2048u16) & op_mant.ge(1024u16);
                            let mant = &op_mant & 0b0000_0011_1111_1111u16;
                            let result = &x_sign | &x_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                    )
                },
                || {
                    join(
                        || {
                            // Thread 3
                            let cmp = op_mant.lt(1024u16) & op_mant.ge(512u16);
                            let mant = (&op_mant & 0b0000_0001_1111_1111u16) << 1u16;
                            let res_exp = &x_exp - 1024u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                        || {
                            // Thread 4
                            let cmp = op_mant.lt(512u16) & op_mant.ge(256u16);
                            let mant = (&op_mant & 0b0000_0000_1111_1111u16) << 2u16;
                            let res_exp = &x_exp - 2048u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                    )
                },
            );
            let (r5, r6) = join(
                || {
                    // Thread 5
                    let cmp = op_mant.lt(256u16) & op_mant.ge(128u16);
                    let mant = (&op_mant & 0b0000_0000_0111_1111u16) << 3u16;
                    let res_exp = &x_exp - 3072u16;
                    let result = &x_sign | &res_exp | &mant;
                    cmp.select(&result, &encrypted_zero)
                },
                || {
                    // Thread 6
                    let cmp = op_mant.lt(128u16) & op_mant.ge(64u16);
                    let mant = (&op_mant & 0b0000_0000_0011_1111u16) << 4u16;
                    let res_exp = &x_exp - 4096u16;
                    let result = &x_sign | &res_exp | &mant;
                    cmp.select(&result, &encrypted_zero)
                },
            );
            (r1, r2, r3, r4, r5, r6)
        },
        // Next 6 results (threads 7–12)
        || {
            let ((r7, r8), (r9, r10)) = join(
                || {
                    join(
                        || {
                            // Thread 7
                            let cmp = op_mant.lt(64u16) & op_mant.ge(32u16);
                            let mant = (&op_mant & 0b0000_0000_0001_1111u16) << 5u16;
                            let res_exp = &x_exp - 5120u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                        || {
                            // Thread 8
                            let cmp = op_mant.lt(32u16) & op_mant.ge(16u16);
                            let mant = (&op_mant & 0b0000_0000_0000_1111u16) << 6u16;
                            let res_exp = &x_exp - 6144u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                    )
                },
                || {
                    join(
                        || {
                            // Thread 9
                            let cmp = op_mant.lt(16u16) & op_mant.ge(8u16);
                            let mant = (&op_mant & 0b0000_0000_0000_0111u16) << 7u16;
                            let res_exp = &x_exp - 7168u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                        || {
                            // Thread 10
                            let cmp = op_mant.lt(8u16) & op_mant.ge(4u16);
                            let mant = (&op_mant & 0b0000_0000_0000_0011u16) << 8u16;
                            let res_exp = &x_exp - 8192u16;
                            let result = &x_sign | &res_exp | &mant;
                            cmp.select(&result, &encrypted_zero)
                        },
                    )
                },
            );
            let (r11, r12) = join(
                || {
                    // Thread 11
                    let cmp = op_mant.lt(4u16) & op_mant.ge(2u16);
                    let mant = (&op_mant & 0b0000_0000_0000_0001u16) << 9u16;
                    let res_exp = &x_exp - 9216u16;
                    let result = &x_sign | &res_exp | &mant;
                    cmp.select(&result, &encrypted_zero)
                },
                || {
                    // Thread 12
                    let cmp = op_mant.lt(2u16);
                    let mant = (&op_mant & 0b0000_0000_0000_0001u16) << 10u16;
                    let res_exp = &x_exp - 10240u16;
                    let result = &x_sign | &res_exp | &mant;
                    cmp.select(&result, &encrypted_zero)
                },
            );
            (r7, r8, r9, r10, r11, r12)
        },
    );
    
    // Combine all results
    let inputs = vec![r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12];
    
    // Use `reduce` to combine all elements in parallel
    let final_result = inputs.par_iter()
    .cloned()
    .reduce_with(|a, b| a | b)
    .expect("inputs must not be empty");

    final_result
    
}
*/