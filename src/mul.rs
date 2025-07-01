use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint16, FheUint32, ClientKey, ServerKey, CudaServerKey};
use std::thread;


pub fn fhe_lmul16(
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

pub fn fhe_lmul16_parallel(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
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