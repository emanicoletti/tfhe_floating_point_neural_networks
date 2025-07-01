use tfhe::prelude::*;
use tfhe::{set_server_key, ConfigBuilder, FheUint16, FheUint32, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use std::time::Instant;
use rand::Rng;
use half::f16;
use rayon::prelude::*;
use rayon::{join, scope};
use std::thread;


pub fn fhe_add(
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

    /* 
    let mut r1 = encrypted_zero.clone();
    let mut r2 = encrypted_zero.clone();
    let mut r3 = encrypted_zero.clone();
    let mut r4 = encrypted_zero.clone();
    let mut r5 = encrypted_zero.clone();
    let mut r6 = encrypted_zero.clone();
    let mut r7 = encrypted_zero.clone();
    let mut r8 = encrypted_zero.clone();
    let mut r9 = encrypted_zero.clone();
    let mut r10 = encrypted_zero.clone();
    let mut r11 = encrypted_zero.clone();
    let mut r12 = encrypted_zero.clone();

    scope(|s| {
        let r1_mut = &mut r1;
        let r2_mut = &mut r2;
        let r3_mut = &mut r3;
        let r4_mut = &mut r4;
        let r5_mut = &mut r5;
        let r6_mut = &mut r6;
        let r7_mut = &mut r7;
        let r8_mut = &mut r8;
        let r9_mut = &mut r9;
        let r10_mut = &mut r10;
        let r11_mut = &mut r11;
        let r12_mut = &mut r12;

        let op_mant = &op_mant;
        let leading_zeros = &leading_zeros;
        let x_exp = &x_exp;
        let x_sign = &x_sign;
        let zero = &encrypted_zero;

        s.spawn(move |_| {
            let cmp = leading_zeros.eq(4u16);
            let mant = (op_mant & 0b0000_0111_1111_1110u16) >> 1u16;
            let res_exp = x_exp + 1024u16;
            let result = x_sign | &res_exp | &mant;
            *r1_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(5u16);
            let mant = op_mant & 0b0000_0011_1111_1111u16;
            let result = x_sign | x_exp | &mant;
            *r2_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(6u16);
            let mant = (op_mant & 0b0000_0001_1111_1111u16) << 1u16;
            let res_exp = x_exp - 1024u16;
            let result = x_sign | &res_exp | &mant;
            *r3_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(7u16);
            let mant = (op_mant & 0b0000_0000_1111_1111u16) << 2u16;
            let res_exp = x_exp - 2048u16;
            let result = x_sign | &res_exp | &mant;
            *r4_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(8u16);
            let mant = (op_mant & 0b0000_0000_0111_1111u16) << 3u16;
            let res_exp = x_exp - 3072u16;
            let result = x_sign | &res_exp | &mant;
            *r5_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(9u16);
            let mant = (op_mant & 0b0000_0000_0011_1111u16) << 4u16;
            let res_exp = x_exp - 4096u16;
            let result = x_sign | &res_exp | &mant;
            *r6_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(10u16);
            let mant = (op_mant & 0b0000_0000_0001_1111u16) << 5u16;
            let res_exp = x_exp - 5120u16;
            let result = x_sign | &res_exp | &mant;
            *r7_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(11u16);
            let mant = (op_mant & 0b0000_0000_0000_1111u16) << 6u16;
            let res_exp = x_exp - 6144u16;
            let result = x_sign | &res_exp | &mant;
            *r8_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(12u16);
            let mant = (op_mant & 0b0000_0000_0000_0111u16) << 7u16;
            let res_exp = x_exp - 7168u16;
            let result = x_sign | &res_exp | &mant;
            *r9_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(13u16);
            let mant = (op_mant & 0b0000_0000_0000_0011u16) << 8u16;
            let res_exp = x_exp - 8192u16;
            let result = x_sign | &res_exp | &mant;
            *r10_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(14u16);
            let mant = (op_mant & 0b0000_0000_0000_0001u16) << 9u16;
            let res_exp = x_exp - 9216u16;
            let result = x_sign | &res_exp | &mant;
            *r11_mut = cmp.select(&result, zero);
        });
        s.spawn(move |_| {
            let cmp = leading_zeros.eq(15u16);
            let mant = (op_mant & 0b0000_0000_0000_0001u16) << 10u16;
            let res_exp = x_exp - 10240u16;
            let result = x_sign | &res_exp | &mant;
            *r12_mut = cmp.select(&result, zero);
        });
    });

    
    // Combine a
    ll results
    let inputs = vec![
        r1.clone(), r2.clone(), r3.clone(), r4.clone(), r5.clone(), r6.clone(),
        r7.clone(), r8.clone(), r9.clone(), r10.clone(), r11.clone(), r12.clone(),
    ];
    
    // Use `reduce` to combine all elements in parallel
    let final_result = inputs.par_iter()
    .cloned()
    .reduce_with(|a, b| a | b)
    .expect("inputs must not be empty");
    */

    //final_result
    
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

pub fn fhe_negate(
    encrypted_a: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16{
    set_server_key(server_keys.clone());
    encrypted_a ^ 0b1000_0000_0000_0000
}