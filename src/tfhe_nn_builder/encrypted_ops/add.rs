use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

/* GPU OPERATIONS */

pub fn fhe_add_int8(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8, 
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let result = &encrypted_a * &encrypted_b;
    result
}

pub fn fhe_add_int32(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32, 
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let result = &encrypted_a + &encrypted_b;
    result
}

pub fn fhe_add_int64(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64, 
    server_keys: CudaServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let result = &encrypted_a * &encrypted_b;
    result
}

/* GPU-oriented integer-based addition for 8 bits floating points (E4M3) */
pub fn fhe_add8_gpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_mask: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a & 0b0111_1111u8;
    let ns_b = &encrypted_b & 0b0111_1111u8;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1000u8;
            let denorm_y = y_exp.eq(0u16);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0111u8),
                &((&encrypted_y & 0b0000_0111u8) | 0b0000_1000u8),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0111u8) | 0b0000_1000u8;
                    let x_sign = &encrypted_x & 0b1000_0000u8;
                    let y_sign = &encrypted_y & 0b1000_0000u8;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros_16: FheUint16 = FheUint16::cast_from(op_mant.leading_zeros());
    let leading_zeros: FheUint8 = FheUint8::cast_from(leading_zeros_16);

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(3u8);
            let mant = (&op_mant & 0b0000_1110u8) >> 1u8;
            let res_exp = &x_exp + 0b0000_1000u8;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 4u8;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 3u8;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

/* GPU-oriented integer-based addition for 16 bits floating points */
pub fn fhe_add16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_mask: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let ns_a = &encrypted_a & 0b0111_1111_1111_1111u16;
    let ns_b = &encrypted_b & 0b0111_1111_1111_1111u16;

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let y_exp = &encrypted_y & 0b0111_1100_0000_0000u16;
            let denorm_y = y_exp.eq(0u16);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0011_1111_1111u16),
                &((&encrypted_y & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000u16;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000u16;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint16::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.eq(4u16);
            let mant = (&op_mant & 0b0000_0111_1111_1110u16) >> 1u16;
            let res_exp = &x_exp + 1024u16;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 5u16;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 10u16;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

pub fn fhe_add32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_mask: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let ns_a = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let ns_b = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
            let denorm_y = y_exp.eq(0u32);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32),
                &((&encrypted_y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = op_mant.leading_zeros();
    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(7u32);
            let mant = (&op_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u16;
            let res_exp = &x_exp + 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 8u32;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 23u16;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

/* GPU-oriented integer-based addition for 64 bits floating points */
pub fn fhe_add64_gpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_mask: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: CudaServerKey,
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
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let denorm_y = y_exp.eq(0u64);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64),
                &((&encrypted_y & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64)
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint64::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(10u64);
            let mant = (&op_mant & 0b0000_0000_0001_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110u64) >> 1u64;
            let res_exp = &x_exp + 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 11u64;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 52u64;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

/* CPU OPERATIONS */

/* CPU-oriented integer-based addition for 8 bits floating points (E4M3) */
pub fn fhe_add8_cpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_mask: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: ServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a & 0b0111_1111u8;
    let ns_b = &encrypted_b & 0b0111_1111u8;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1000u8;
            let denorm_y = y_exp.eq(0u16);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0111u8),
                &((&encrypted_y & 0b0000_0111u8) | 0b0000_1000u8),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0111u8) | 0b0000_1000u8;
                    let x_sign = &encrypted_x & 0b1000_0000u8;
                    let y_sign = &encrypted_y & 0b1000_0000u8;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros_16: FheUint16 = FheUint16::cast_from(op_mant.leading_zeros());
    let leading_zeros: FheUint8 = FheUint8::cast_from(leading_zeros_16);

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(3u8);
            let mant = (&op_mant & 0b0000_1110u8) >> 1u8;
            let res_exp = &x_exp + 0b0000_1000u8;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 4u8;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 3u8;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

/* CPU-oriented integer-based addition for 16 bits floating points */
pub fn fhe_add16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_mask: FheUint16,
    encrypted_zero: FheUint16,
    _server_keys: ServerKey,
) -> FheUint16 {
    //rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a & 0b0111_1111_1111_1111u16;
    let ns_b = &encrypted_b & 0b0111_1111_1111_1111u16;

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            // Thread 1: Mantissas
            let y_exp = &encrypted_y & 0b0111_1100_0000_0000u16;
            let denorm_y = y_exp.eq(0u16);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0011_1111_1111u16),
                &((&encrypted_y & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000u16;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000u16;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;
    
    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint16::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(4u16);
            let mant = (&op_mant & 0b0000_0111_1111_1110u16) >> 1u16;
            let res_exp = &x_exp + 1024u16;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 5u16;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 10u16;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}

/* CPU-oriented integer-based addition for 32 bits floating points */
pub fn fhe_add32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_mask: FheUint32,
    encrypted_zero: FheUint32,
    _server_keys: ServerKey,
) -> FheUint32 {
    //rayon::broadcast(|_| set_server_key(server_keys.clone()));

    
    let ns_a = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let ns_b = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    

    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    // Extract mantissas, exponent difference, and sign in parallel
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
            let denorm_y = y_exp.eq(0u32);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32),
                &((&encrypted_y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32),
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = op_mant.leading_zeros();

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(7u32);
            let mant = (&op_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u16;
            let res_exp = &x_exp + 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 8u32;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 23u16;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}


/* CPU-oriented integer-based addition for 64 bits floating points */
pub fn fhe_add64_cpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_mask: FheUint64,
    encrypted_zero: FheUint64,
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
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign, same_sign))) = rayon::join(
        || {
            let y_exp = &encrypted_y & 0b0111_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let denorm_y = y_exp.eq(0u64);
            let y_mant = denorm_y.select(
                &(&encrypted_y & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64),
                &((&encrypted_y & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64)
            );
            y_mant
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
                    let x_mant = (&encrypted_x & 0b0000_0000_0000_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64) | 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let x_sign = &encrypted_x & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let y_sign = &encrypted_y & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
                    let same_sign = x_sign.eq(&y_sign);
                    (x_mant, x_sign, same_sign)
                },
            )
        },
    );
    
    let shifted_y = &y_mant >> &diff_exp;
    let sum_mant = &x_mant + &shifted_y;
    let diff_mant = &x_mant - &shifted_y;

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint64::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        ||{
            let overflow = leading_zeros.clone().eq(10u8);
            let mant = (&op_mant & 0b0000_0000_0001_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1110u64) >> 1u8;
            let res_exp = &x_exp + 0b0000_0000_0001_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        ||{
            let diff = &leading_zeros - 11u64;
            let mask = &encrypted_mask >> &diff;
            let mant = (&op_mant & &mask) << &diff;
            let sub_exp = &diff << 52u64;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}