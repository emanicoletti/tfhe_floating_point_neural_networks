use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

/* GPU OPERATIONS */

/* GPU-oriented addition-based multiplication (LMUL) for 8 bits floating points (E4M3)*/
pub fn fhe_lmul8_gpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 0b0111_1000u8) >> 3u8,
        || (&encrypted_b & 0b0111_1000u8) >> 3u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 7u8;
            let y_sign = &encrypted_b >> 7u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 7u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111u8;
            let y_digits = &encrypted_b & 0b0111_1111u8;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0b0011_0111u8;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(7u8));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u8));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u8));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* GPU-oriented addition-based multiplication (LMUL) for 16 bits floating points */
pub fn fhe_lmul16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 31744u16) >> 10u8,
        || (&encrypted_b & 31744u16) >> 10u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 15u8;
            let y_sign = &encrypted_b >> 15u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 15u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 15296u16;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(15u16));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u16));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u16));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* GPU-oriented addition-based multiplication (LMUL) for 32 bits floating points */
pub fn fhe_lmul32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (x_exp, y_exp) = rayon::join(
            || (&encrypted_a & 2139095040u32) >> 23u8,
            || (&encrypted_b & 2139095040u32) >> 23u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 31u8;
            let y_sign = &encrypted_b >> 31u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 31u8;
            result_sign = Some(sign);
        });
        s.spawn(|_| {
            let x_digits = &encrypted_a & 2147483647u32;
            let y_digits = &encrypted_b & 2147483647u32;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 1064828928u32;
            result_digits = Some(digits);
        });
        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(127u8));
        });
        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u32));
        });
        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u32));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* GPU-oriented addition-based multiplication (LMUL) for 64 bits floating points */
pub fn fhe_lmul64_gpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: CudaServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 0x7FF0_0000_0000_0000u64) >> 52u8,
        || (&encrypted_b & 0x7FF0_0000_0000_0000u64) >> 52u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 63u8;
            let y_sign = &encrypted_b >> 63u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 63u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
            let y_digits = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0x3FEF_0000_0000_0000u64;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(1023u64));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u64));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u64));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* CPU OPERATIONS */

/* CPU-oriented addition-based multiplication (LMUL) for 8 bits floating points (E4M3) */
pub fn fhe_lmul8_cpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: ServerKey,
) -> FheUint8 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

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
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* CPU-oriented addition-based multiplication (LMUL) for 16 bits floating points */
pub fn fhe_lmul16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    _server_keys: ServerKey,
) -> FheUint16 {
let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 31744u16) >> 10u8,
        || (&encrypted_b & 31744u16) >> 10u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 15u8;
            let y_sign = &encrypted_b >> 15u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 15u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 15296u16;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(15u16));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u16));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u16));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* CPU-oriented addition-based multiplication (LMUL) for 32 bits floating points */
pub fn fhe_lmul32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    _server_keys: ServerKey,
) -> FheUint32 {

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 2139095040u32) >> 23u8,
        || (&encrypted_b & 2139095040u32) >> 23u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 31u8;
            let y_sign = &encrypted_b >> 31u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 31u8;
            result_sign = Some(sign);
        });
        s.spawn(|_| {
            let x_digits = &encrypted_a & 2147483647u32;
            let y_digits = &encrypted_b & 2147483647u32;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 1064828928u32;
            result_digits = Some(digits);
        });
        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(127u8));
        });
        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u32));
        });
        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u32));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* CPU-oriented addition-based multiplication (LMUL) for 64 bits floating points */
pub fn fhe_lmul64_cpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    _server_keys: ServerKey,
) -> FheUint64 {

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 0x7FF0_0000_0000_0000u64) >> 52u8,
        || (&encrypted_b & 0x7FF0_0000_0000_0000u64) >> 52u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 63u8;
            let y_sign = &encrypted_b >> 63u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 63u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
            let y_digits = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0x3FEF_0000_0000_0000u64;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(1023u64));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u64));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u64));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* GPU-oriented addition-based multiplication (LMUL) for 8 bits floating points (E4M3)*/
pub fn fhe_pam_mul8_gpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

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
            digits = digits - 0b0011_1000u8;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* GPU-oriented addition-based multiplication (LMUL) for 16 bits floating points */
pub fn fhe_pam_mul16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

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
            digits = digits - 15360u16;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* GPU-oriented addition-based multiplication (LMUL) for 32 bits floating points */
pub fn fhe_pam_mul32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

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
            digits = digits - 1065353216u32;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;
    final_result
}

/* GPU-oriented addition-based multiplication (LMUL) for 64 bits floating points */
pub fn fhe_pam_mul64_gpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: CudaServerKey,
) -> FheUint64 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

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
            digits = digits - 0b0011_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* CPU OPERATIONS */

/* CPU-oriented addition-based multiplication (LMUL) for 8 bits floating points (E4M3) */
pub fn fhe_pam_mul8_cpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    _server_keys: ServerKey,
) -> FheUint8 {

    let (x_exp, y_exp) = rayon::join(
        || (&encrypted_a & 0b0111_1000u8) >> 3u8,
        || (&encrypted_b & 0b0111_1000u8) >> 3u8
    );

    let mut result_sign = None;
    let mut result_digits = None;
    let mut d_out = None;
    let mut d1_out = None;
    let mut d2_out = None;

    rayon::scope(|s| {
        s.spawn(|_| {
            let x_sign = &encrypted_a >> 7u8;
            let y_sign = &encrypted_b >> 7u8;
            let mut sign = &x_sign ^ &y_sign;
            sign <<= 7u8;
            result_sign = Some(sign);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111u8;
            let y_digits = &encrypted_b & 0b0111_1111u8;
            let mut digits = &x_digits + &y_digits;
            digits = digits - 0b0011_0111u8;
            result_digits = Some(digits);
        });

        s.spawn(|_| {
            let exp = &x_exp + &y_exp;
            d_out = Some(exp.lt(7u8));
        });

        s.spawn(|_| {
            d1_out = Some(x_exp.eq(0u8));
        });

        s.spawn(|_| {
            d2_out = Some(y_exp.eq(0u8));
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let mut result_digits = result_digits.expect("digits result missing");
    let d = d_out.expect("d missing");
    let d1 = d1_out.expect("d1 missing");
    let d2 = d2_out.expect("d2 missing");

    let denorm_flag = d | d1 | d2;

    result_digits = denorm_flag.select(&encrypted_zero, &result_digits);
    result_digits | result_sign
}

/* CPU-oriented addition-based multiplication (LMUL) for 16 bits floating points */
pub fn fhe_pam_mul16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    _server_keys: ServerKey,
) -> FheUint16 {

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
            digits = digits - 15360u16;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* CPU-oriented addition-based multiplication (LMUL) for 32 bits floating points */
pub fn fhe_pam_mul32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    _server_keys: ServerKey,
) -> FheUint32 {

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
            digits = digits - 1065353216u32;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}

/* CPU-oriented addition-based multiplication (LMUL) for 64 bits floating points */
pub fn fhe_pam_mul64_cpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    _server_keys: ServerKey,
) -> FheUint64 {

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
            digits = digits - 0b0011_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
            result_digits = Some(digits);
        });
    });

    let result_sign = result_sign.expect("sign result missing");
    let denorm = denorm.expect("denorm result missing");
    let mut result_digits = result_digits.expect("digits result missing");

    result_digits = denorm.select(&encrypted_zero, &result_digits);
    let final_result = result_digits | result_sign;

    final_result
}