use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

/* GPU OPERATIONS */

/* GPU-oriented addition-based division (LDIV) for 8 bits floating points (E4M3) */
pub fn fhe_ldiv8_gpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x80u8,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x78u8) >> 3u8;
                    let y_exp = (&encrypted_b & 0x78u8) >> 3u8;
                    let exp = (&x_exp - &y_exp) + 8u8;
                    let d = exp.gt(15u8);
                    let d_zero = encrypted_a.eq(0u8);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7Fu8;
                    let y_digits = &encrypted_b & 0x7Fu8;
                    ((&x_digits - &y_digits) + 0x37u8) & 0x7Fu8
                }
            )
        }
    );

    result_digits = result_digits | result_sign;

    denorm.select(&encrypted_zero, &result_digits)

}

/* GPU-oriented addition-based division (LDIV) for 16 bits floating points */
pub fn fhe_ldiv16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x8000u16,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x7C00u16) >> 10u8;
                    let y_exp = (&encrypted_b & 0x7C00u16) >> 10u8;
                    let exp = (&x_exp - &y_exp) + 15u16;
                    let d = exp.gt(31u16);
                    let d_zero = encrypted_a.eq(0u16);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7FFFu16;
                    let y_digits = &encrypted_b & 0x7FFFu16;
                    ((&x_digits - &y_digits) + 0x3BC0u16) & 0x7FFFu16
                }
            )
        }
    );

    result_digits = result_digits | result_sign;

    denorm.select(&encrypted_zero, &result_digits)
}

/* GPU-oriented addition-based division (LDIV) for 32 bits floating points */
pub fn fhe_ldiv32_gpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {

        rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
            || {
                (&encrypted_a ^ &encrypted_b) & 0x8000_0000u32
            },
            || {
                rayon::join(
                    || {
                        let x_exp = (&encrypted_a & 0x7F80_0000u32) >> 23u8;
                        let y_exp = (&encrypted_b & 0x7F80_0000u32) >> 23u8;
                        let exp = (&x_exp - &y_exp) + 127u32;
                        let d = exp.gt(255u32);
                        let d_zero = encrypted_a.eq(0u32);
                        d | d_zero
                    },
                    || {
                        let x_digits = &encrypted_a & 0x7FFF_FFFFu32;
                        let y_digits = &encrypted_b & 0x7FFF_FFFFu32;
                        
                        ((&x_digits - &y_digits) + 0x3F78_0000u32) & 0x7FFF_FFFFu32
                    }
                )
            }
        );

        result_digits = result_digits | result_sign;
        
        denorm.select(&encrypted_zero, &result_digits)
        
}

/* GPU-oriented addition-based division (LDIV) for 64 bits floating points */
pub fn fhe_ldiv64_gpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: CudaServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x8000_0000_0000_0000u64,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x7FF0_0000_0000_0000u64) >> 52u8;
                    let y_exp = (&encrypted_b & 0x7FF0_0000_0000_0000u64) >> 52u8;
                    let exp = (&x_exp - &y_exp) + 1023u64;
                    let d = exp.gt(2047u64);
                    let d_zero = encrypted_a.eq(0u64);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
                    let y_digits = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
                    ((&x_digits - &y_digits) + 0x3FEF_0000_0000_0000u64) & 0x7FFF_FFFF_FFFF_FFFFu64
                }
            )
        }
    );

    result_digits = result_digits | result_sign;

    denorm.select(&encrypted_zero, &result_digits)
    
}

/* CPU OPERATIONS */

/* CPU-oriented addition-based division (LDIV) for 8 bits floating points (E4M3) */
pub fn fhe_ldiv8_cpu(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    encrypted_zero: FheUint8,
    server_keys: ServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x80u8,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x78u8) >> 3u8;
                    let y_exp = (&encrypted_b & 0x78u8) >> 3u8;
                    let exp = (&x_exp - &y_exp) + 8u8;
                    let d = exp.gt(15u8);
                    let d_zero = encrypted_a.eq(0u8);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7Fu8;
                    let y_digits = &encrypted_b & 0x7Fu8;
                    ((&x_digits - &y_digits) + 0x37u8) & 0x7Fu8
                }
            )
        }
    );

    result_digits = result_digits | result_sign;
    denorm.select(&encrypted_zero, &result_digits)
    
}

/* CPU-oriented addition-based division (LDIV) for 16 bits floating points */
pub fn fhe_ldiv16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: ServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x8000u16,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x7C00u16) >> 10u8;
                    let y_exp = (&encrypted_b & 0x7C00u16) >> 10u8;
                    let exp = (&x_exp - &y_exp) + 15u16;
                    let d = exp.gt(31u16);
                    let d_zero = encrypted_a.eq(0u16);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7FFFu16;
                    let y_digits = &encrypted_b & 0x7FFFu16;
                    ((&x_digits - &y_digits) + 0x3BC0u16) & 0x7FFFu16
                }
            )
        }
    );

    result_digits = result_digits | result_sign;
    denorm.select(&encrypted_zero, &result_digits)
}

/* CPU-oriented addition-based division (LDIV) for 32 bits floating points */
pub fn fhe_ldiv32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: ServerKey,
) -> FheUint32 {

    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
            || {
                (&encrypted_a ^ &encrypted_b) & 0x8000_0000u32
            },
            || {
                rayon::join(
                    || {
                        let x_exp = (&encrypted_a & 0x7F80_0000u32) >> 23u8;
                        let y_exp = (&encrypted_b & 0x7F80_0000u32) >> 23u8;
                        let exp = (&x_exp - &y_exp) + 127u32;
                        let d = exp.gt(255u32);
                        let d_zero = encrypted_a.eq(0u32);
                        d | d_zero
                    },
                    || {
                        let x_digits = &encrypted_a & 0x7FFF_FFFFu32;
                        let y_digits = &encrypted_b & 0x7FFF_FFFFu32;
                        
                        ((&x_digits - &y_digits) + 0x3F78_0000u32) & 0x7FFF_FFFFu32
                    }
                )
            }
        );

        result_digits = result_digits | result_sign;

        denorm.select(&encrypted_zero, &result_digits) 
}

/* CPU-oriented addition-based division (LDIV) for 64 bits floating points */
pub fn fhe_ldiv64_cpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: ServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (result_sign, (denorm, mut result_digits)) = rayon::join(
        || (&encrypted_a ^ &encrypted_b) & 0x8000_0000_0000_0000u64,
        || {
            rayon::join(
                || {
                    let x_exp = (&encrypted_a & 0x7FF0_0000_0000_0000u64) >> 52u8;
                    let y_exp = (&encrypted_b & 0x7FF0_0000_0000_0000u64) >> 52u8;
                    let exp = (&x_exp - &y_exp) + 1023u64;
                    let d = exp.gt(2047u64);
                    let d_zero = encrypted_a.eq(0u64);
                    d | d_zero
                },
                || {
                    let x_digits = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
                    let y_digits = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
                    ((&x_digits - &y_digits) + 0x3FEF_0000_0000_0000u64) & 0x7FFF_FFFF_FFFF_FFFFu64
                }
            )
        }
    );

    result_digits = result_digits | result_sign;
    denorm.select(&encrypted_zero, &result_digits)
}

/* GPU-oriented addition-based division (LDIV) for 8 bits floating points (E4M3)*/
pub fn fhe_pam_div8_gpu(
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
            let mut exp = &x_exp - &y_exp;
            exp += 8u8;
            let d = exp.gt(15u8);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111u8;
            let y_digits = &encrypted_b & 0b0111_1111u8;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 0b0011_1000u8;
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

/* GPU-oriented addition-based division (LDIV) for 16 bits floating points */
pub fn fhe_pam_div16_gpu(
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
            let mut exp = &x_exp - &y_exp;
            exp += 15u16;
            let d = exp.gt(31u16);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 15360u16;
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

/* GPU-oriented addition-based division (LDIV) for 32 bits floating points */
pub fn fhe_pam_div32_gpu(
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
            let mut exp = &x_exp - &y_exp;
            exp += 127u32;
            let d = exp.gt(255u32) | exp.eq(0u32);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 2147483647u32;
            let y_digits = &encrypted_b & 2147483647u32;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 1065353216u32;
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

/* GPU-oriented addition-based division (LDIV) for 64 bits floating points */
pub fn fhe_pam_div64_gpu(
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
            let mut exp = &x_exp - &y_exp;
            exp += 1023u64;
            let d = exp.gt(2047u64);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let y_digits = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 0b0011_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
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

/* CPU-oriented addition-based division (LDIV) for 8 bits floating points (E4M3) */
pub fn fhe_pam_div8_cpu(
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
            let mut exp = &x_exp - &y_exp;
            exp += 8u8;
            let d = exp.gt(15u8);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111u8;
            let y_digits = &encrypted_b & 0b0111_1111u8;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 0b0011_0111u8;
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

/* CPU-oriented addition-based division (LDIV) for 16 bits floating points */
pub fn fhe_pam_div16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: ServerKey,
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
            let mut exp = &x_exp - &y_exp;
            exp += 15u16;
            let d = exp.gt(31u16);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 32767u16;
            let y_digits = &encrypted_b & 32767u16;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 15360u16;
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

/* CPU-oriented addition-based division (LDIV) for 32 bits floating points */
pub fn fhe_pam_div32_cpu(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: ServerKey,
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
            let mut exp = &x_exp - &y_exp;
            exp += 127u32;
            let d = exp.gt(255u32) | exp.eq(0u32);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 2147483647u32;
            let y_digits = &encrypted_b & 2147483647u32;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 1065353216u32;
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

/* CPU-oriented addition-based division (LDIV) for 64 bits floating points */
pub fn fhe_pam_div64_cpu(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    encrypted_zero: FheUint64,
    server_keys: ServerKey,
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
            let mut exp = &x_exp - &y_exp;
            exp += 1023u64;
            let d = exp.gt(2047u64);
            denorm = Some(d);
        });

        s.spawn(|_| {
            let x_digits = &encrypted_a & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let y_digits = &encrypted_b & 0b0111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111_1111u64;
            let mut digits = &x_digits - &y_digits;
            digits = digits + 0b0011_1111_1111_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000u64;
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