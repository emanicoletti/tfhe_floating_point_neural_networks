use tfhe::prelude::*;
use tfhe::{CudaServerKey, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, set_server_key};

/* GPU OPERATIONS */

pub fn fhe_add_int8(
    encrypted_a: FheUint8,
    encrypted_b: FheUint8,
    server_keys: CudaServerKey,
) -> FheUint8 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));
    &encrypted_a + &encrypted_b
}

pub fn fhe_add_int32(
    encrypted_a: FheUint32,
    encrypted_b: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));
    &encrypted_a + &encrypted_b
}

pub fn fhe_add_int64(
    encrypted_a: FheUint64,
    encrypted_b: FheUint64,
    server_keys: CudaServerKey,
) -> FheUint64 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));
    &encrypted_a + &encrypted_b
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

    let ns_a = &encrypted_a & 0x7Fu8;
    let ns_b = &encrypted_b & 0x7Fu8;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x78u8;
            let mant = (&encrypted_x & 0x07u8) | 0x08u8;
            let sign = &encrypted_x & 0x80u8;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x78u8;
            let mant = &encrypted_y & 0x07u8;
            let sign = &encrypted_y & 0x80u8;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u8);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x08u8))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 3u8;
                    diff_exp.min(7u8)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    // FIX RESTORED: Cast FheUint32 back down to FheUint8 safely
    let leading_zeros_16 = FheUint16::cast_from(op_mant.leading_zeros());
    let leading_zeros = FheUint8::cast_from(leading_zeros_16);

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(3u8);
            let mant = (&op_mant & 0x0Eu8) >> 1u8;
            let res_exp = &x_exp + 0x08u8;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 4u8;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 3u8;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
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

    let ns_a = &encrypted_a & 0x7FFFu16;
    let ns_b = &encrypted_b & 0x7FFFu16;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7C00u16;
            let mant = (&encrypted_x & 0x03FFu16) | 0x0400u16;
            let sign = &encrypted_x & 0x8000u16;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7C00u16;
            let mant = &encrypted_y & 0x03FFu16;
            let sign = &encrypted_y & 0x8000u16;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u16);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0400u16))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 10u16;
                    diff_exp.min(15u16)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    // FIX RESTORED: Cast FheUint32 back down to FheUint16
    let leading_zeros = FheUint16::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(4u16);
            let mant = (&op_mant & 0x07FEu16) >> 1u16;
            let res_exp = &x_exp + 0x0400u16;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 5u16;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 10u16;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
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

    let ns_a = &encrypted_a & 0x7FFF_FFFFu32;
    let ns_b = &encrypted_b & 0x7FFF_FFFFu32;
    let ab_cmp = ns_a.ge(&ns_b);
    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7F80_0000u32;
            let mant = (&encrypted_x & 0x007F_FFFFu32) | 0x0080_0000u32;
            let sign = &encrypted_x & 0x8000_0000u32;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7F80_0000u32;
            let mant = &encrypted_y & 0x007F_FFFFu32;
            let sign = &encrypted_y & 0x8000_0000u32;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u32);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0080_0000u32))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 23u16;
                    diff_exp.min(31u16)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);
    let leading_zeros = op_mant.leading_zeros();
    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.clone().eq(7u32);
            let mant = (&op_mant & 0x00FF_FFFEu32) >> 1u16;
            let res_exp = &x_exp + 0x0080_0000u32;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 8u32;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;
            let sub_exp = &diff << 23u16;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        },
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

    let ns_a = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
    let ns_b = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7FF0_0000_0000_0000u64;
            let mant = (&encrypted_x & 0x000F_FFFF_FFFF_FFFFu64) | 0x0010_0000_0000_0000u64;
            let sign = &encrypted_x & 0x8000_0000_0000_0000u64;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7FF0_0000_0000_0000u64;
            let mant = &encrypted_y & 0x000F_FFFF_FFFF_FFFFu64;
            let sign = &encrypted_y & 0x8000_0000_0000_0000u64;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u64);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0010_0000_0000_0000u64))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 52u64;
                    diff_exp.min(1023u64)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint64::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(10u64);
            let mant = (&op_mant & 0x001F_FFFF_FFFF_FFFEu64) >> 1u64;
            let res_exp = &x_exp + 0x0010_0000_0000_0000u64;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 11u64;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 52u64;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
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

    let ns_a = &encrypted_a & 0x7Fu8;
    let ns_b = &encrypted_b & 0x7Fu8;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x78u8;
            let mant = (&encrypted_x & 0x07u8) | 0x08u8;
            let sign = &encrypted_x & 0x80u8;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x78u8;
            let mant = &encrypted_y & 0x07u8;
            let sign = &encrypted_y & 0x80u8;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u8);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x08u8))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 3u8;
                    diff_exp.min(7u8)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros_16 = FheUint16::cast_from(op_mant.leading_zeros());
    let leading_zeros = FheUint8::cast_from(leading_zeros_16);

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(3u8);
            let mant = (&op_mant & 0x0Eu8) >> 1u8;
            let res_exp = &x_exp + 0x08u8;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 4u8;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 3u8;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
    );

    overflow.select(&ov_result, &result)
}

/* CPU-oriented integer-based addition for 16 bits floating points */
pub fn fhe_add16_cpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_mask: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: ServerKey,
) -> FheUint16 {
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let ns_a = &encrypted_a & 0x7FFFu16;
    let ns_b = &encrypted_b & 0x7FFFu16;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7C00u16;
            let mant = (&encrypted_x & 0x03FFu16) | 0x0400u16;
            let sign = &encrypted_x & 0x8000u16;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7C00u16;
            let mant = &encrypted_y & 0x03FFu16;
            let sign = &encrypted_y & 0x8000u16;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u16);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0400u16))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 10u16;
                    diff_exp.min(15u16)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint16::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(4u16);
            let mant = (&op_mant & 0x07FEu16) >> 1u16;
            let res_exp = &x_exp + 0x0400u16;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 5u16;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 10u16;
            let denorm = sub_exp.gt(&x_exp);
            let res_exp = &x_exp - &sub_exp;
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
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
    let ns_a = &encrypted_a & 0x7FFF_FFFFu32;
    let ns_b = &encrypted_b & 0x7FFF_FFFFu32;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7F80_0000u32;
            let mant = (&encrypted_x & 0x007F_FFFFu32) | 0x0080_0000u32;
            let sign = &encrypted_x & 0x8000_0000u32;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7F80_0000u32;
            let mant = &encrypted_y & 0x007F_FFFFu32;
            let sign = &encrypted_y & 0x8000_0000u32;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u32);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0080_0000u32))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 23u16;
                    diff_exp.min(31u16)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;

    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = op_mant.leading_zeros();

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.clone().eq(7u32);
            let mant = (&op_mant & 0x00FF_FFFEu32) >> 1u16;
            let res_exp = &x_exp + 0x0080_0000u32;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 8u32;

            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 23u16;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);
            let result = &x_sign | &final_exp | &mant;
            result
        },
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

    let ns_a = &encrypted_a & 0x7FFF_FFFF_FFFF_FFFFu64;
    let ns_b = &encrypted_b & 0x7FFF_FFFF_FFFF_FFFFu64;
    let ab_cmp = ns_a.ge(&ns_b);

    let (encrypted_x, encrypted_y) = rayon::join(
        || ab_cmp.select(&encrypted_a, &encrypted_b),
        || ab_cmp.select(&encrypted_b, &encrypted_a),
    );

    let ((x_exp, x_mant, x_sign), (y_exp, y_mant_raw, y_sign)) = rayon::join(
        || {
            let exp = &encrypted_x & 0x7FF0_0000_0000_0000u64;
            let mant = (&encrypted_x & 0x000F_FFFF_FFFF_FFFFu64) | 0x0010_0000_0000_0000u64;
            let sign = &encrypted_x & 0x8000_0000_0000_0000u64;
            (exp, mant, sign)
        },
        || {
            let exp = &encrypted_y & 0x7FF0_0000_0000_0000u64;
            let mant = &encrypted_y & 0x000F_FFFF_FFFF_FFFFu64;
            let sign = &encrypted_y & 0x8000_0000_0000_0000u64;
            (exp, mant, sign)
        },
    );

    let (same_sign, (y_mant, clipped_diff_exp)) = rayon::join(
        || x_sign.eq(&y_sign),
        || {
            rayon::join(
                || {
                    let denorm_y = y_exp.eq(0u64);
                    denorm_y.select(&y_mant_raw, &(&y_mant_raw | 0x0010_0000_0000_0000u64))
                },
                || {
                    let diff_exp = (&x_exp - &y_exp) >> 52u64;
                    diff_exp.min(1023u64)
                },
            )
        },
    );

    let shifted_y = &y_mant >> &clipped_diff_exp;
    let (sum_mant, diff_mant) = rayon::join(|| &x_mant + &shifted_y, || &x_mant - &shifted_y);

    let op_mant = same_sign.select(&sum_mant, &diff_mant);

    let leading_zeros = FheUint64::cast_from(op_mant.leading_zeros());

    let ((ov_result, overflow), result) = rayon::join(
        || {
            let overflow = leading_zeros.eq(10u64);
            let mant = (&op_mant & 0x001F_FFFF_FFFF_FFFEu64) >> 1u64;
            let res_exp = &x_exp + 0x0010_0000_0000_0000u64;
            let result = &x_sign | &res_exp | &mant;
            (result, overflow)
        },
        || {
            let diff = &leading_zeros - 11u64;
            let shifted_mant = &op_mant << &diff;
            let mant = &shifted_mant & &encrypted_mask;

            let sub_exp = &diff << 52u64;
            let res_exp = &x_exp - &sub_exp;
            let denorm = res_exp.gt(&x_exp);
            let final_exp = denorm.select(&encrypted_zero, &res_exp);

            &x_sign | &final_exp | &mant
        },
    );

    overflow.select(&ov_result, &result)
}
