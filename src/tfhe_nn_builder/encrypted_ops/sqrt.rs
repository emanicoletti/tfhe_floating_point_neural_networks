use tfhe::prelude::*;
use tfhe::{CudaServerKey, FheUint16, FheUint32, ServerKey, set_server_key};

static SQRT_BITS_U16: [u16; 11] = [
    1024u16, 256u16, 64u16, 16u16, 4u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16,
];

static SQRT_BITS_U32: [u32; 24] = [
    8388608u32, 2097152u32, 524288u32, 131072u32, 32768u32, 8192u32, 2048u32, 512u32, 128u32,
    32u32, 8u32, 2u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
];

pub fn fhe_sqrt16_gpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (neg_denorm, (res_exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 15u16;
            let negative_sign = sign.eq(1u16);
            let denorm = encrypted_a.lt(1024u16); // Safely catches exp == 0
            negative_sign | denorm
        },
        || {
            let raw_exp = (&encrypted_a & 0x7C00u16) >> 10u16;

            // MATH OPTIMIZATION: Safe Exponent Halving for FP16 (Bias 15)
            let exp_lsb = &raw_exp & 1u16;
            let res_exp = (&raw_exp >> 1u16) + 7u16 + &exp_lsb;

            let mut mantissa = (&encrypted_a & 0x03FFu16) | 0x0400u16;

            let exp_is_even = exp_lsb.eq(0u16);
            mantissa = exp_is_even.select(&(&mantissa << 1u16), &mantissa);

            (res_exp, mantissa)
        },
    );

    let mut res_mantissa = encrypted_zero.clone();
    let mut result = encrypted_zero.clone();
    let mut x = mantissa;

    for i in 0..11 {
        // Step 1: Compute independent base values
        let (res_plus_bit, lt_result) =
            rayon::join(|| &result + SQRT_BITS_U16[i], || &result >> 1u16);

        // Step 2: Compute dependent comparison and subtractions/additions
        let ((grt, ge_x), ge_result) = rayon::join(
            || rayon::join(|| x.ge(&res_plus_bit), || &x - &res_plus_bit),
            || &lt_result + SQRT_BITS_U16[i],
        );

        // Step 3: Multiplex the results for the next iteration
        let (new_x, (new_result, new_res_mantissa)) = rayon::join(
            || grt.select(&ge_x, &x),
            || {
                rayon::join(
                    || grt.select(&ge_result, &lt_result),
                    || {
                        let ge_mantissa = &res_mantissa | (1u16 << (10 - i));
                        grt.select(&ge_mantissa, &res_mantissa)
                    },
                )
            },
        );

        x = new_x;
        result = new_result;
        res_mantissa = new_res_mantissa;
    }

    res_mantissa = &res_mantissa & 0x03FFu16;
    let pos_sign_result = (res_exp << 10u16) | res_mantissa;

    neg_denorm.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt32_gpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (negative_sign, (res_exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 31u32;
            sign.eq(1u32)
        },
        || {
            let raw_exp = (&encrypted_a & 0x7F80_0000u32) >> 23u32;

            let exp_lsb = &raw_exp & 1u32;
            let res_exp = (&raw_exp >> 1u32) + 63u32 + &exp_lsb;

            let mut mantissa = (&encrypted_a & 0x007F_FFFFu32) | 0x0080_0000u32;

            let exp_is_even = exp_lsb.eq(0u32);
            mantissa = exp_is_even.select(&(&mantissa << 1u32), &mantissa);

            (res_exp, mantissa)
        },
    );

    let mut res_mantissa = encrypted_zero.clone();
    let mut result = encrypted_zero.clone();
    let mut x = mantissa;

    for i in 0..24 {
        let (res_plus_bit, lt_result) =
            rayon::join(|| &result + SQRT_BITS_U32[i], || &result >> 1u32);

        let ((grt, ge_x), ge_result) = rayon::join(
            || rayon::join(|| x.ge(&res_plus_bit), || &x - &res_plus_bit),
            || &lt_result + SQRT_BITS_U32[i],
        );

        let (new_x, (new_result, new_res_mantissa)) = rayon::join(
            || grt.select(&ge_x, &x),
            || {
                rayon::join(
                    || grt.select(&ge_result, &lt_result),
                    || {
                        let ge_mantissa = &res_mantissa | (1u32 << (23 - i));
                        grt.select(&ge_mantissa, &res_mantissa)
                    },
                )
            },
        );

        x = new_x;
        result = new_result;
        res_mantissa = new_res_mantissa;
    }

    res_mantissa = &res_mantissa & 0x007F_FFFFu32;
    let pos_sign_result = (res_exp << 23u32) | res_mantissa;

    negative_sign.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt16_cpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: ServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (neg_denorm, (res_exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 15u16;
            let negative_sign = sign.eq(1u16);
            let denorm = encrypted_a.lt(1024u16);
            negative_sign | denorm
        },
        || {
            let raw_exp = (&encrypted_a & 0x7C00u16) >> 10u16;

            let exp_lsb = &raw_exp & 1u16;
            let res_exp = (&raw_exp >> 1u16) + 7u16 + &exp_lsb;

            let mut mantissa = (&encrypted_a & 0x03FFu16) | 0x0400u16;

            let exp_is_even = exp_lsb.eq(0u16);
            mantissa = exp_is_even.select(&(&mantissa << 1u16), &mantissa);

            (res_exp, mantissa)
        },
    );

    let mut res_mantissa = encrypted_zero.clone();
    let mut result = encrypted_zero.clone();
    let mut x = mantissa;

    for i in 0..11 {
        let (res_plus_bit, lt_result) =
            rayon::join(|| &result + SQRT_BITS_U16[i], || &result >> 1u16);

        let ((grt, ge_x), ge_result) = rayon::join(
            || rayon::join(|| x.ge(&res_plus_bit), || &x - &res_plus_bit),
            || &lt_result + SQRT_BITS_U16[i],
        );

        let (new_x, (new_result, new_res_mantissa)) = rayon::join(
            || grt.select(&ge_x, &x),
            || {
                rayon::join(
                    || grt.select(&ge_result, &lt_result),
                    || {
                        let ge_mantissa = &res_mantissa | (1u16 << (10 - i));
                        grt.select(&ge_mantissa, &res_mantissa)
                    },
                )
            },
        );

        x = new_x;
        result = new_result;
        res_mantissa = new_res_mantissa;
    }

    res_mantissa = &res_mantissa & 0x03FFu16;
    let pos_sign_result = (res_exp << 10u16) | res_mantissa;

    neg_denorm.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt32_cpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: ServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (negative_sign, (res_exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 31u32;
            sign.eq(1u32)
        },
        || {
            let raw_exp = (&encrypted_a & 0x7F80_0000u32) >> 23u32;

            let exp_lsb = &raw_exp & 1u32;
            let res_exp = (&raw_exp >> 1u32) + 63u32 + &exp_lsb;

            let mut mantissa = (&encrypted_a & 0x007F_FFFFu32) | 0x0080_0000u32;

            let exp_is_even = exp_lsb.eq(0u32);
            mantissa = exp_is_even.select(&(&mantissa << 1u32), &mantissa);

            (res_exp, mantissa)
        },
    );

    let mut res_mantissa = encrypted_zero.clone();
    let mut result = encrypted_zero.clone();
    let mut x = mantissa;

    for i in 0..24 {
        let (res_plus_bit, lt_result) =
            rayon::join(|| &result + SQRT_BITS_U32[i], || &result >> 1u32);

        let ((grt, ge_x), ge_result) = rayon::join(
            || rayon::join(|| x.ge(&res_plus_bit), || &x - &res_plus_bit),
            || &lt_result + SQRT_BITS_U32[i],
        );

        let (new_x, (new_result, new_res_mantissa)) = rayon::join(
            || grt.select(&ge_x, &x),
            || {
                rayon::join(
                    || grt.select(&ge_result, &lt_result),
                    || {
                        let ge_mantissa = &res_mantissa | (1u32 << (23 - i));
                        grt.select(&ge_mantissa, &res_mantissa)
                    },
                )
            },
        );

        x = new_x;
        result = new_result;
        res_mantissa = new_res_mantissa;
    }

    res_mantissa = &res_mantissa & 0x007F_FFFFu32;
    let pos_sign_result = (res_exp << 23u32) | res_mantissa;

    negative_sign.select(&encrypted_zero, &pos_sign_result)
}
