use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint16, FheUint32, ServerKey, CudaServerKey};

static SQRT_BITS_U16: [u16; 11] = [
    1024u16, 256u16, 64u16, 16u16, 4u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16
];

static SQRT_BITS_U32: [u32; 24] = [
    8388608u32, 2097152u32, 524288u32, 131072u32, 32768u32, 8192u32, 2048u32, 512u32, 128u32, 32u32, 8u32, 2u32, 0u32, 0u32, 0u32, 0u32,
    0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32
];

pub fn fhe_sqrt16_gpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (neg_denorm,(exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 15u16;
            let negative_sign = sign.eq(1u16);
            let denorm = encrypted_a.lt(1024u16);
            negative_sign | denorm
        },
        ||{
            let exp = ((&encrypted_a & 0b0111_1111_1000_0000u16) >> 10u16) - 15u16;
            let mut mantissa = &encrypted_a & 0b0000_0011_1111_1111u16;
            mantissa += 1024u16;
            let exp_odd = exp.is_odd();
            mantissa = exp_odd.select(
                &(&mantissa << 1u16),
                &mantissa,
            );
            (exp, mantissa)
        },
    );
    let mut res_mantissa = encrypted_zero.clone();
    let res_exp = (&exp >> 1u16) + 15u16;
    let mut result = encrypted_zero.clone();
    let mut x = mantissa.clone();
    for i in 0..11 {
        let ((grt, lt_result), (ge_result, ge_x)) = rayon::join(
            || {
                let res_plus_bit = result.clone() + SQRT_BITS_U16[i];
                let grt = x.ge(&res_plus_bit);
                let lt_result = result.clone() >> 1u16;
                (grt, lt_result)
            }, 
            || {
                let ge_x = x.clone() - (result.clone() + SQRT_BITS_U16[i]);
                let ge_result = (result.clone() >> 1u16) + SQRT_BITS_U16[i];
                (ge_result, ge_x)
            }
        );
        (x, res_mantissa) = rayon::join(
            ||{
                x = grt.select(&ge_x, &x);
                result = grt.select(&ge_result, &lt_result);
                x
            }, 
            || {
                let ge_mantissa = res_mantissa.clone() | (1u16 << (10 - i));
                res_mantissa = grt.select(&ge_mantissa, &res_mantissa);
                res_mantissa
            },
        );
    }
    res_mantissa = res_mantissa & 0b0000_0011_1111_1111u16;
    let pos_sign_result = (0u16 << 15u16) | (res_exp << 10u16) | res_mantissa.clone();
    neg_denorm.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt32_gpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (negative_sign,(exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 31u32;
            let negative_sign = sign.eq(1u32);
            negative_sign
        },
        || {
            let exp = ((&encrypted_a & 0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32) - 127u32;
            let mut mantissa = &encrypted_a & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
            mantissa += 8388608u32;
            let exp_odd = exp.is_odd();
            mantissa = exp_odd.select(
                &(&mantissa << 1u32),
                &mantissa,
            );
            (exp, mantissa)
        },
    );
    let mut res_mantissa = encrypted_zero.clone();
    let res_exp = (&exp >> 1u32) + 127u32;
    let mut result = encrypted_zero.clone();
    let mut x = mantissa.clone();

    for i in 0..24 {
        let ((grt, lt_result), (ge_result, ge_x)) = rayon::join(
            ||{
                let res_plus_bit = result.clone() + SQRT_BITS_U32[i];
                let grt = x.ge(&res_plus_bit);
                let lt_result = result.clone() >> 1u32;
                (grt, lt_result)
            },
            ||{
                let ge_x = x.clone() - (result.clone() + SQRT_BITS_U32[i]);
                let ge_result = (result.clone() >> 1u32) + SQRT_BITS_U32[i];
                (ge_result, ge_x)
            },
        );
        (x, res_mantissa) = rayon::join(
            ||{
                x = grt.select(&ge_x, &x);
                result = grt.select(&ge_result, &lt_result);
                x
            },
            ||{
                let ge_mantissa = res_mantissa.clone() | (1u32 << (23 - i));
                res_mantissa = grt.select(&ge_mantissa, &res_mantissa);
                res_mantissa
            },
        );
    }
    res_mantissa = res_mantissa & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    let pos_sign_result = (0u32 << 31u32) | (res_exp << 23u32) | res_mantissa.clone();
    negative_sign.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt16_cpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: ServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (neg_denorm,(exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 15u16;
            let negative_sign = sign.eq(1u16);
            let denorm = encrypted_a.lt(1024u16);
            negative_sign | denorm
        },
        ||{
            let exp = ((&encrypted_a & 0b0111_1111_1000_0000u16) >> 10u16) - 15u16;
            let mut mantissa = &encrypted_a & 0b0000_0011_1111_1111u16;
            mantissa += 1024u16;
            let exp_odd = exp.is_odd();
            mantissa = exp_odd.select(
                &(&mantissa << 1u16),
                &mantissa,
            );
            (exp, mantissa)
        },
    );
    let mut res_mantissa = encrypted_zero.clone();
    let res_exp = (&exp >> 1u16) + 15u16;
    let mut result = encrypted_zero.clone();
    let mut x = mantissa.clone();
    for i in 0..11 {
        let ((grt, lt_result), (ge_result, ge_x)) = rayon::join(
            || {
                let res_plus_bit = result.clone() + SQRT_BITS_U16[i];
                let grt = x.ge(&res_plus_bit);
                let lt_result = result.clone() >> 1u16;
                (grt, lt_result)
            }, 
            || {
                let ge_x = x.clone() - (result.clone() + SQRT_BITS_U16[i]);
                let ge_result = (result.clone() >> 1u16) + SQRT_BITS_U16[i];
                (ge_result, ge_x)
            }
        );
        (x, res_mantissa) = rayon::join(
            ||{
                x = grt.select(&ge_x, &x);
                result = grt.select(&ge_result, &lt_result);
                x
            }, 
            || {
                let ge_mantissa = res_mantissa.clone() | (1u16 << (10 - i));
                res_mantissa = grt.select(&ge_mantissa, &res_mantissa);
                res_mantissa
            },
        );
    }
    res_mantissa = res_mantissa & 0b0000_0011_1111_1111u16;
    let pos_sign_result = (0u16 << 15u16) | (res_exp << 10u16) | res_mantissa.clone();
    neg_denorm.select(&encrypted_zero, &pos_sign_result)
}

pub fn fhe_sqrt32_cpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: ServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let (negative_sign,(exp, mantissa)) = rayon::join(
        || {
            let sign = &encrypted_a >> 31u32;
            let negative_sign = sign.eq(1u32);
            negative_sign
        },
        || {
            let exp = ((&encrypted_a & 0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32) - 127u32;
            let mut mantissa = &encrypted_a & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
            mantissa += 8388608u32;
            let exp_odd = exp.is_odd();
            mantissa = exp_odd.select(
                &(&mantissa << 1u32),
                &mantissa,
            );
            (exp, mantissa)
        },
    );
    let mut res_mantissa = encrypted_zero.clone();
    let res_exp = (&exp >> 1u32) + 127u32;
    let mut result = encrypted_zero.clone();
    let mut x = mantissa.clone();

    for i in 0..24 {
        let ((grt, lt_result), (ge_result, ge_x)) = rayon::join(
            ||{
                let res_plus_bit = result.clone() + SQRT_BITS_U32[i];
                let grt = x.ge(&res_plus_bit);
                let lt_result = result.clone() >> 1u32;
                (grt, lt_result)
            },
            ||{
                let ge_x = x.clone() - (result.clone() + SQRT_BITS_U32[i]);
                let ge_result = (result.clone() >> 1u32) + SQRT_BITS_U32[i];
                (ge_result, ge_x)
            },
        );
        (x, res_mantissa) = rayon::join(
            ||{
                x = grt.select(&ge_x, &x);
                result = grt.select(&ge_result, &lt_result);
                x
            },
            ||{
                let ge_mantissa = res_mantissa.clone() | (1u32 << (23 - i));
                res_mantissa = grt.select(&ge_mantissa, &res_mantissa);
                res_mantissa
            },
        );
    }
    res_mantissa = res_mantissa & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    let pos_sign_result = (0u32 << 31u32) | (res_exp << 23u32) | res_mantissa.clone();
    negative_sign.select(&encrypted_zero, &pos_sign_result)
}