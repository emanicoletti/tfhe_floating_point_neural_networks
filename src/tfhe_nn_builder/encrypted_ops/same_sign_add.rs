use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

pub fn fhe_ss_add16_gpu(
    encrypted_a: FheUint16,
    encrypted_b: FheUint16,
    encrypted_mask: FheUint16,
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
    let (y_mant, ((x_exp, diff_exp), (x_mant, x_sign))) = rayon::join(
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
                    (x_mant, x_sign)
                },
            )
        },
    );
    
    let op_mant = &x_mant + (&y_mant >> &diff_exp);

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
            let mant = (&op_mant & &encrypted_mask);
            let result = &x_sign | &x_exp | &mant;
            result
        }
    );

    overflow.select(&ov_result, &result)
}