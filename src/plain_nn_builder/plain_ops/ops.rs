
pub fn lmul32 (
    a: u32,
    b: u32,
 ) -> u32 {

    let a_sign = &a >> 31u32;
    let b_sign = &b >> 31u32;
    let mut sign = &a_sign ^ b_sign;
    sign <<= 31u32;

    let a_exp = (&a &  2139095040u32) >> 23u32;
    let b_exp = (&b &  2139095040u32) >> 23u32;
    let exp = &a_exp + &b_exp;

    if exp < 127u32 {
        return 0;
    }
    
    let a_digits = &a & 2147483647u32;
    let b_digits = &b & 2147483647u32;
    let mut digits = &a_digits + &b_digits;
    digits -= 1064828928u32;
    digits &= 2147483647u32;

    return digits | sign
}

pub fn add32 (
    a: u32,
    b: u32
 ) -> u32 {

    let ns_a = &a & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let ns_b = &b & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let (mut x, mut y);

    if ns_a >= ns_b {
        x = a;
        y = b;
    }
    else {
        x = b;
        y = a;
    }

    let x_exp = &x & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
    let y_exp = &y & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;

    let (mut x_mant, mut y_mant);
    y_mant = &y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    if y_exp != 0 {
        y_mant |= 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
    }
    let mut diff_exp = (&x_exp - &y_exp) >> 23u16;
    if diff_exp > 31u32 {
        diff_exp = 31u32;
    }

    x_mant = (&x & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
    let x_sign = &x & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
    let y_sign = &y & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
    let mut same_sign = 0u32;
    
    if x_sign == y_sign {
        same_sign = 1u32;
    }

    let mut sum_mant;
    if same_sign == 1 {
        sum_mant = &x_mant + (&y_mant >> &diff_exp);
    }
    else {
        sum_mant = &x_mant - (&y_mant >> &diff_exp);
    }
    
    let leading_zeros = sum_mant.leading_zeros();

    let overflow = leading_zeros == 7u32;

    let encrypted_mask = 8388607u32;

    let (ov_result, overflow_flag, result) = if overflow {
        // true branch (equivalent to first closure)
        let mant = (sum_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u16;
        let res_exp = x_exp + 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
        let result = x_sign | res_exp | mant;
        (result, overflow, result) // The third `result` is same as `ov_result`
    } else {
        let diff = leading_zeros - 8u32;
        let mask = encrypted_mask >> diff;
        let mant = (sum_mant & mask) << diff;
        let sub_exp = diff << 23u32;
        let res_exp = x_exp - sub_exp;
        let result = x_sign | res_exp | mant;
        (result, overflow, result)
    };

    if overflow_flag {
        ov_result
    } else {
        result
    }
}

pub fn ldiv32 (
    a: u32,
    b: u32,
 ) -> u32 {

    let a_sign = &a >> 31u32;
    let b_sign = &b >> 31u32;
    let mut sign = &a_sign ^ b_sign;
    sign <<= 31u32;

    let a_exp = (&a & 2139095040u32) >> 23u32;
    let b_exp = (&b & 2139095040u32) >> 23u32;
    let exp = &a_exp - &b_exp + 127u32;

    if exp > 255u32 || exp == 0u32{
        return 0u32;
    }
    
    let a_digits = &a & 2147483647u32;
    let b_digits = &b & 2147483647u32;
    let mut digits = &a_digits - &b_digits;
    digits += 1064828928u32;
    digits &= 2147483647u32;

    return digits | sign
}

pub fn sub32 (
    a: u32,
    b: u32,
) -> u32 {
    let b_negate = &b ^ 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
    add32(a, b_negate)
}

pub fn tanh32 (
    a: u32
) -> (u32, u32) {

    let (out, dx) = match a {
        3221225472u32..=4294967295u32 => {
            (3212836864u32, 0u32)
        }, // [-∞, -2]
        3210040661u32..=3221225471u32 => {
            let mul = lmul32(a, 1048576000u32);
            let add = add32(mul, 3204448256u32);
            (add, 1048576000u32)
        }, // [-2, -0.8333]
        2147483648u32..=3210040660u32 => {
            let mul = lmul32(a, 1062836634u32);
            (mul, 1062836634u32)
        }, // [-0.833, -0]
        0u32..=1062557013u32          => {
            let mul = lmul32(a, 1062836634u32);
            (mul, 1062836634u32)
        }, // [0, 0.833]
        1062557014u32..=1073741824u32 => {
            let mul = lmul32(a, 1048576000u32);
            let add = add32(mul, 1056964608u32);
            (add, 1048576000u32)
        }, // [0.833, 2]
        1073741825u32..=2147483647u32 => {
            (1065353217u32, 0u32)
        }, // [2, ∞]
        _ => panic!("unexpected input {:?}", a),
    };
    (out, dx)
}



