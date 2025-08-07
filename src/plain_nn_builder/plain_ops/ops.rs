use half::f16;


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
    //digits -= 1065353216u32;
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
    //digits += 1065353216u32;
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

pub fn relu32 (
    a: u32
) -> (u32, u32) {
   if a >= 2147483648u32 {
       (0u32, 0u32)
   } else {
       (a, 1u32)
   }
}

pub fn backward_relu32 (
    a: u32,
    grad_output: u32,
) -> u32 {
    if a == 1u32 {
        grad_output
    } else {
        0u32
    }
}


pub fn lmul16 (
    a: u16,
    b: u16,
 ) -> u16 {

    let a_sign = &a >> 15u16;
    let b_sign = &b >> 15u16;
    let mut sign = &a_sign ^ b_sign;
    sign <<= 15u16;

    let a_exp = (&a &  31744u16) >> 10u32;
    let b_exp = (&b &  31744u16) >> 10u32;
    let exp = &a_exp + &b_exp;

    if exp < 15u16 {
        return 0;
    }
    
    let a_digits = &a & 32767u16;
    let b_digits = &b & 32767u16;
    let mut digits = &a_digits + &b_digits;
    digits -= 15296u16;
    //digits -= 15360u16;
    digits &= 32767u16;

    return digits | sign
}
    

pub fn add16 (
    a: u16,
    b: u16
 ) -> u16 {

    let ns_a = &a & 0b0111_1111_1111_1111u16;
    let ns_b = &b & 0b0111_1111_1111_1111u16;
    let (mut x, mut y);

    if ns_a >= ns_b {
        x = a;
        y = b;
    }
    else {
        x = b;
        y = a;
    }

    let x_exp = &x & 0b0111_1100_0000_0000u16;
    let y_exp = &y & 0b0111_1100_0000_0000u16;

    let (mut x_mant, mut y_mant);
    y_mant = &y & 0b0000_0011_1111_1111u16;
    if y_exp != 0 {
        y_mant |= 0b0000_0100_0000_0000u16;
    }
    let mut diff_exp = (&x_exp - &y_exp) >> 10u16;
    if diff_exp > 15u16 {
        diff_exp = 15u16;
    }

    x_mant = (&x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
    let x_sign = &x & 0b1000_0000_0000_0000u16;
    let y_sign = &y & 0b1000_0000_0000_0000u16;
    let mut same_sign = 0u16;
    
    if x_sign == y_sign {
        same_sign = 1u16;
    }

    let mut sum_mant;
    if same_sign == 1 {
        sum_mant = &x_mant + (&y_mant >> &diff_exp);
    }
    else {
        sum_mant = &x_mant - (&y_mant >> &diff_exp);
    }
    
    let leading_zeros = sum_mant.leading_zeros() as u16;

    let overflow = leading_zeros == 4u16;

    let encrypted_mask = 1023u16;

    let (ov_result, overflow_flag, result) = if overflow {
        // true branch (equivalent to first closure)
        let mant = (sum_mant & 0b0000_0111_1111_1110u16) >> 1u16;
        let res_exp = x_exp + 1024u16;
        let result = x_sign | res_exp | mant;
        (result, overflow, result) // The third `result` is same as `ov_result`
    } else {
        let diff = leading_zeros - 5u16;
        let mask = encrypted_mask >> diff;
        let mant = (sum_mant & mask) << diff;
        let sub_exp: u16 = diff << 10u16;
        let mut res_exp = x_exp - sub_exp;
        if(sub_exp > x_exp){
            res_exp = 0u16;
        }
        let result = x_sign | res_exp | mant;
        (result, overflow, result)
    };

    if overflow_flag {
        ov_result
    } else {
        result
    }
}

pub fn same_sign_add16(
    a: u16,
    b: u16, 
) -> u16 {

    let ns_a = &a & 0b0111_1111_1111_1111u16;
    let ns_b = &b & 0b0111_1111_1111_1111u16;
    let (mut x, mut y);

    if ns_a >= ns_b {
        x = a;
        y = b;
    }
    else {
        x = b;
        y = a;
    }

    let x_exp = &x & 0b0111_1100_0000_0000u16;
    let y_exp = &y & 0b0111_1100_0000_0000u16;

    let (mut x_mant, mut y_mant);
    y_mant = &y & 0b0000_0011_1111_1111u16;
    if y_exp != 0 {
        y_mant |= 0b0000_0100_0000_0000u16;
    }
    let mut diff_exp = (&x_exp - &y_exp) >> 10u16;
    if diff_exp > 15u16 {
        diff_exp = 15u16;
    }

    x_mant = (&x & 0b0000_0011_1111_1111u16) | 0b0000_0100_0000_0000u16;
    let x_sign = &x & 0b1000_0000_0000_0000u16;

    let mut sum_mant = &x_mant + (&y_mant >> &diff_exp);

    let leading_zeros = sum_mant.leading_zeros() as u16;

    let overflow = leading_zeros == 4u16;

    let encrypted_mask = 1023u16;

    let (ov_result, overflow_flag, result) = if overflow {
        // true branch (equivalent to first closure)
        let mant = (sum_mant & 0b0000_0111_1111_1110u16) >> 1u16;
        let res_exp = x_exp + 1024u16;
        let result = x_sign | res_exp | mant;
        (result, overflow, result) 
    } else {
        let mant = (sum_mant & encrypted_mask);
        let result = x_sign | x_exp | mant;
        (result, overflow, result)
    };

    if overflow_flag {
        ov_result
    } else {
        result
    }
}


pub fn ldiv16 (
    a: u16,
    b: u16,
) -> u16 {

    let a_sign = &a >> 15u16;
    let b_sign = &b >> 15u16;
    let mut sign = &a_sign ^ b_sign;
    sign <<= 15u16;

    let a_exp = (&a &  31744u16) >> 10u16;
    let b_exp = (&b &  31744u16) >> 10u16;
    let exp = &a_exp - &b_exp + 15u16;

    if exp > 31u16 || exp == 0u16{
        return 0u16;
    }
    
    let a_digits = &a & 32767u16;
    let b_digits = &b & 32767u16;
    let mut digits = &a_digits - &b_digits;
    digits += 15296u16;
    //digits += 15360u16;
    digits &= 32767u16;

    return digits | sign
}


pub fn sub16 (
    a: u16,
    b: u16,
) -> u16 {
    let b_negate = &b ^ 0b1000_0000_0000_0000u16;
    add16(a, b_negate)
}

pub fn tanh16 (
    a: u16
) -> (u16, u16) {

    let (out, dx) = match a {
        49152u16..=65535u16 => {
            (48128u16, 0u16)
        }, // [-∞, -2]
        47787u16..=49151u16 => {
            let mul = lmul16(a, 13312u16);
            let add = same_sign_add16(mul, 47104u16);
            (add, 13312u16)
        }, // [-2, -0.8333]
        32768u16..=47786u16 => {
            let mul = lmul16(a, 15053u16);
            (mul, 15053u16)
        }, // [-0.833, -0]
        0u16..=15019u16         => {
            let mul = lmul16(a, 15053u16);
            (mul, 15053u16)
        }, // [0, 0.833]
        15020u16..=16384u16 => {
            let mul = lmul16(a, 13312u16);
            let add = same_sign_add16(mul, 14336u16);
            (add, 13312u16)
        }, // [0.833, 2]
        16385u16..=32767u16 => {
            (15360u16, 0u16)
        }, // [2, ∞]
        _ => panic!("unexpected input {:?}", a),
    };
    (out, dx)
}

pub fn relu16 (
    a: u16
) -> (u16, u16) {
   if a >= 32768u16 {
        (0u16, 0u16)
   } else {
       (a, 1u16)
   }
}

pub fn backward_relu16 (
    a: u16,
    grad_output: u16,
) -> u16 {
    if a == 1u16 {
        grad_output
    } else {
        0u16
    }
}
 
/* 
 pub fn tanh16 (
    a: u16
) -> (u16, u16) {
    let f_a = f16::from_bits(a).to_f32();
    let tanh_value = f16::from_f32(f_a.tanh());
    let derivative = f16::from_f32_const(1.0) - tanh_value * tanh_value;
    (tanh_value.to_bits(), derivative.to_bits())
}



pub fn tanh32 (
    a: u32
) -> (u32, u32) {
    let f_a = f32::from_bits(a);
    let tanh_value = f_a.tanh();
    let derivative = 1.0 - tanh_value * tanh_value;
    (tanh_value.to_bits(), derivative.to_bits())
}


 pub fn sub32 (
    a: u32,
    b: u32, 
) -> u32 {
    (f32::from_bits(a) - f32::from_bits(b)).to_bits()
}

pub fn ldiv32 (
    a: u32,
    b: u32,
) -> u32 {
    (f32::from_bits(a) / f32::from_bits(b)).to_bits()
}


pub fn lmul32 (
    a: u32,
    b: u32,
 ) -> u32 {
    (f32::from_bits(a) * f32::from_bits(b)).to_bits()
 }




pub fn add32 (
    a: u32,
    b: u32
) -> u32 {
    (f32::from_bits(a) + f32::from_bits(b)).to_bits()
}
*/