use half::f16;

/* EXACT INTEGER-BASED OPERATIONS FP-32 */

/* 
pub fn add32 (
    mut a: u32,
    mut b: u32
 ) -> u32 {

    let abs_a = a & 0x7FFF_FFFF;
    let abs_b = b & 0x7FFF_FFFF;

    if abs_b > abs_a {
        std::mem::swap(&mut a, &mut b);
    }

    if abs_b == 0 { return a; }

    let sign_a = a & 0x8000_0000;
    let sign_b = b & 0x8000_0000;
    
    let exp_a = (a >> 23) & 0xFF;
    let exp_b = (b >> 23) & 0xFF;

    let mant_a = (a & 0x007F_FFFF) | 0x0080_0000;
    let mut mant_b = (b & 0x007F_FFFF) | 0x0080_0000;

    let align = exp_a - exp_b;
    if align > 24 {
        return a; 
    }
    mant_b >>= align;

    let mant_res = if sign_a == sign_b {
        mant_a + mant_b
    } else {
        mant_a - mant_b
    };

    if mant_res == 0 {
        return 0; 
    }

    let lz = mant_res.leading_zeros();
    let shift = 8i32 - lz as i32;

    let (result_mant, new_exp) = if shift > 0 {
        let shift_u = shift as u32;
        ((mant_res >> shift_u) & 0x007F_FFFF, exp_a + shift_u)
    } else if shift < 0 {
        let shift_u = (-shift) as u32;
        if shift_u >= exp_a {
            return sign_a; 
        }
        ((mant_res << shift_u) & 0x007F_FFFF, exp_a - shift_u)
    } else {
        (mant_res & 0x007F_FFFF, exp_a)
    };

    if new_exp >= 255 {
        return sign_a | 0x7F80_0000; 
    }

    sign_a | (new_exp << 23) | result_mant
}
*/

pub fn add32 (
    a: u32,
    b: u32
 ) -> u32 {

    let ns_a = &a & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let ns_b = &b & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let (x, y);

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

    let (x_mant, mut y_mant);
    y_mant = &y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    if y_exp != 0 {
        y_mant |= 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
    }
    let mut diff_exp = (&x_exp - &y_exp) >> 23u32;
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

    let sum_mant;
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
        let mant = (sum_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u32;
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

pub fn same_sign_add32(
    a: u32,
    b: u32,
) -> u32 {

    let ns_a = &a & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let ns_b = &b & 0b0111_1111_1111_1111_1111_1111_1111_1111u32;
    let (x, y);

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

    let (x_mant, mut y_mant);
    y_mant = &y & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    if y_exp != 0 {
        y_mant |= 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
    }
    let mut diff_exp = (&x_exp - &y_exp) >> 23u32;
    if diff_exp > 31u32 {
        diff_exp = 31u32;
    }

    x_mant = (&x & 0b0000_0000_0111_1111_1111_1111_1111_1111u32) | 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
    let x_sign = &x & 0b1000_0000_0000_0000_0000_0000_0000_0000u32;

    let sum_mant = &x_mant + (&y_mant >> &diff_exp);

    let leading_zeros = sum_mant.leading_zeros() as u32;

    let overflow = leading_zeros == 7u32;

    let encrypted_mask = 8388607u32;

    let (ov_result, overflow_flag, result) = if overflow {
        // true branch (equivalent to first closure)
        let mant = (sum_mant & 0b0000_0000_1111_1111_1111_1111_1111_1110u32) >> 1u32;
        let res_exp = x_exp + 0b0000_0000_1000_0000_0000_0000_0000_0000u32;
        let result = x_sign | res_exp | mant;
        (result, overflow, result) 
    } else {
        let mant = sum_mant & encrypted_mask;
        let result = x_sign | x_exp | mant;
        (result, overflow, result)
    };

    if overflow_flag {
        ov_result
    } else {
        result
    }
}

pub fn sub32 (
    a: u32,
    b: u32,
) -> u32 {
    let b_negate = &b ^ 0b1000_0000_0000_0000_0000_0000_0000_0000u32;
    add32(a, b_negate)
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

static SQRT_BITS_U32: [u32; 24] = [
    8388608u32, 2097152u32, 524288u32, 131072u32, 32768u32, 8192u32, 2048u32, 512u32, 128u32, 32u32, 8u32, 2u32, 0u32, 0u32, 0u32, 0u32,
    0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32
];


pub fn sqrt32(
    a: u32
) -> u32 {
    let sign = &a >> 31u32;
    if sign == 1u32 {
        return 0u32; // Return 0 for negative numbers
    }
    let exp = ((&a & 0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32) - 127u32;
    let mut mantissa = &a & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    mantissa += 8388608u32;
    if exp % 2 == 1 {
        mantissa <<= 1; 
    }
    let mut res_mantissa = 0u32;
    let res_exp = (exp >> 1u32) + 127u32;
    let mut result = 0u32;
    let mut x = mantissa;
    for i in 0..24 {
        if x >= (result + SQRT_BITS_U32[i]) {
            res_mantissa |= 1u32 << (23 - i);
            x -= result + SQRT_BITS_U32[i];
            result = (result >> 1u32) + SQRT_BITS_U32[i];
        }
        else {
            result = result >> 1u32;
        }
    }
    res_mantissa &= 0b0000_0000_0111_1111_1111_1111_1111_1111u32; // Mask to keep only the mantissa bits
    (0u32 << 31u32) | (res_exp << 23u32) | res_mantissa
}

#[allow(dead_code)]
static EXP_TO_RESULT_U32: [u32; 128] = [
    0b00111111100000000000000000000000u32, 0b00111111100000000000000000000000u32, 0b01000000000000000000000000000000u32, 0b01000000000000000000000000000000u32, 
    0b01000000100000000000000000000000u32, 0b01000000100000000000000000000000u32, 0b01000000100000000000000000000000u32, 0b01000000100000000000000000000000u32,
    0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32,
    0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32, 0b01000001000000000000000000000000u32,
    0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32,
    0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32,
    0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32,
    0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32, 0b01000001100000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32, 0b01000010000000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
    0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32, 0b01000010100000000000000000000000u32,
];

#[allow(dead_code)]
pub fn log2_u32(
    a: u32
) -> u32 {
    if a >= 2147483648u32 || a < 8388608u32 {
        return 0u32;
    }
    let mut exp_value = (&a &  0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32;
    if exp_value < 127u32 {
        exp_value = 127u32 - exp_value;
    }
    else {
        exp_value = exp_value - 127u32;
    }
    let mut sign = false;
    let mut result = EXP_TO_RESULT_U32[exp_value as usize];
    if a < 1065353216u32 {
        result |= 2147483648u32; //negative sign of log2
        sign = true;
    }
    let mut log_mant = 0u32;
    let mut mant = &a & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    mant += 0b0000_0000_1000_0000_0000_0000_0000_0000u32; // add 1 to mantissa
    for i in 0..23 {
        let squared_shifted = ((mant as u64) * (mant as u64) >> 23u32) as u32;
        if squared_shifted >= 16777216u32 {
            log_mant |= 1u32 << (22 - i);
            mant = squared_shifted >> 1u32;
        }
        else{
            mant = squared_shifted;
        }
    }
    let mut sum;
    if sign {
        sum = (exp_value<<23u32) - log_mant;
    }
    else {
        sum = (exp_value<<23u32) | log_mant;
    }
    let mut res_exp_value = (sum & 0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32;
    let mut diff_exp = 0u32;
    if res_exp_value < exp_value {
        if res_exp_value == 0u32 {
            while res_exp_value == 0u32 {
                sum = sum << 1u32;
                res_exp_value = sum & 0b0111_1111_1000_0000_0000_0000_0000_0000u32;
                diff_exp += 1u32;
            }
        }
        else
        {    
            diff_exp = exp_value.ilog2() - res_exp_value.ilog2();
        }
    }
    else {
        if res_exp_value == 0u32 {
            while sum & 0b0111_1111_1000_0000_0000_0000_0000_0000u32 == 0u32 {
                sum = sum << 1u32;
                diff_exp += 1u32;
            }
        }
    }
    let ilog2 = sum.ilog2();
    let mut result_mant = sum & (0b0000_0000_0111_1111_1111_1111_1111_1111u32 << (ilog2 - 23u32));
    result_mant >>= ilog2 - 23u32;
    result -= diff_exp << 23u32;
    result |= result_mant;
    result
}

/* EXACT INTEGER-BASED OPERATIONS FP-16 */

pub fn add16 (
    a: u16,
    b: u16
 ) -> u16 {

    let ns_a = &a & 0b0111_1111_1111_1111u16;
    let ns_b = &b & 0b0111_1111_1111_1111u16;
    let (x, y);

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

    let (x_mant, mut y_mant);
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

    let sum_mant;
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
        if sub_exp > x_exp{
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
    let (x, y);

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

    let (x_mant, mut y_mant);
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

    let sum_mant = &x_mant + (&y_mant >> &diff_exp);

    let leading_zeros = sum_mant.leading_zeros() as u16;

    let overflow = leading_zeros == 4u16;

    let encrypted_mask = 1023u16;

    let (ov_result, overflow_flag, result) = if overflow {
        
        let mant = (sum_mant & 0b0000_0111_1111_1110u16) >> 1u16;
        let res_exp = x_exp + 1024u16;
        let result = x_sign | res_exp | mant;
        (result, overflow, result) 
    } else {
        let mant = sum_mant & encrypted_mask;
        let result = x_sign | x_exp | mant;
        (result, overflow, result)
    };

    if overflow_flag {
        ov_result
    } else {
        result
    }
}

pub fn sub16 (
    a: u16,
    b: u16,
) -> u16 {
    let b_negate = &b ^ 0b1000_0000_0000_0000u16;
    add16(a, b_negate)
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

#[allow(dead_code)]
static EXP_TO_RESULT_U16: [u16; 16] = [
    15360u16, 15360u16, 16384u16, 16384u16, 17408u16, 17408u16, 17408u16, 17408u16, 18432u16, 18432u16, 18432u16, 18432u16, 18432u16, 18432u16, 18432u16, 18432u16
];

#[allow(dead_code)]
pub fn log2_u16(
    a: u16
) -> u16 {
    if a >= 32768u16 || a < 1024u16 {
        return 0u16;
    }
    let mut exp_value = (&a &  31744u16) >> 10u16;
    if exp_value < 15u16 {
        exp_value = 15u16 - exp_value;
    }
    else {
        exp_value = exp_value - 15u16;
    }
    let mut sign = false;
    let mut result = EXP_TO_RESULT_U16[exp_value as usize];
    if a < 15360u16 {
        result |= 32768u16; //negative sign of log2
        sign = true;
    }
    let mut log_mant = 0u16;
    let mut mant = &a & 0b0000_0011_1111_1111u16;
    mant += 1024u16; // add 1 to mantissa
    for i in 0..10 {
        let squared_shifted = (((mant as u32) * (mant as u32)) >> 10u32) as u16;
        if squared_shifted >= 2048u16 {
            log_mant |= 1u16 << (9 - i);
            mant = squared_shifted >> 1u16;
        }
        else{
            mant = squared_shifted;
        }
    }
    let mut sum;
    if sign {
        sum = (exp_value<<10u16) - log_mant;
    }
    else {
        sum = (exp_value<<10u16) | log_mant;
    }
    let mut res_exp_value = (sum & 0b0111_1100_0000_0000u16) >> 10u16;
    let mut diff_exp = 0u16;
    if res_exp_value < exp_value {
        if res_exp_value == 0u16 {
            while res_exp_value == 0u16 {
                sum = sum << 1u16;
                res_exp_value = sum & 0b0111_1100_0000_0000u16;
                diff_exp += 1u16;
            }
        }
        else
        {    
            diff_exp = (exp_value.ilog2() - res_exp_value.ilog2()) as u16;
        }
    }
    else {
        if res_exp_value == 0u16 {
            while sum & 0b0111_1100_0000_0000u16 == 0u16 {
                sum = sum << 1u16;
                diff_exp += 1u16;
            }
        }
    }
    let ilog2 = sum.ilog2() as u16;
    let mut result_mant = sum & (0b0000_0011_1111_1111u16 << (ilog2 - 10u16));
    result_mant >>= ilog2 - 10u16;
    result -= diff_exp << 10u16;
    result |= result_mant;
    result
}

static SQRT_BITS_U16: [u16; 11] = [
    1024u16, 256u16, 64u16, 16u16, 4u16, 1u16, 0u16, 0u16, 0u16, 0u16, 0u16
];

pub fn sqrt16(
    a: u16
) -> u16 {
    let sign = &a >> 15u16;
    if sign == 1u16 {
        return 0u16; // Return 0 for negative numbers
    }
    if a < 1024u16 {
        return 0u16; // Return 0 for numbers less than 1
    }
    let exp = ((&a & 0b0111_1100_0000_0000u16) >> 10u16) - 15u16;
    let mut mantissa = &a & 0b0000_0011_1111_1111u16;
    mantissa += 1024u16; 
    if exp % 2 == 1 {
        mantissa <<= 1; 
    }
    let mut res_mantissa = 0u16;
    let res_exp = (exp >> 1u16) + 15u16;
    let mut result = 0u16;
    let mut x = mantissa;
    for i in 0..11 {
        if x >= (result + SQRT_BITS_U16[i]) {
            res_mantissa |= 1u16 << (10 - i);
            x -= result + SQRT_BITS_U16[i];
            result = (result >> 1u16) + SQRT_BITS_U16[i];
        }
        else {
            result = result >> 1u16;
        }
    }
    res_mantissa &= 0b0000_0011_1111_1111u16; // Mask to keep only the mantissa bits
    (0u16 << 15u16) | (res_exp << 10u16) | res_mantissa
}

/* APPROX INTEGER-BASED OPERATIONS FP-32 ACCORDING TO LMUL ALGORITHM */

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

    if exp < 127u32 || a == 0u32 || b == 0u32 {
        return 0u32;
    }
    
    let a_digits = &a & 2147483647u32;
    let b_digits = &b & 2147483647u32;
    let mut digits = &a_digits + &b_digits;
    digits -= 1064828928u32;
    digits &= 2147483647u32;

    return digits | sign
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

pub fn lmul_tanh32 (
    a: u32
) -> (u32, u32) {

    let (out, dx) = match a {
        3221225472u32..=4294967295u32 => {
            (3212836864u32, 0u32)
        }, // [-∞, -2]
        3210040661u32..=3221225471u32 => {
            let mul = lmul32(a, 1048576000u32);
            let add = same_sign_add32(mul, 3204448256u32);
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
            let add = same_sign_add32(mul, 1056964608u32);
            (add, 1048576000u32)
        }, // [0.833, 2]
        1073741825u32..=2147483647u32 => {
            (1065353217u32, 0u32)
        }, // [2, ∞]
    };
    (out, dx)
}

/* APPROX INTEGER-BASED OPERATIONS FP-32 ACCORDING TO PAM ALGORITHM */

#[allow(dead_code)]
pub fn pam_mul32 (
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

    if exp < 127u32 || a == 0u32 || b == 0u32 {
        return 0;
    }
    
    let a_digits = &a & 2147483647u32;
    let b_digits = &b & 2147483647u32;
    let mut digits = &a_digits + &b_digits;
    digits -= 1065353216u32;
    digits &= 2147483647u32;

    return digits | sign
}    

#[allow(dead_code)]
pub fn pam_div32 (
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
    digits += 1065353216u32;
    digits &= 2147483647u32;

    return digits | sign
}

#[allow(dead_code)]
pub fn pam_tanh32 (
    a: u32
) -> (u32, u32) {

    let (out, dx) = match a {
        3221225472u32..=4294967295u32 => {
            (3212836864u32, 0u32)
        }, // [-∞, -2]
        3210040661u32..=3221225471u32 => {
            let mul = pam_mul32(a, 1048576000u32);
            let add = same_sign_add32(mul, 3204448256u32);
            (add, 1048576000u32)
        }, // [-2, -0.8333]
        2147483648u32..=3210040660u32 => {
            let mul = pam_mul32(a, 1062836634u32);
            (mul, 1062836634u32)
        }, // [-0.833, -0]
        0u32..=1062557013u32          => {
            let mul = pam_mul32(a, 1062836634u32);
            (mul, 1062836634u32)
        }, // [0, 0.833]
        1062557014u32..=1073741824u32 => {
            let mul = pam_mul32(a, 1048576000u32);
            let add = same_sign_add32(mul, 1056964608u32);
            (add, 1048576000u32)
        }, // [0.833, 2]
        1073741825u32..=2147483647u32 => {
            (1065353217u32, 0u32)
        }, // [2, ∞]
    };
    (out, dx)
}

/* APPROX INTEGER-BASED OPERATIONS FP-16 ACCORDING TO LMUL ALGORITHM */

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

    if exp < 15u16 || a == 0u16 || b == 0u16 {
        return 0;
    }
    
    let a_digits = &a & 32767u16;
    let b_digits = &b & 32767u16;
    let mut digits = &a_digits + &b_digits;
    digits -= 15296u16;
    digits &= 32767u16;

    return digits | sign
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
    digits &= 32767u16;

    return digits | sign
}

pub fn lmul_tanh16 (
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
    };
    (out, dx)
}

/* APPROX INTEGER-BASED OPERATIONS FP-16 ACCORDING TO PAM ALGORITHM */
#[allow(dead_code)]
pub fn pam_mul16 (
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

    if exp < 15u16 || a == 0u16 || b == 0u16 {
        return 0;
    }
    
    let a_digits = &a & 32767u16;
    let b_digits = &b & 32767u16;
    let mut digits = &a_digits + &b_digits;
    digits -= 15360u16;
    digits &= 32767u16;

    return digits | sign
}
    
#[allow(dead_code)]
pub fn pam_div16 (
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
    digits += 15360u16;
    digits &= 32767u16;

    return digits | sign
}

#[allow(dead_code)]
pub fn pam_tanh16 (
    a: u16
) -> (u16, u16) {
    let (out, dx) = match a {
        49152u16..=65535u16 => {
            (48128u16, 0u16)
        }, // [-∞, -2]
        47787u16..=49151u16 => {
            let mul = pam_mul16(a, 13312u16);
            let add = same_sign_add16(mul, 47104u16);
            (add, 13312u16)
        }, // [-2, -0.8333]
        32768u16..=47786u16 => {
            let mul = pam_mul16(a, 15053u16);
            (mul, 15053u16)
        }, // [-0.833, -0]
        0u16..=15019u16         => {
            let mul = pam_mul16(a, 15053u16);
            (mul, 15053u16)
        }, // [0, 0.833]
        15020u16..=16384u16 => {
            let mul = pam_mul16(a, 13312u16);
            let add = same_sign_add16(mul, 14336u16);
            (add, 13312u16)
        }, // [0.833, 2]
        16385u16..=32767u16 => {
            (15360u16, 0u16)
        }, // [2, ∞]
    };
    (out, dx)
}

/* CANONICAL OPERATIONS */

#[allow(dead_code)]
 pub fn canonical_tanh16 (
    a: u16
) -> (u16, u16) {
    let f_a = f16::from_bits(a).to_f32();
    let tanh_value = f16::from_f32(f_a.tanh());
    let derivative = f16::from_f32_const(1.0) - tanh_value * tanh_value;
    (tanh_value.to_bits(), derivative.to_bits())
}

#[allow(dead_code)]
pub fn canonical_tanh32 (
    a: u32
) -> (u32, u32) {
    let f_a = f32::from_bits(a);
    let tanh_value = f_a.tanh();
    let derivative = 1.0 - tanh_value * tanh_value;
    (tanh_value.to_bits(), derivative.to_bits())
}

#[allow(dead_code)]
pub fn canonical_sub32 (
    a: u32,
    b: u32, 
) -> u32 {
    (f32::from_bits(a) - f32::from_bits(b)).to_bits()
}

#[allow(dead_code)]
pub fn canonical_div32 (
    a: u32,
    b: u32,
) -> u32 {
    (f32::from_bits(a) / f32::from_bits(b)).to_bits()
}

#[allow(dead_code)]
pub fn canonical_mul32 (
    a: u32,
    b: u32,
 ) -> u32 {
    (f32::from_bits(a) * f32::from_bits(b)).to_bits()
 }

 #[allow(dead_code)]
 pub fn canonical_add32 (
    a: u32,
    b: u32
) -> u32 {
    (f32::from_bits(a) + f32::from_bits(b)).to_bits()
}

#[allow(dead_code)]
pub fn canonical_sqrt32(
    a: u32
) -> u32 {
    (f32::from_bits(a).sqrt()).to_bits()
}

#[allow(dead_code)]
pub fn canonical_add16 (
    a: u16,
    b: u16
) -> u16 {
    (f16::from_bits(a) + f16::from_bits(b)).to_bits()
}

#[allow(dead_code)]
 pub fn canonical_sub16 (
    a: u16,
    b: u16, 
) -> u16 {
    (f16::from_bits(a) - f16::from_bits(b)).to_bits()
}

#[allow(dead_code)]
pub fn canonical_div16 (
    a: u16,
    b: u16,
) -> u16 {
    (f16::from_bits(a) / f16::from_bits(b)).to_bits()
}

#[allow(dead_code)]
pub fn canonical_mul16 (
    a: u16,
    b: u16,
 ) -> u16 {
    (f16::from_bits(a) * f16::from_bits(b)).to_bits()
 }

