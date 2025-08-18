use tfhe::prelude::*;
use tfhe::{set_server_key, FheUint8, FheUint16, FheUint32, FheUint64, ServerKey, CudaServerKey};

pub fn fhe_log2_32_gpu(
    encrypted_a: FheUint32,
    encrypted_zero: FheUint32,
    server_keys: CudaServerKey,
) -> FheUint32 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));


    let mut exp_value = (&encrypted_a &  0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32;
    let lt_127 = exp_value.lt(127u32);
    let (exp_value_lt, exp_value_gt) = rayon::join(
        ||{
            127u32 - exp_value.clone()
        },
        ||{
            exp_value.clone() - 127u32
        },
    );
    exp_value = lt_127.select(&exp_value_lt, &exp_value_gt);
    let neg_sign = encrypted_a.lt(1065353216u32);
    let mut result = (exp_value.clone() >> 1u32);
    let res_eq_zero = result.eq(0u32);
    result = (result.ilog2() + 128u32) << 23u32;
    result = res_eq_zero.select(&((&encrypted_zero.clone() | 127u32) << 23u32), &result.clone());
    result = neg_sign.select(&(result.clone() | 2147483648u32), &result);


    let mut log_mant = encrypted_zero.clone();
    let mut mant = &encrypted_a & 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    mant += 0b0000_0000_1000_0000_0000_0000_0000_0000u32;

    for i in 0..23 {
        let mant64: FheUint64 = mant.clone().cast_into();
        let squared_shift: FheUint32 = ((mant64.clone() * mant64.clone()) >> 23u64).cast_into();
        let gt = squared_shift.ge(16777216u32);
        let log = log_mant.clone() | (1u32 << (22 - i));
        log_mant = gt.select(&log, &log_mant);
        mant = gt.select(&(squared_shift.clone() >> 1u32), &squared_shift);
    }

    let mut sum = encrypted_zero.clone();
    let (neg, pos) = rayon::join(
        || {
            let neg = (exp_value.clone() << 23u32) - log_mant.clone();
            neg
        },
        || {
            let pos = (exp_value.clone() << 23u32) | log_mant.clone();
            pos
        },
    );
    sum = neg_sign.select(&neg, &pos);
    
    let mut res_exp_value = (sum.clone() & 0b0111_1111_1000_0000_0000_0000_0000_0000u32) >> 23u32;
    let mut diff_exp = encrypted_zero.clone();
    let res_exp_lt_exp = res_exp_value.lt(exp_value.clone());
    let res_exp_eq_zero = res_exp_value.eq(0u32);
    let mut sum_lt = sum.clone();;
    let mut sum_gt = sum.clone();
    let mut diff_exp_lt = encrypted_zero.clone();
    let mut diff_exp_gt = encrypted_zero.clone();

    for i in 0..23 {
        ((sum_lt, res_exp_value, diff_exp_lt), (sum_gt, diff_exp_gt)) = rayon::join(
            ||{
                // LT branch
                let while_res_exp_eq_zero = res_exp_value.eq(0u32);
                sum_lt = while_res_exp_eq_zero.select(&(sum_lt.clone() << 1u32), &sum_lt.clone());
                diff_exp_lt = while_res_exp_eq_zero.select(&(&diff_exp_lt + 1u32),&diff_exp_lt);
                res_exp_value = while_res_exp_eq_zero.select(&(sum_lt.clone() & 0b0111_1111_1000_0000_0000_0000_0000_0000u32), &res_exp_value);
                (sum_lt, res_exp_value, diff_exp_lt)
            },
            ||{
                // GE branch
                let while_sum_eq_zero = (sum_gt.clone() & 0b0111_1111_1000_0000_0000_0000_0000_0000u32).eq(0u32);
                sum_gt = while_sum_eq_zero.select(&(sum_gt.clone() << 1u32), &sum_gt);
                diff_exp_gt = while_sum_eq_zero.select(&(&diff_exp_gt + 1u32),&diff_exp_gt);
                (sum_gt, diff_exp_gt)
            },
        );
    }

    diff_exp_lt = res_exp_eq_zero.select(&diff_exp_lt, &(exp_value.ilog2() - res_exp_value.ilog2()));
    diff_exp = res_exp_lt_exp.select(&diff_exp_lt, &diff_exp_gt);
    sum = res_exp_lt_exp.select(&sum_lt.clone(), &sum_gt);

    let ilog2 = sum.ilog2();
    let mut mask = encrypted_zero.clone();
    mask |= 0b0000_0000_0111_1111_1111_1111_1111_1111u32;
    let mut result_mant = sum & (mask << (ilog2.clone() - 23u32));
    result_mant >>= (ilog2.clone() - 23u32);
    result -= (diff_exp << 23u32);
    result |= result_mant;
    result
    
}

pub fn fhe_log2_16_gpu(
    encrypted_a: FheUint16,
    encrypted_zero: FheUint16,
    server_keys: CudaServerKey,
) -> FheUint16 {
    set_server_key(server_keys.clone());
    rayon::broadcast(|_| set_server_key(server_keys.clone()));

    let denorm = encrypted_a.lt(1024u16);
    let mut exp_value = (&encrypted_a &  0b0111_1100_0000_0000u16) >> 10u16;
    let lt_15 = exp_value.lt(15u16);
    let (exp_value_lt, exp_value_gt) = rayon::join(
        ||{
            15u16 - exp_value.clone()
        },
        ||{
            exp_value.clone() - 15u16
        },
    );
    exp_value = lt_15.select(&exp_value_lt, &exp_value_gt);
    let neg_sign = encrypted_a.lt(15360u32);
    let mut result = (exp_value.clone() >> 1u16);
    let res_eq_zero = result.eq(0u16);
    let ilog_16: FheUint16 = result.ilog2().cast_into();
    result = (ilog_16 + 16u16) << 10u16;
    result = res_eq_zero.select(&((&encrypted_zero.clone() | 15u16) << 10u16), &result.clone());
    result = neg_sign.select(&(result.clone() | 32768u16), &result);


    let mut log_mant = encrypted_zero.clone();
    let mut mant = &encrypted_a & 0b0000_0011_1111_1111u16;
    mant += 0b0000_0100_0000_0000u16;

    for i in 0..10 {
        let mant32: FheUint32 = mant.clone().cast_into();
        let squared_shift: FheUint16 = ((mant32.clone() * mant32.clone()) >> 10u32).cast_into();
        let gt = squared_shift.ge(2048u16);
        let log = log_mant.clone() | (1u16 << (9 - i));
        log_mant = gt.select(&log, &log_mant);
        mant = gt.select(&(squared_shift.clone() >> 1u16), &squared_shift);
    }

    let mut sum = encrypted_zero.clone();
    let (neg, pos) = rayon::join(
        || {
            let neg = (exp_value.clone() << 10u16) - log_mant.clone();
            neg
        },
        || {
            let pos = (exp_value.clone() << 10u16) | log_mant.clone();
            pos
        },
    );
    sum = neg_sign.select(&neg, &pos);
    
    let mut res_exp_value = (sum.clone() & 0b0111_1100_0000_0000u16) >> 10u16;
    let mut diff_exp = encrypted_zero.clone();
    let res_exp_lt_exp = res_exp_value.lt(exp_value.clone());
    let res_exp_eq_zero = res_exp_value.eq(0u16);
    let mut sum_lt = sum.clone();;
    let mut sum_gt = sum.clone();
    let mut diff_exp_lt = encrypted_zero.clone();
    let mut diff_exp_gt = encrypted_zero.clone();

    for i in 0..10 {
        ((sum_lt, res_exp_value, diff_exp_lt), (sum_gt, diff_exp_gt)) = rayon::join(
            ||{
                // LT branch
                let while_res_exp_eq_zero = res_exp_value.eq(0u16);
                sum_lt = while_res_exp_eq_zero.select(&(sum_lt.clone() << 1u16), &sum_lt.clone());
                diff_exp_lt = while_res_exp_eq_zero.select(&(&diff_exp_lt + 1u16),&diff_exp_lt);
                res_exp_value = while_res_exp_eq_zero.select(&(sum_lt.clone() & 0b0111_1100_0000_0000u16), &res_exp_value);
                (sum_lt, res_exp_value, diff_exp_lt)
            },
            ||{
                // GE branch
                let while_sum_eq_zero = (sum_gt.clone() & 0b0111_1100_0000_0000u16).eq(0u16);
                sum_gt = while_sum_eq_zero.select(&(sum_gt.clone() << 1u16), &sum_gt);
                diff_exp_gt = while_sum_eq_zero.select(&(&diff_exp_gt + 1u16),&diff_exp_gt);
                (sum_gt, diff_exp_gt)
            },
        );
    }
    let ilog_diff: FheUint16 = (exp_value.ilog2() - res_exp_value.ilog2()).cast_into();
    diff_exp_lt = res_exp_eq_zero.select(&diff_exp_lt, &ilog_diff);
    diff_exp = res_exp_lt_exp.select(&diff_exp_lt, &diff_exp_gt);
    sum = res_exp_lt_exp.select(&sum_lt.clone(), &sum_gt);

    let ilog2: FheUint16 = sum.ilog2().cast_into();
    let mut mask = encrypted_zero.clone();
    mask |= 0b0000_0011_1111_1111u16;
    let mut result_mant = sum & (mask << (ilog2.clone() - 10u16));
    result_mant >>= (ilog2.clone() - 10u16);
    result -= (diff_exp << 10u16);
    result |= result_mant;
    denorm.select(&encrypted_zero, &result)
}