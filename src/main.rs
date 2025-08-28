mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod ops_playground;

use crate::experiment_1_2::exp_1_settings::*;
use crate::experiment_1_2::exp_2_settings::*;
use crate::experiment_3::exp_3_settings::*;
use crate::tfhe_nn_builder::add;
use crate::tfhe_nn_builder::add::fhe_add16_gpu;
use crate::tfhe_nn_builder::div::fhe_ldiv16_gpu;
use crate::tfhe_nn_builder::mul::fhe_lmul16_gpu;

use rand_distr::num_traits::float;
use tfhe::core_crypto::gpu;
use tfhe::{prelude::*, HlCompressible};
use tfhe::shortint::client_key;
use tfhe::{set_server_key, generate_keys, ConfigBuilder, FheUint8, FheUint16, FheUint32, FheUint64, ClientKey, ServerKey, CompressedServerKey, CudaServerKey};
use std::time::Instant;
use rand::Rng;
use half::f16;
use rayon::prelude::*;
use rayon::{join, scope};
use std::thread;
use tfhe::shortint::parameters::{PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64};

use crate::plain_nn_builder::plain_ops::ops::*;


fn main() -> Result<(), Box<dyn std::error::Error>> {

    experiment_1_fp16(true, true, false, false)?;
    
    Ok(())
}





