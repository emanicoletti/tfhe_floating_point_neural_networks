mod tfhe_nn_builder;
mod plain_nn_builder;
mod experiment_1_2;
mod experiment_3;
mod ops_playground;

#[allow(unused_imports)]
use crate::experiment_1_2::exp_1_settings::*;
#[allow(unused_imports)]
use crate::experiment_1_2::exp_2_settings::*;
#[allow(unused_imports)]
use crate::experiment_3::exp_3_settings::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    //test_encrypted_ops("lmul", 32, false, 1, -5.0, 5.0)?;
    experiment_1_fp32(true, false, true)?;

    Ok(())
}






